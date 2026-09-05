from __future__ import annotations
"""录制 / 回放 / 对账: 回测与实盘一致性的验收工具

工作流 (实盘上线前的标准验收, 见 kbs/13):
  1. 录制: 实盘进程每收到一根 1m bar, 追加一行到日志
     (append_bar / write_bars_log; 格式 stime,code,open,high,low,close,volume)
  2. 回放: 收盘后把日志喂给内核, 得到逐 bar 信号轨迹与成交流
     (replay_kernel)
  3. 对账: 对比 实盘当日信号 vs 回放信号 (diff_signals), 或对比 内核 vs 参考引擎
     (reconcile) —— 逐 bar 信号全对上 = 还原性有了持续的生产证据。

约定与回测完全一致: stime 为 14 位 YYYYMMDDHHmmss 且升序; 成交 ts 记桶右端点。
"""

import numpy as np

from .kernel import (bars_to_arrays, make_state, resolve_period_seconds,
                     run_backtest, summarize, trades_to_list)
from .models import Bar

BAR_HEADER = "stime,code,open,high,low,close,volume"


# ============ 录制 (实盘侧: 每根 bar 追加一行) ============

def append_bar(path: str, bar: Bar):
    """实盘进程调用: 把收到的 bar 追加到日志 (首行自动写表头)"""
    import os
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(BAR_HEADER + "\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{bar.stime},{bar.code},{bar.open},{bar.high},{bar.low},"
                f"{bar.close},{bar.volume}\n")


def write_bars_log(path: str, bars) -> int:
    """批量写日志 (工具/测试用); 返回根数"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(BAR_HEADER + "\n")
        n = 0
        for b in bars:
            f.write(f"{b.stime},{b.code},{b.open},{b.high},{b.low},"
                    f"{b.close},{b.volume}\n")
            n += 1
    return n


def read_bars_log(path: str) -> list[Bar]:
    """读日志 -> list[Bar] (校验 stime 为 14 位且严格升序)"""
    bars = []
    last = ""
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()
        if header != BAR_HEADER:
            raise ValueError(f"日志表头不符: {header!r} (期望 {BAR_HEADER!r})")
        for line in f:
            line = line.strip()
            if not line:
                continue
            stime, code, o, h, l, c, v = line.split(",")
            if len(stime) != 14 or not stime.isdigit():
                raise ValueError(f"stime 非法: {stime!r}")
            if stime <= last:
                raise ValueError(f"stime 非升序: {stime} <= {last}")
            last = stime
            bars.append(Bar(stime=stime, code=code, open=float(o), high=float(h),
                            low=float(l), close=float(c), volume=int(float(v))))
    return bars


# ============ 回放 (两条引擎路径, 同一数据各跑一遍) ============

def replay_kernel(bars, period: str, warmup_until: int, tf1: int,
                  low1: float, low2: float, high1: float, high2: float,
                  init_cash: float = 200000.0, init_position: float = 200000.0,
                  trade_qty: float = 10000.0, scale: float = 1.0) -> dict:
    """内核回放: 返回逐 bar 信号轨迹 (全 bar 对齐) + 成交流 + 绩效"""
    arr = bars_to_arrays(bars)
    n = len(bars)
    st = make_state(period=period, warmup_until=warmup_until, tf1=tf1,
                    low1=low1, low2=low2, high1=high1, high2=high2,
                    init_cash=init_cash, init_position=init_position,
                    trade_qty=trade_qty, scale=scale,
                    record_trades=True, trade_cap=n)
    sig = np.zeros(n, np.int8)
    up = np.full(n, np.nan)
    dw = np.full(n, np.nan)
    run_backtest(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                 arr["close"], arr["volume"], sig, up, dw)
    return {"sig": sig, "up": up, "dw": dw,
            "trades": trades_to_list(st), "summary": summarize(st)}


def replay_engine(bars, period: str, warmup_until: int, tf1: int,
                  low1: float, low2: float, high1: float, high2: float,
                  init_cash: float = 200000.0, init_position: float = 200000.0,
                  trade_qty: float = 10000.0, scale: float = 1.0) -> dict:
    """参考引擎 (Engine 全链路) 回放: 输出与 replay_kernel 同构 (全 bar 对齐)"""
    from evtrade.account import Account
    from evtrade.aggregator import BarAggregator
    from evtrade.engine import Engine
    from evtrade.execution import SimulatedExecutor
    from evtrade.strategy import ChannelDeviationStrategy

    class _Feed:
        def __init__(self, bs):
            self.bs = bs

        def stream(self):
            yield from self.bs

    class _RecStrategy:
        def __init__(self, inner):
            self.inner = inner
            self.sig = []
            self.up = []
            self.dw = []

        def check(self, cur, up, dw):
            s, info = self.inner.check(cur, up, dw)
            self.sig.append(0 if s is None else (1 if s == "BUY" else -1))
            self.up.append(np.nan if up is None else up)
            self.dw.append(np.nan if dw is None else dw)
            return s, info

    class _RecExec(SimulatedExecutor):
        def __init__(self, account, qty, scale):
            super().__init__(account, qty, verbose=False, scale=scale)
            self.records = []

        def trade(self, signal, price, ts):
            ok = super().trade(signal, price, ts)
            if ok:
                t = self.account.trades[-1]
                self.records.append({"ts": int(t["ts"]), "side": t["side"],
                                     "qty": float(t["qty"]),
                                     "price": float(t["price"])})
            return ok

    n = len(bars)
    account = Account(cash=init_cash, position=init_position)
    executor = _RecExec(account, qty=trade_qty, scale=scale)
    strategy = _RecStrategy(ChannelDeviationStrategy(low1=low1, low2=low2,
                                                     high1=high1, high2=high2))
    aggregator = BarAggregator(resolve_period_seconds(period), on_bars=None,
                               warmup_until=str(warmup_until) if warmup_until else None)
    engine = Engine(_Feed(bars), aggregator, strategy, executor, tf1=tf1,
                    verbose=False)
    engine.run()

    # 参考引擎只在 mark=1 的 bar 调 check (结尾 flush 多一次, 不产信号);
    # 对齐到全 bar: 预热段填 0/NaN。
    stime = [b.stime for b in bars]
    sig = np.zeros(n, np.int8)
    up = np.full(n, np.nan)
    dw = np.full(n, np.nan)
    offset = 0
    if warmup_until:
        offset = int(np.searchsorted(arr_stime := np.array([int(s) for s in stime]),
                                     warmup_until))
    m = len(strategy.sig)
    if m:
        # flush 收尾会多出一条 check 记录 (不产信号), 裁剪到回放段长度
        mm = min(m, n - offset)
        sig[offset:offset + mm] = np.array(strategy.sig[:mm], dtype=np.int8)
        up[offset:offset + mm] = np.array(strategy.up[:mm], dtype=np.float64)
        dw[offset:offset + mm] = np.array(strategy.dw[:mm], dtype=np.float64)
    return {"sig": sig, "up": up, "dw": dw,
            "trades": executor.records, "summary": None}


# ============ 对账 ============

def diff_signals(sig_a: np.ndarray, sig_b: np.ndarray) -> dict:
    """逐 bar 对比两条信号轨迹; 返回首个分歧位置与统计"""
    if len(sig_a) != len(sig_b):
        return {"n_diff": -1, "first_idx": -1,
                "reason": f"长度不等 {len(sig_a)} vs {len(sig_b)}"}
    d = np.flatnonzero(sig_a != sig_b)
    return {"n_diff": int(len(d)),
            "first_idx": int(d[0]) if len(d) else -1,
            "n_a": int((sig_a != 0).sum()),
            "n_b": int((sig_b != 0).sum())}


def reconcile(bars, period: str, warmup_until: int, tf1: int,
              low1: float, low2: float, high1: float, high2: float,
              init_cash: float = 200000.0, init_position: float = 200000.0,
              trade_qty: float = 10000.0, scale: float = 1.0,
              verbose: bool = True) -> dict:
    """内核 vs 参考引擎 全面对账: 信号 / 通道值 / 成交 / 终态"""
    k = replay_kernel(bars, period, warmup_until, tf1, low1, low2, high1, high2,
                      init_cash, init_position, trade_qty, scale)
    r = replay_engine(bars, period, warmup_until, tf1, low1, low2, high1, high2,
                      init_cash, init_position, trade_qty, scale)
    d_sig = diff_signals(k["sig"], r["sig"])
    trades_ok = (len(k["trades"]) == len(r["trades"]) and all(
        kt["ts"] == rt["ts"] and kt["side"] == rt["side"]
        and kt["qty"] == rt["qty"] and kt["price"] == rt["price"]
        for kt, rt in zip(k["trades"], r["trades"])))
    report = {"n_bars": len(bars), "sig": d_sig, "trades_ok": bool(trades_ok),
              "n_trades": len(k["trades"]),
              "summary": k["summary"], "pass": d_sig["n_diff"] == 0 and trades_ok}
    if verbose:
        status = "PASS ✓" if report["pass"] else "FAIL ✗"
        print(f"对账 [{status}] bars={report['n_bars']} 信号={d_sig['n_a']} "
              f"分歧={d_sig['n_diff']} (首处 idx={d_sig['first_idx']}) "
              f"成交={report['n_trades']} 笔逐笔一致={trades_ok}")
        if not report["pass"] and d_sig.get("reason"):
            print(f"  原因: {d_sig['reason']}")
    return report
