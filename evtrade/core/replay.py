from __future__ import annotations
"""录制 / 回放 / 对账: 回测与实盘一致性的验收工具

公开 API:
  - append_bar / write_bars_log / read_bars_log: 实盘日志录制与读取
  - replay_vectorized: vectorized 引擎回放 -> 信号 + 成交
  - replay_engine: Engine 全链路回放 -> 信号 + 成交
  - reconcile: 两条路径逐 bar 信号 + 逐笔成交 bitwise 对账

工作流:
  1. 录制: 实盘进程每收到一根 1m bar, 追加一行到日志
     (append_bar / write_bars_log; 格式 stime,code,open,high,low,close,volume)
  2. 回放: 收盘后把日志喂给 vectorized 引擎, 得到逐 bar 信号轨迹与成交流
     (replay_vectorized)
  3. 对账: 对比 vectorized vs Engine.on_bars (reconcile)

约定: stime 为 14 位 YYYYMMDDHHmmss 且升序; 成交 ts 记桶右端点。
"""

import numpy as np

from ..primitives import Bar
from .config import INIT_CASH, INIT_POSITION, TRADE_QTY
from .metrics import bars_to_arrays
from .timeutils import resolve_period_seconds

BAR_HEADER = "stime,code,open,high,low,close,volume"


def _format_bar_line(b) -> str:
    return f"{b.stime},{b.code},{b.open},{b.high},{b.low},{b.close},{b.volume}"


def append_bar(path: str, bar: Bar):
    """实盘进程调用: 把收到的 bar 追加到日志 (首行自动写表头)"""
    import os
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(BAR_HEADER + "\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(_format_bar_line(bar) + "\n")


def write_bars_log(path: str, bars) -> int:
    """批量写日志 (工具/测试用); 返回根数"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(BAR_HEADER + "\n")
        n = 0
        for b in bars:
            f.write(_format_bar_line(b) + "\n")
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

def replay_vectorized(bars, period: str, warmup_until: int,
                      strategy_name: str, strategy_params: dict,
                      init_cash: float = INIT_CASH,
                      init_position: float = INIT_POSITION,
                      trade_qty: float = TRADE_QTY, scale: float = 1.0,
                      buy_pct: float = 0.0, sell_pct: float = 0.0) -> dict:
    """vectorized 引擎回放: 返回逐 bar 信号轨迹 (桶级对齐) + 成交流 + 绩效

    返回 dict:
      "sig"        桶级信号轨迹 (int8[N_live_buckets])
      "ts"         桶级 ts (与 sig 对齐, 与 trades 的 ts 同源)
      "trades"     成交流 (list)
      "summary"    绩效摘要 (dict)
    """
    from ..strategies import get_strategy
    from .vectorized_engine import run_vectorized

    arr = bars_to_arrays(bars)
    strategy = get_strategy(strategy_name, params=strategy_params or {})
    out = run_vectorized(
        bars_1m=arr, period=period, warmup_until=warmup_until,
        strategy=strategy, params=strategy.params,
        init_cash=init_cash, init_position=init_position,
        trade_qty=trade_qty, scale=scale,
        buy_pct=buy_pct, sell_pct=sell_pct,
    )
    return {
        "sig": out["sig"],
        "ts": out["buckets"]["ts"][out["buckets"]["mark"] == 1],
        "trades": out["trades"],
        "summary": out["summary"],
    }


def replay_engine(bars, period: str, warmup_until: int,
                  strategy_name: str, strategy_params: dict,
                  init_cash: float = INIT_CASH, init_position: float = INIT_POSITION,
                  trade_qty: float = TRADE_QTY, scale: float = 1.0,
                  buy_pct: float = 0.0, sell_pct: float = 0.0,
                  all_in: bool = False) -> dict:
    """参考引擎 (Engine 全链路) 回放: 输出与 replay_vectorized 同构 (桶级对齐)

    策略实例从 strategies registry 取 (按 strategy_name + strategy_params)。
    走逐 bar on_bars 路径, instance-level FSM (channel_deviation) 在此路径生效。
    """
    from ..strategies import get_strategy
    from ..account import Account
    from ..aggregator import BarAggregator
    from .engine import Engine
    from ..execution import SimulatedExecutor
    from ._harness import ListBarFeed

    feed = ListBarFeed(bars)

    records: list[dict] = []
    account = Account(cash=init_cash, position=init_position)
    # Executor 仅调 trade_decision + Account.apply; 不再硬编码 print 成交明细 (2026-09-12 cleanup).
    # Engine.verbose=True 让 sig!=0 时调 strategy.format_signal_line 并 print.
    executor = SimulatedExecutor(account, qty=trade_qty,
                                 scale=scale, buy_pct=buy_pct,
                                 sell_pct=sell_pct, all_in=all_in,
                                 record_to=records)
    strategy = get_strategy(strategy_name, params=strategy_params or {})
    aggregator = BarAggregator(resolve_period_seconds(period), on_bars=None,
                               warmup_until=str(warmup_until) if warmup_until else None)
    engine = Engine(feed, aggregator, strategy, executor, verbose=True)
    engine.run()

    # bucket_signals 已是桶级信号列表, 直接返桶级数组
    return {"sig": np.array(engine.bucket_signals, dtype=np.int8),
            "trades": records,
            "summary": None}


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


def reconcile(bars, period: str, warmup_until: int,
              strategy_name: str, strategy_params: dict,
              init_cash: float = INIT_CASH, init_position: float = INIT_POSITION,
              trade_qty: float = TRADE_QTY, scale: float = 1.0,
              buy_pct: float = 0.0, sell_pct: float = 0.0, all_in: bool = False,
              verbose: bool = True,
              strict: bool | None = None,
              bucket_diff_cap: int | None = None) -> dict:
    """vectorized vs Engine.on_bars 全面对账: 信号 / 成交 / 终态 (任意 VectorizedStrategy)

    strict:
      - True:  信号分歧必须为 0 (严口径, 默认在 EVT_RECONCILE_STRICT=1 时生效)
      - False: 默认允许 bucket_diff_cap 桶差异 (默认 8, 应对 EMA high vs first-1m-high 累积漂移)
    bucket_diff_cap: 自定义差异上限 (None 时按 strict 取 0 或 8)
    """
    if strict is None:
        import os
        strict = os.environ.get("EVT_RECONCILE_STRICT") == "1"
    if bucket_diff_cap is None:
        bucket_diff_cap = 0 if strict else 8
    k = replay_vectorized(bars, period, warmup_until, strategy_name, strategy_params,
                          init_cash, init_position, trade_qty, scale,
                          buy_pct=buy_pct, sell_pct=sell_pct)
    r = replay_engine(bars, period, warmup_until, strategy_name, strategy_params,
                      init_cash, init_position, trade_qty, scale,
                      buy_pct=buy_pct, sell_pct=sell_pct, all_in=all_in)
    d_sig = diff_signals(k["sig"], r["sig"])
    trades_ok = (len(k["trades"]) == len(r["trades"]) and all(
        kt["ts"] == rt["ts"] and kt["side"] == rt["side"]
        and kt["qty"] == rt["qty"] and kt["price"] == rt["price"]
        for kt, rt in zip(k["trades"], r["trades"])))
    sig_pass = d_sig["n_diff"] <= bucket_diff_cap
    report = {"n_bars": len(bars), "sig": d_sig, "trades_ok": bool(trades_ok),
              "n_trades": len(k["trades"]),
              "summary": k["summary"], "pass": sig_pass and trades_ok,
              "cap": bucket_diff_cap, "strict": strict}
    if verbose:
        status = "PASS" if report["pass"] else "FAIL"
        print(f"对账 [{status}] bars={report['n_bars']} 信号={d_sig['n_a']} "
              f"分歧={d_sig['n_diff']} (cap={bucket_diff_cap} 首处 idx={d_sig['first_idx']}) "
              f"成交={report['n_trades']} 笔逐笔一致={trades_ok}")
        if not report["pass"] and d_sig.get("reason"):
            print(f"  原因: {d_sig['reason']}")
    return report
