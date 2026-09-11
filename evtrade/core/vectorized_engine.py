from __future__ import annotations
"""向量化引擎 (numpy 向量化桶聚合 + strategy-step-only 信号循环)

引擎职责:
  - 桶聚合 (numpy 向量化, 任意 m/h/d 周期)
  - 信号循环: strategy.step(state, bar, params) -> (state, sig)
  - 成交执行 (顺序; 与 SimulatedExecutor 同语义)
  - 汇总 metrics (30 字段)

state 由引擎持有, 跨调用持续。strategy.step 是策略唯一入口。
"""

import numpy as np

from ..execution.base import trade_decision
from .config import INIT_CASH, INIT_POSITION, TRADE_QTY
from .metrics import summarize
from .timeutils import resolve_period_seconds
from .tsbucket import precompute_ts_mark


# ============ 桶聚合 (numpy 向量化) ============

def _aggregate_buckets(bars_1m: dict, period: str, warmup_until: int) -> dict:
    """1m bar 数组 -> 周期桶聚合数组

    返回: {"ts", "o", "h", "l", "c", "v", "mark", "n_bars"}
    每行 = 一个闭合桶的最终 OHLCV + 含 1m 根数 + mark。

    空输入 (n=0) 返回空数组 (避免 reduceat 在 0-size 上 IndexError)。
    """
    stime = bars_1m["stime"]
    n = len(stime)
    if n == 0:
        empty = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=np.float64)
        empty_i8 = np.array([], dtype=np.int8)
        return {"ts": empty, "o": empty_f, "h": empty_f, "l": empty_f,
                "c": empty_f, "v": empty_f, "mark": empty_i8, "n_bars": empty}
    ts_1m_np, mark_1m_np = precompute_ts_mark({"stime": stime}, period, warmup_until)
    ts_1m = np.asarray(ts_1m_np)
    mark_1m = np.asarray(mark_1m_np)

    o_1m = np.asarray(bars_1m["open"])
    h_1m = np.asarray(bars_1m["high"])
    l_1m = np.asarray(bars_1m["low"])
    c_1m = np.asarray(bars_1m["close"])
    v_1m = np.asarray(bars_1m["volume"])

    n = len(stime)
    new_bucket = np.ones(n, dtype=bool)
    new_bucket[1:] = ts_1m[1:] != ts_1m[:-1]
    first_idx = np.flatnonzero(new_bucket)
    last_idx = np.flatnonzero(np.concatenate([new_bucket[1:], np.array([True])]))

    ts_b = ts_1m[first_idx]
    o_b = o_1m[first_idx]
    h_b = np.maximum.reduceat(h_1m, first_idx)
    l_b = np.minimum.reduceat(l_1m, first_idx)
    c_b = c_1m[last_idx]
    v_b = np.add.reduceat(v_1m, first_idx)
    n_bars = np.diff(np.concatenate([first_idx, np.array([n], dtype=first_idx.dtype)]))

    # mark 取桶首根 1m bar 的标记 (与 Engine.on_bars 仅在桶首触发语义对齐)
    mark_b = mark_1m[first_idx]

    return {"ts": ts_b, "o": o_b, "h": h_b, "l": l_b, "c": c_b, "v": v_b,
            "mark": mark_b, "n_bars": n_bars}


# ============ 成交执行 (顺序; 与 SimulatedExecutor 同语义) ============

def _execute_trades(sig_np, close_np, ts_np,
                    init_cash, init_position, trade_qty,
                    scale, buy_pct, sell_pct):
    """顺序遍历信号数组, 模拟成交 (与 SimulatedExecutor.trade + Account.apply 同式)

    sig_np / close_np / ts_np 必须为 numpy 1D array。
    返回: 终态 dict + equity_curve + baseline_curve (供 metrics.summarize 用)。
    """
    cash = init_cash
    position = init_position
    last_side = 0
    cur_qty = trade_qty
    trades = []
    n_trades = 0
    n_buy = 0
    n_sell = 0
    turnover = 0.0
    last_price = 0.0

    n = len(sig_np)
    equity_curve = np.empty(n, dtype=np.float64)
    baseline_curve = np.empty(n, dtype=np.float64)

    for i in range(n):
        price = float(close_np[i])
        baseline_curve[i] = init_cash + init_position * price
        equity_curve[i] = cash + position * price

        s = int(sig_np[i])
        if s == 0:
            continue
        ts = int(ts_np[i])
        last_price = price

        if s == last_side:
            cur_qty = cur_qty * scale
        else:
            cur_qty = trade_qty
            last_side = s

        cash, position, q, filled = trade_decision(
            s, price, cash, position, cur_qty, buy_pct, sell_pct)
        if not filled:
            continue
        n_trades += 1
        turnover += q * price
        if s == 1:
            n_buy += 1
            trades.append({"ts": ts, "side": "BUY", "qty": float(q),
                           "price": float(price), "cash_after": float(cash)})
        else:
            n_sell += 1
            trades.append({"ts": ts, "side": "SELL", "qty": float(q),
                           "price": float(price), "cash_after": float(cash)})

    return {"cash": cash, "position": position, "last_price": last_price,
            "n_trades": n_trades, "n_buy": n_buy, "n_sell": n_sell,
            "turnover": turnover, "trades": trades,
            "equity_curve": equity_curve,
            "baseline_curve": baseline_curve}


# ============ 信号循环 (strategy-step-only) ============

def _compute_signals(strategy, params: dict, buckets: dict,
                     verbose: bool = False) -> np.ndarray:
    """vectorized 路径: 循环调 strategy.step, state 由 engine 持有 (Python 对象)

    桶级 OHLCV 由 _aggregate_buckets 算好 (numpy 数组)。
    verbose=True 时, sig!=0 调 strategy.format_signal_line 并 print (与 Engine 路径同语义)。

    返回: sig 序列 (numpy int8), 已 mark=0 清零。
    """
    state = strategy.init_state(params)
    n = len(buckets["ts"])
    sig = np.zeros(n, dtype=np.int8)

    ts_np = buckets["ts"]
    o_np = buckets["o"]
    h_np = buckets["h"]
    l_np = buckets["l"]
    c_np = buckets["c"]
    v_np = buckets["v"]
    mark_np = buckets["mark"]

    for i in range(n):
        bar = {
            "ts": int(ts_np[i]),
            "o": float(o_np[i]),
            "h": float(h_np[i]),
            "l": float(l_np[i]),
            "c": float(c_np[i]),
            "v": float(v_np[i]),
            "mark": int(mark_np[i]),
        }
        state, s = strategy.step(state, bar, params)
        sig[i] = s
        if verbose and s != 0:
            info = getattr(strategy, "_last_info", None)
            print(strategy.format_signal_line(int(ts_np[i]), int(s), info),
                  flush=True)

    sig = sig * mark_np.astype(np.int8)
    return sig




# ============ 主入口 ============

def run_vectorized(bars_1m: dict, period: str, warmup_until: int,
                   strategy, params: dict,
                   init_cash: float = INIT_CASH, init_position: float = INIT_POSITION,
                   trade_qty: float = TRADE_QTY, scale: float = 1.0,
                   buy_pct: float = 0.0, sell_pct: float = 0.0,
                   verbose: bool = False) -> dict:
    """向量化回测 (strategy-step-only)

    bars_1m: numpy dict (bars_to_arrays 输出)
    strategy: VectorizedStrategy 实例 (step 方法)
    verbose: True 时, sig!=0 调 strategy.format_signal_line 并 print

    返回: {"sig", "trades", "summary", "buckets"}
    """
    # 1) 桶聚合
    buckets = _aggregate_buckets(bars_1m, period, warmup_until)

    # 2) 信号: 循环调 strategy.step, state 由引擎持有
    sig_np = _compute_signals(strategy, params, buckets, verbose=verbose)

    # 3) 成交 (顺序执行)
    close_np = buckets["c"]
    ts_np = buckets["ts"]
    mark_np = buckets["mark"]

    strat_mask = mark_np == 1
    first_ts = int(ts_np[strat_mask][0]) if strat_mask.any() else 0
    last_ts = int(ts_np[strat_mask][-1]) if strat_mask.any() else 0

    exec_state = _execute_trades(
        sig_np, close_np, ts_np, init_cash, init_position, trade_qty,
        scale, buy_pct, sell_pct)

    # 汇总 (metrics.summarize, 30 字段)
    summary = summarize(
        final_state={
            "cash": exec_state["cash"],
            "position": exec_state["position"],
            "last_price": exec_state.get("last_price", 0.0),
            "n_trades": exec_state["n_trades"],
            "n_buy": exec_state["n_buy"],
            "n_sell": exec_state["n_sell"],
            "turnover": exec_state["turnover"],
        },
        init_cash=init_cash,
        init_position=init_position,
        equity_curve=exec_state.get("equity_curve"),
        baseline_curve=exec_state.get("baseline_curve"),
        first_ts=first_ts,
        last_ts=last_ts,
        trades=exec_state["trades"],
        bucket_seconds=resolve_period_seconds(period),
    )
    summary["trades"] = exec_state["trades"]

    sig_live = sig_np[mark_np == 1]

    return {"sig": sig_live, "trades": exec_state["trades"],
            "summary": summary, "buckets": buckets}
