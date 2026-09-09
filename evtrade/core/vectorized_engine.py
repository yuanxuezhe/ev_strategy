from __future__ import annotations
"""向量化引擎 (CuPy 统一 CPU/GPU 路径)

唯一执行路径 (DSL/ref 引擎已下线):
  - 桶聚合: 向量化 (xp 算子), GPU 加速显著
  - 信号:   策略 compute_signals(xp, ...) 纯数组算子
  - 成交:   顺序 Python 循环 (cash/position 累积依赖; 先保证正确)
  - 汇总:   metrics.summarize (16 字段, 含 equity_curve)

device="cpu" -> xp=numpy; device="gpu" -> xp=cupy。
策略代码一份, framework 0 渲染逻辑。
"""

import numpy as np

from ..backends import get_xp
from .gpu import precompute_ts_mark
from .metrics import summarize as _summarize_full


# ============ helpers ============

def _to_host(arr):
    """xp 数组 -> numpy (cupy 拉回 host; numpy 原样)"""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


# ============ 桶聚合 (向量化; xp=np|cp) ============

def _aggregate_buckets_xp(xp, bars_1m: dict, period: str, warmup_until: int) -> dict:
    """1m bar 数组 -> 周期桶聚合数组 (向量化)

    返回: {"ts", "o", "h", "l", "c", "v", "mark", "n_bars"}
    每行 = 一个闭合桶的最终 OHLCV + 含 1m 根数 + mark。

    复用 gpu.precompute_ts_mark 算每根 1m bar 的桶 ts (已是纯 numpy 算术,
    CuPy 兼容); 再用 searchsorted 找桶边界, reduceat 聚合。
    """
    stime = bars_1m["stime"]
    # precompute_ts_mark 接受 numpy stime (14 位整数); 返回 (ts int64[n], mark int8[n])
    # GPU 路径下先把 stime 拉到 device
    stime_np = _to_host(stime)
    ts_1m_np, mark_1m_np = precompute_ts_mark(
        {"stime": stime_np}, period, warmup_until)
    # 搬到 xp (cupy 时上 device)
    ts_1m = xp.asarray(ts_1m_np)
    mark_1m = xp.asarray(mark_1m_np)

    o_1m = bars_1m["open"]; h_1m = bars_1m["high"]; l_1m = bars_1m["low"]
    c_1m = bars_1m["close"]; v_1m = bars_1m["volume"]
    # 确保在 device 上
    o_1m = xp.asarray(o_1m); h_1m = xp.asarray(h_1m); l_1m = xp.asarray(l_1m)
    c_1m = xp.asarray(c_1m); v_1m = xp.asarray(v_1m)

    n = len(stime)
    # 桶边界: ts 变化处
    new_bucket = xp.ones(n, dtype=xp.bool_)
    new_bucket[1:] = ts_1m[1:] != ts_1m[:-1]
    first_idx = xp.flatnonzero(new_bucket)
    last_idx = xp.flatnonzero(xp.concatenate([new_bucket[1:], xp.array([True])]))

    # 每桶 OHLCV (reduceat 向量化)
    ts_b = ts_1m[first_idx]
    o_b = o_1m[first_idx]
    h_b = _reduceat_max(xp, h_1m, first_idx)
    l_b = _reduceat_min(xp, l_1m, first_idx)
    c_b = c_1m[last_idx]
    v_b = _reduceat_sum(xp, v_1m, first_idx)
    n_bars = xp.diff(xp.concatenate([first_idx, xp.array([n], dtype=first_idx.dtype)]))
    # mark 取桶首根 1m bar 的标记 (与 Engine.on_bars 仅在桶首触发语义对齐)
    mark_b = mark_1m[first_idx]

    return {"ts": ts_b, "o": o_b, "h": h_b, "l": l_b, "c": c_b, "v": v_b,
            "mark": mark_b, "n_bars": n_bars}


def _reduceat_max(xp, a, indices):
    """分段 max (reduceat 等价; cupy 兼容)"""
    if len(indices) == 0:
        return xp.empty(0, dtype=a.dtype)
    out = xp.empty(len(indices), dtype=a.dtype)
    for i in range(len(indices)):
        s = indices[i]
        e = indices[i + 1] if i + 1 < len(indices) else len(a)
        out[i] = xp.max(a[s:e])
    return out


def _reduceat_min(xp, a, indices):
    if len(indices) == 0:
        return xp.empty(0, dtype=a.dtype)
    out = xp.empty(len(indices), dtype=a.dtype)
    for i in range(len(indices)):
        s = indices[i]
        e = indices[i + 1] if i + 1 < len(indices) else len(a)
        out[i] = xp.min(a[s:e])
    return out


def _reduceat_sum(xp, a, indices):
    if len(indices) == 0:
        return xp.empty(0, dtype=a.dtype)
    out = xp.empty(len(indices), dtype=a.dtype)
    for i in range(len(indices)):
        s = indices[i]
        e = indices[i + 1] if i + 1 < len(indices) else len(a)
        out[i] = xp.sum(a[s:e])
    return out


# ============ 成交执行 (顺序; 与 SimulatedExecutor 同语义) ============

def _execute_trades(sig_np: np.ndarray, close_np: np.ndarray, ts_np: np.ndarray,
                    init_cash: float, init_position: float, trade_qty: float,
                    scale: float, buy_pct: float, sell_pct: float):
    """顺序遍历信号数组, 模拟成交 (与 SimulatedExecutor.trade + Account.apply 同式)

    sig_np / close_np / ts_np 已拉回 host (numpy)。
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
    running_max = -np.inf
    max_dd = 0.0

    for i in range(n):
        price = float(close_np[i])
        # baseline = init_cash + init_position * price (持仓按当前 close 估值)
        baseline_curve[i] = init_cash + init_position * price
        equity_curve[i] = cash + position * price

        # 逐 bar 跟踪 max_drawdown (在 equity 收盘时点)
        if equity_curve[i] > running_max:
            running_max = equity_curve[i]
        dd = equity_curve[i] - running_max
        if dd < max_dd:
            max_dd = dd

        s = int(sig_np[i])
        if s == 0:
            continue
        ts = int(ts_np[i])
        last_price = price

        # 倍投
        if s == last_side:
            cur_qty = cur_qty * scale
        else:
            cur_qty = trade_qty
            last_side = s

        if s == 1:  # BUY
            if price > 0:
                max_by_cash = cash / price
                if buy_pct > 0:
                    target = buy_pct * max_by_cash
                    q = target if target < max_by_cash else max_by_cash
                else:
                    q = cur_qty if cur_qty < max_by_cash else max_by_cash
            else:
                q = 0.0
            if q > 0:
                cash -= q * price
                position += q
                n_trades += 1; n_buy += 1
                turnover += q * price
                trades.append({"ts": ts, "side": "BUY", "qty": float(q),
                               "price": float(price), "cash_after": float(cash)})
        else:  # SELL
            if sell_pct > 0:
                target = sell_pct * position
                q = target if target < position else position
            else:
                q = cur_qty if cur_qty < position else position
            if q > 0:
                cash += q * price
                position -= q
                n_trades += 1; n_sell += 1
                turnover += q * price
                trades.append({"ts": ts, "side": "SELL", "qty": float(q),
                               "price": float(price), "cash_after": float(cash)})

    return {"cash": cash, "position": position, "last_price": last_price,
            "n_trades": n_trades, "n_buy": n_buy, "n_sell": n_sell,
            "turnover": turnover, "trades": trades,
            "max_drawdown": -max_dd if max_dd < 0.0 else 0.0,
            "equity_curve": equity_curve,
            "baseline_curve": baseline_curve}


# ============ 汇总 (metrics.summarize 16 字段) ============

def _summarize(exec_state: dict, init_cash: float, init_position: float,
               first_ts: int, last_ts: int) -> dict:
    """终态 + equity 序列 -> 绩效字典 (16 字段, 由 metrics.summarize 算)"""
    eq = exec_state.get("equity_curve")
    bl = exec_state.get("baseline_curve")
    last_price = exec_state.get("last_price", 0.0)
    cash = exec_state["cash"]
    position = exec_state["position"]

    summary = _summarize_full(
        final_state={
            "cash": cash,
            "position": position,
            "last_price": last_price,
            "n_trades": exec_state["n_trades"],
            "n_buy": exec_state["n_buy"],
            "n_sell": exec_state["n_sell"],
            "turnover": exec_state["turnover"],
            "max_drawdown": exec_state.get("max_drawdown", 0.0),
        },
        init_cash=init_cash,
        init_position=init_position,
        equity_curve=eq,
        baseline_curve=bl,
        first_ts=first_ts,
        last_ts=last_ts,
    )
    summary["trades"] = exec_state["trades"]
    return summary


# ============ 主入口 ============

def run_vectorized(bars_1m: dict, period: str, warmup_until: int,
                   strategy, params: dict,
                   init_cash: float = 200000.0, init_position: float = 200000.0,
                   trade_qty: float = 10000.0, scale: float = 1.0,
                   buy_pct: float = 0.0, sell_pct: float = 0.0,
                   device: str = "cpu") -> dict:
    """向量化回测 (CPU/GPU 统一入口)

    bars_1m: kernel.bars_to_arrays 输出 (numpy dict)
    strategy: VectorizedStrategy 实例 (compute_signals 方法)
    device: "cpu" -> numpy; "gpu" -> cupy

    返回: {"sig", "trades", "summary", "buckets"}
    """
    xp = get_xp(device)

    # 1) 桶聚合 (向量化, GPU 加速)
    buckets = _aggregate_buckets_xp(xp, bars_1m, period, warmup_until)

    # 2) 信号 (策略算, 纯数组)
    sig = strategy.compute_signals(xp, buckets, params)
    sig = xp.asarray(sig, dtype=xp.int8)

    # 3) 成交 (拉回 host 顺序执行; 正确性优先)
    sig_np = _to_host(sig)
    close_np = _to_host(buckets["c"])
    ts_np = _to_host(buckets["ts"])
    # 只在策略期 (mark=1) 成交; mark 也拉回
    mark_np = _to_host(buckets["mark"])

    # 预热段信号清零 (mark=0 的桶不应成交)
    sig_np = sig_np * mark_np

    # 首末策略期 ts (年化用)
    strat_mask = mark_np == 1
    first_ts = int(ts_np[strat_mask][0]) if strat_mask.any() else 0
    last_ts = int(ts_np[strat_mask][-1]) if strat_mask.any() else 0

    exec_state = _execute_trades(
        sig_np, close_np, ts_np, init_cash, init_position, trade_qty,
        scale, buy_pct, sell_pct)

    summary = _summarize(exec_state, init_cash, init_position, first_ts, last_ts)

    # sig 只保留 mark=1 的桶 (与 Engine.bucket_signals 同形: 预热段不进 sig)
    sig_live = sig_np[mark_np == 1]

    return {"sig": sig_live, "trades": exec_state["trades"],
            "summary": summary, "buckets": {k: _to_host(v) for k, v in buckets.items()}}
