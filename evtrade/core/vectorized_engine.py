from __future__ import annotations
"""向量化引擎 (CuPy 统一 CPU/GPU 路径)

================================================================
✅  可改层 (core 主调度)  ✅
================================================================
第三条执行路径, 与 Engine (ref 逐 bar) / kernel (DSL numba) 并存:
  - 桶聚合: 向量化 (xp.reduceat / searchsorted), GPU 加速显著
  - 信号:   策略 compute_signals(xp, ...) 纯数组算子
  - 成交:   顺序 Python 循环 (cash/position 累积依赖; 先保证正确)
  - 汇总:   与 kernel.summarize 同口径

device="cpu" -> xp=numpy; device="gpu" -> xp=cupy。
策略代码一份, framework 0 渲染逻辑。
"""

import numpy as np

from ..backends import get_xp
from .gpu import precompute_ts_mark
from .kernel import encoded_to_epoch, resolve_period_seconds


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
    stime_np = stime.get() if hasattr(stime, "get") else stime
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
    mark_b = mark_1m[last_idx]

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

    sig_np / close_np / ts_np 已拉回 host (numpy)。返回 trades list + 终态。
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

    all_in = buy_pct >= 1.0 and sell_pct >= 1.0

    for i in range(len(sig_np)):
        s = int(sig_np[i])
        if s == 0:
            continue
        price = float(close_np[i])
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
            "turnover": turnover, "trades": trades}


# ============ 汇总 (与 kernel.summarize 同口径) ============

def _summarize(exec_state: dict, init_cash: float, init_position: float,
               first_ts: int, last_ts: int) -> dict:
    """终态 -> 绩效字典 (口径与 kernel.summarize 一致; 子集)"""
    cash = exec_state["cash"]
    position = exec_state["position"]
    last_price = exec_state["last_price"]
    baseline = init_cash + init_position * last_price
    equity = cash + position * last_price
    diff = equity - baseline
    pct = (diff / baseline * 100) if baseline else 0.0

    years = 0.0
    if first_ts and last_ts > first_ts:
        years = ((encoded_to_epoch(last_ts) - encoded_to_epoch(first_ts))
                 / (365.25 * 86400.0))
    ann_excess_pct = (pct / years) if years > 0 else 0.0

    return {
        "final_price": last_price,
        "n_trades": exec_state["n_trades"],
        "n_buy": exec_state["n_buy"],
        "n_sell": exec_state["n_sell"],
        "final_cash": cash,
        "final_position": position,
        "final_equity": equity,
        "baseline": baseline,
        "excess": diff,
        "excess_pct": pct,
        "years": years,
        "ann_excess_pct": ann_excess_pct,
        "turnover": exec_state["turnover"],
        "trades": exec_state["trades"],
    }


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
    sig_np = sig.get() if hasattr(sig, "get") else np.asarray(sig)
    close_np = (buckets["c"].get() if hasattr(buckets["c"], "get")
                else np.asarray(buckets["c"]))
    ts_np = (buckets["ts"].get() if hasattr(buckets["ts"], "get")
             else np.asarray(buckets["ts"]))
    # 只在策略期 (mark=1) 成交; mark 也拉回
    mark_np = (buckets["mark"].get() if hasattr(buckets["mark"], "get")
               else np.asarray(buckets["mark"]))

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

    return {"sig": sig_np, "trades": exec_state["trades"],
            "summary": summary, "buckets": {
                k: (v.get() if hasattr(v, "get") else np.asarray(v))
                for k, v in buckets.items()}}
