from __future__ import annotations
"""向量化引擎 (numpy 向量化桶聚合 + strategy-step-only 信号循环)

引擎职责 (2026-09-13 重构):
  - 桶聚合 (numpy 向量化, 任意 m/h/d 周期)
  - 信号循环: strategy.step(state, bar, params) -> (state, sig)
  - 透传策略 final_state

framework 不持有资金/持仓/撮合/PnL/收益概念; 这些由策略 step 自管理。
"""
import numpy as np

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


# ============ 信号循环 (strategy-step-only) ============

def _compute_signals(strategy, params: dict, buckets: dict,
                     verbose: bool = False) -> tuple[list, list, np.ndarray]:
    """vectorized 路径: 循环调 strategy.step, state 由 engine 持有 (Python 对象)

    桶级 OHLCV 由 _aggregate_buckets 算好 (numpy 数组)。
    verbose=True 时, sig!=0 调 strategy.format_signal_line 并 print (与 Engine 路径同语义)。

    返回: (final_state, bucket_signals, sig_array)。
      final_state   策略 step 末尾 state (透传, framework 不读字段)
      bucket_signals 每桶调用 step 时返回的 sig (list[int])
      sig_array     与 buckets["mark"] 同形 numpy int8 (mark=0 段已清零)
    """
    state = strategy.init_state(params)
    n = len(buckets["ts"])
    sig = np.zeros(n, dtype=np.int8)
    bucket_signals: list[int] = []

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
        bucket_signals.append(int(s))
        if verbose and s != 0:
            info = getattr(strategy, "_last_info", None)
            print(strategy.format_signal_line(int(ts_np[i]), int(s), info),
                  flush=True)

    sig = sig * mark_np.astype(np.int8)
    return state, bucket_signals, sig


# ============ 主入口 ============

def run_vectorized(bars_1m: dict, period: str, warmup_until: int,
                   strategy, params: dict,
                   verbose: bool = False) -> dict:
    """向量化回测 (strategy-step-only; framework 仅驱动 step)

    bars_1m: numpy dict (bars_to_arrays 输出)
    strategy: VectorizedStrategy 实例 (step 方法)
    verbose: True 时, sig!=0 调 strategy.format_signal_line 并 print

    返回 (2026-09-13 重构): {"sig", "buckets", "final_state"}
      sig         桶级 sig (mark=0 段已清零)
      buckets     桶聚合数组 dict
      final_state 策略 step 末尾 state 透传 (framework 不读字段)
    """
    # 1) 桶聚合
    buckets = _aggregate_buckets(bars_1m, period, warmup_until)

    # 2) 信号: 循环调 strategy.step, state 由引擎持有
    final_state, _, sig_np = _compute_signals(strategy, params, buckets, verbose=verbose)

    return {"sig": sig_np, "buckets": buckets, "final_state": final_state}