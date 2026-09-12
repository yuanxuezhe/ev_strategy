"""GPU-batched sweep 编排 (opt-in; 2026-09-13 重构)

策略类可实现 `batched_step` 类方法 hook, sweep 在满足 device/cuda/n_combos
阈值时走 batched 路径: 一次 batched_step 调用产出 [N, T] 信号, 逐 combo
调 run_vectorized 串行路径补 final_state (framework 仅驱动 step + 透传)。

约束 (跟 spec 一致):
  - run_vectorized 签名不动 (spec R MUST NOT 带 device); batched 路径用新入口
  - 浮点必须 float64, 跟 per-combo step 循环 bitwise 一致
  - 异常立即透传, 不延后到 sync point
  - CUDA OOM -> 回退 ThreadPool + warning
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .timeutils import resolve_period_seconds


def _init_batched_state(strategy_cls, n_combos: int, device):
    """从策略类构造 batched_step 用的 state dataclass (全 N=0/False 初始)"""
    import torch
    state_fields = getattr(strategy_cls, "MABatchedState", None)
    if state_fields is None:
        raise NotImplementedError(
            f"{strategy_cls.__name__}: batched_step 需配套 MABatchedState dataclass")
    import dataclasses
    field_dicts = {}
    for f in dataclasses.fields(state_fields):
        if f.type is bool or "bool" in str(f.type).lower():
            field_dicts[f.name] = torch.zeros(n_combos, dtype=torch.bool, device=device)
        elif "int" in str(f.type).lower():
            field_dicts[f.name] = torch.zeros(n_combos, dtype=torch.int64, device=device)
        else:
            field_dicts[f.name] = torch.zeros(n_combos, dtype=torch.float64, device=device)
    return state_fields(**field_dicts)


def _build_param_tensors(strategy_cls, params_list: list[dict], device):
    """从 per-combo params dict 列表构建 batched param dict[str, Tensor[N]]"""
    import torch
    spec = getattr(strategy_cls, "params_spec", {}) or {}
    out = {}
    for name, schema in spec.items():
        t = schema.get("type")
        if t is int:
            dt = torch.int64
            vals = [int(p.get(name, schema.get("default"))) for p in params_list]
        elif t is float:
            dt = torch.float64
            vals = [float(p.get(name, schema.get("default"))) for p in params_list]
        else:
            raise NotImplementedError(
                f"{strategy_cls.__name__}.{name}: batched_step 仅支持 int/float 标量参数")
        out[name] = torch.tensor(vals, dtype=dt, device=device)
    return out


def _build_bars_tensor(buckets: dict, device) -> dict:
    """桶级 numpy 数组 -> batched bars dict[str, Tensor[T]]"""
    import torch
    ts_t = torch.from_numpy(np.ascontiguousarray(buckets["ts"])).to(device)
    o_t = torch.from_numpy(np.ascontiguousarray(buckets["o"].astype(np.float64))).to(device)
    h_t = torch.from_numpy(np.ascontiguousarray(buckets["h"].astype(np.float64))).to(device)
    l_t = torch.from_numpy(np.ascontiguousarray(buckets["l"].astype(np.float64))).to(device)
    c_t = torch.from_numpy(np.ascontiguousarray(buckets["c"].astype(np.float64))).to(device)
    v_t = torch.from_numpy(np.ascontiguousarray(buckets["v"].astype(np.float64))).to(device)
    mark_t = torch.from_numpy(np.ascontiguousarray(buckets["mark"].astype(np.int8))).to(device)
    return {"ts": ts_t, "o": o_t, "h": h_t, "l": l_t, "c": c_t, "v": v_t, "mark": mark_t}


def run_batched(bars: dict, base: dict, params_list: list[dict],
                *, strategy_cls, device: str, splits: list = None,
                split_ymd: str = None,
                _progress_print=None) -> list[list[dict]]:
    """一次 batched_step 出 N_combos 份信号; final_state 由策略 step 串行补

    返回 results[win_idx][combo_idx] = run_vectorized 返回 dict
    ({sig, buckets, final_state}); 与 ThreadPool 路径同构。
    """
    import torch
    from ..backends import get_xp
    from .vectorized_engine import _aggregate_buckets
    from .sweep import run_one_vectorized

    device_t = get_xp(device)
    n_combos = len(params_list)

    # 窗口切分 (跟 sweep() 同算法)
    start = base["start"]
    if split_ymd and not splits:
        splits = [split_ymd]
    if splits:
        sp = sorted(str(s) for s in splits)
        if len(sp) == 1:
            wins = [("train", None, start), ("test", None, sp[0])]
        else:
            wins = [("train", sp[0], start)]
            for i, s in enumerate(sp):
                end_b = sp[i + 1] if i + 1 < len(sp) else None
                wins.append((f"test{i + 1}", end_b, s))
    else:
        wins = [("full", None, start)]

    def window_bars(end_ymd):
        if end_ymd is None:
            return bars
        cutoff = int(end_ymd) * 1_000_000
        idx = int(np.searchsorted(bars["stime"], cutoff, side="left"))
        return {k: v[:idx] for k, v in bars.items()}

    win_data = [(nm, window_bars(e), int(w) * 1_000_000) for nm, e, w in wins]

    # init batched state + param tensors (跟 device / N_combos 绑)
    batched_state = _init_batched_state(strategy_cls, n_combos, device_t)
    params_t = _build_param_tensors(strategy_cls, params_list, device_t)

    results = [[None] * n_combos for _ in win_data]

    # 逐窗: 聚合 -> tensor -> batched_step 一次 -> 逐 combo run_vectorized 串行补
    for wi, (nm, wb, warm) in enumerate(win_data):
        t0 = time.perf_counter()
        buckets = _aggregate_buckets(wb, base["period"], warm)
        n_bars = len(buckets["ts"])
        if n_bars == 0:
            # 窗内无 bar -> 全部填空 final_state
            for ci in range(n_combos):
                results[wi][ci] = {"sig": np.zeros(0, dtype=np.int8),
                                   "buckets": buckets,
                                   "final_state": None}
            continue
        bars_t = _build_bars_tensor(buckets, device_t)

        # 调 batched_step 一次 (异常立即透传)
        _, sig = strategy_cls.batched_step(
            batched_state, bars_t, params_t,
            n_combos=n_combos, n_bars=n_bars)
        # sig: [N, T] int8 (per spec hook 约定)
        sig_np = sig.detach().cpu().numpy().astype(np.int8)  # [N, T]
        mark_np = buckets["mark"].astype(np.int8)
        sig_np = sig_np * mark_np[np.newaxis, :]

        # 逐 combo 调 run_vectorized 串行补 final_state (cash/position/PnL 等)
        for ci, p in enumerate(params_list):
            spec_keys_set = set(getattr(strategy_cls, "params_spec", {}).keys())
            sp_params = {k: p[k] for k in spec_keys_set if k in p}
            results[wi][ci] = run_one_vectorized(
                wb, p["period"], warm,
                strategy_name=strategy_cls.strategy_key,
                strategy_params=sp_params,
            )

        if _progress_print is not None:
            _progress_print(
                f"  -- batched 窗 {nm} 耗时 {time.perf_counter() - t0:.2f}s "
                f"[{device_t}]"
            )

    return results