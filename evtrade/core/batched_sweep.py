"""GPU-batched sweep 编排 (opt-in)

策略类可实现 `batched_step` 类方法 hook, sweep 在满足 device/cuda/n_combos
阈值时走 batched 路径: 一次 batched_step 调用产出 [N, T] 信号, 逐 combo
调现有 `_execute_trades` 拿 trade/equity (per-combo 串行, 跟 ThreadPool 路径同)。

约束 (跟 spec 一致):
  - run_vectorized 签名不动 (spec R10 MUST NOT 带 device); batched 路径用新入口
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
    # ma_crossover 当前实现: state 字段为 fast_count/slow_count/prev_diff/has_prev
    # 通用做法: 探测 fields, 给数值字段 0 张量、bool 字段 False 张量
    # 第一刀只支持 ma_crossover (MABatchedState); 后续策略可各自实现 init 逻辑
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
        # 第一刀仅支持 int / float 标量参数
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
                fee_bp: float = 5.0, lam: float = 1.0,
                min_trades: int = 30, max_mdd: float = 1.0,
                split_ymd: str = None,
                _progress_print=None) -> list[list[dict]]:
    """一次 batched_step 出 N_combos 份信号, 逐 combo 调 _execute_trades

    返回 metrics[win_idx][combo_idx] = summary dict, 与 ThreadPool 路径同构。

    参数:
      bars         1m bar dict (numpy 数组)
      base         sweep base (含 start/period/trade_qty/init_cash/init_position/buy_pct/sell_pct)
      params_list  per-combo 已合并的参数字典列表
      strategy_cls VectorizedStrategy 子类
      device       "cpu" / "gpu" (字符串; 用 backends.get_xp 转 torch.device)
      splits       WFO 窗口列表
      fee_bp / lam / min_trades / max_mdd / split_ymd  跟 sweep() 透传
      _progress_print  可选回调 (msg: str) -> None
    """
    import torch
    from ..backends import get_xp
    from .vectorized_engine import _aggregate_buckets, _execute_trades
    from .metrics import summarize

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

    metrics = [[None] * n_combos for _ in win_data]

    # 逐窗: 聚合 -> tensor -> batched_step 一次 -> 逐 combo _execute_trades
    for wi, (nm, wb, warm) in enumerate(win_data):
        t0 = time.perf_counter()
        buckets = _aggregate_buckets(wb, base["period"], warm)
        n_bars = len(buckets["ts"])
        if n_bars == 0:
            # 窗内无 bar -> 全部填"零成交"summary (跟 vectorized 路径语义对齐)
            empty_summary = _zero_summary(base)
            for ci in range(n_combos):
                metrics[wi][ci] = empty_summary
            continue
        bars_t = _build_bars_tensor(buckets, device_t)

        # 调 batched_step 一次 (异常立即透传)
        _, sig = strategy_cls.batched_step(
            batched_state, bars_t, params_t,
            n_combos=n_combos, n_bars=n_bars)
        # sig: [N, T] int8 (per spec hook 约定)

        # 逐 combo 调现有 _execute_trades (numpy 路径, 跟 ThreadPool 路径同)
        sig_np = sig.detach().cpu().numpy().astype(np.int8)  # [N, T]
        # 应用 mark 遮罩 (跟 _compute_signals 末行同语义: sig *= mark)
        mark_np = buckets["mark"].astype(np.int8)
        sig_np = sig_np * mark_np[np.newaxis, :]
        # 注意: run_vectorized 末尾的 `sig = sig * mark_np.astype(np.int8)` 是 in-place
        # 桶级 mask; 这里 sig 已经含 mark 遮罩 (batched_step 内部已处理), 但 spec 兼容起见
        # 再做一次 per-combo 遮罩, 跟 _compute_signals 末行 bit-equal

        close_np = buckets["c"].astype(np.float64)
        ts_np = buckets["ts"].astype(np.int64)
        for ci in range(n_combos):
            p = params_list[ci]
            try:
                exec_state = _execute_trades(
                    sig_np[ci], close_np, ts_np,
                    init_cash=base["init_cash"],
                    init_position=base["init_position"],
                    trade_qty=p["trade_qty"],
                    buy_pct=p.get("buy_pct", 0.0),
                    sell_pct=p.get("sell_pct", 0.0),
                )
            except Exception:
                # batched_step 异常已上抛; 此处 _execute_trades 抛也透传
                raise
            summary = summarize(
                final_state=exec_state,
                init_cash=base["init_cash"],
                init_position=base["init_position"],
                equity_curve=exec_state["equity_curve"],
                baseline_curve=exec_state["baseline_curve"],
                first_ts=int(ts_np[0]) if len(ts_np) > 0 else 0,
                last_ts=int(ts_np[-1]) if len(ts_np) > 0 else 0,
                trades=exec_state["trades"],
                bucket_seconds=resolve_period_seconds(base["period"]),
            )
            summary["trades"] = exec_state["trades"]
            metrics[wi][ci] = summary

        if _progress_print is not None:
            _progress_print(
                f"  -- batched 窗 {nm} 耗时 {time.perf_counter() - t0:.2f}s "
                f"[{device_t}]"
            )

    return metrics


def _zero_summary(base: dict) -> dict:
    """窗内无 bar 时填的零成交 summary (跟 vectorized 路径语义对齐)"""
    from .config import INIT_CASH, INIT_POSITION
    init_cash = base.get("init_cash", INIT_CASH)
    init_pos = base.get("init_position", INIT_POSITION)
    last_price = 0.0
    eq = init_cash + init_pos * last_price
    return {
        "final_price": last_price,
        "final_cash": init_cash,
        "final_position": float(init_pos),
        "final_equity": eq,
        "baseline": eq,
        "n_trades": 0, "n_buy": 0, "n_sell": 0,
        "turnover": 0.0,
        "excess_pct": 0.0,
        "years": 0.0,
        "cagr": 0.0, "cagr_excess": 0.0,
        "sharpe_excess": 0.0, "sortino_excess": 0.0,
        "calmar": 0.0, "ir": 0.0,
        "max_drawdown": 0.0, "max_dd_days": 0.0,
        "max_dd_recovered": -1,
        "x_mdd": 0.0,
        "win_rate": 0.0, "profit_factor": 0.0, "avg_pnl": 0.0,
        "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        "avg_hold_bars": 0.0, "max_hold_bars": 0,
        "baseline_max_dd": 0.0, "dd_excess": 0.0,
        "trades": [],
    }
