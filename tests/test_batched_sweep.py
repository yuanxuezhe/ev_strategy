"""GPU-Batched sweep 路径测试

锁定 batched_step hook + torch_ema kernel + sweep 路由 + 异常 fallback 行为
(openspec/changes/gpu-batched-sweep)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import evtrade
from evtrade.core import sweep as sweep_mod
from evtrade.core.data import synthetic_bars
from evtrade.indicators import ema, torch_ema
from evtrade.strategies.channel_deviation import ChannelDeviationStrategy
from evtrade.strategies.ma_crossover import MACrossoverStrategy, MABatchedState
from evtrade.strategies.vectorized_base import VectorizedStrategy


def _synthetic_arr(days: int = 30, start_ymd: str = "20260101", seed: int = 42):
    """合成 bars -> numpy dict (跟 load_bars 输出同型)"""
    bars = synthetic_bars(days=days, start_ymd=start_ymd, seed=seed)
    return {
        "stime": np.array([int(b.stime) for b in bars], dtype=np.int64),
        "open": np.array([b.open for b in bars], dtype=np.float64),
        "high": np.array([b.high for b in bars], dtype=np.float64),
        "low": np.array([b.low for b in bars], dtype=np.float64),
        "close": np.array([b.close for b in bars], dtype=np.float64),
        "volume": np.array([b.volume for b in bars], dtype=np.float64),
    }


# ============ 1. Hook 默认 + 覆写检测 ============

def test_batched_step_default_raises():
    """VectorizedStrategy.batched_step 默认 NotImplementedError; channel 不覆写"""
    with pytest.raises(NotImplementedError):
        VectorizedStrategy.batched_step(None, None, None, n_combos=0, n_bars=0)
    # 基类自身 __dict__ 里有 batched_step (默认实现)
    assert "batched_step" in vars(VectorizedStrategy)
    # channel_deviation 不覆写
    assert "batched_step" not in vars(ChannelDeviationStrategy)
    # ma_crossover 真覆写
    assert "batched_step" in vars(MACrossoverStrategy)


def test_ma_crossover_init_state_returns_batched_state_dataclass():
    """MACrossoverStrategy.MABatchedState 是 @dataclass, 字段为 Tensor 类型注解"""
    from dataclasses import is_dataclass
    assert is_dataclass(MABatchedState)
    # 注: batched_step 的 state 由 _init_batched_state 构造, 字段类型注解是 Tensor
    # (用字符串注解避免运行期硬依赖 torch)


# ============ 2. torch_ema 跟 numpy 参考 bit-equal ============

@pytest.mark.parametrize("p", [2, 5, 21, 60])
def test_torch_ema_matches_numpy_reference(p):
    """p ∈ {2,5,21,60} 时 torch_ema vs numpy ema() float64 bit-equal"""
    rng = np.random.default_rng(42)
    v_np = rng.standard_normal(200).astype(np.float64)
    v_t = torch.from_numpy(v_np)

    out_t = torch_ema(v_t, p).cpu().numpy()
    ref = ema(v_np, p)

    # 前 p-1 个 torch_ema 是 0.0, numpy 是 NaN; 跳过 NaN 比对
    mask = ~np.isnan(ref)
    assert out_t.shape == ref.shape
    assert np.array_equal(out_t[mask], ref[mask]), (
        f"p={p} bit-equal fail; first diff idx = "
        f"{np.argwhere(out_t[mask] != ref[mask])[:3].tolist()}"
    )


def test_torch_ema_short_input():
    """n < p 全 0.0 (无 NaN, 跟 ema_step 一致)"""
    v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    out = torch_ema(v, 5)
    assert out.shape == (3,)
    assert torch.all(out == 0.0)


def test_torch_ema_preserves_dtype_device():
    """非 float64 输入被内部转 float64, device 保留"""
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
    out = torch_ema(v, 3)
    assert out.dtype == torch.float64
    assert out.device == v.device


# ============ 3. batched_step 等价于 per-combo step 循环 ============

def _run_per_combo_loop(bars: dict, mark: np.ndarray,
                        fast_grid: list, slow_grid: list) -> np.ndarray:
    """per-combo step 循环产出 sig[N, T] (跟 batched_step 同形, 用来对比)"""
    N = len(fast_grid)
    T = len(bars["close"])
    sig = np.zeros((N, T), dtype=np.int8)
    for ci, (fp, sp) in enumerate(zip(fast_grid, slow_grid)):
        s = MACrossoverStrategy().init_state({"fast": fp, "slow": sp})
        strat = MACrossoverStrategy()
        for t in range(T):
            bar = {
                "ts": int(t),
                "o": float(bars["open"][t]),
                "h": float(bars["high"][t]),
                "l": float(bars["low"][t]),
                "c": float(bars["close"][t]),
                "v": float(bars["volume"][t]),
                "mark": int(mark[t]),
            }
            s, x = strat.step(s, bar, {"fast": fp, "slow": sp})
            sig[ci, t] = x
    return sig


def _run_batched(bars: dict, mark: np.ndarray,
                 fast_grid: list, slow_grid: list) -> np.ndarray:
    N = len(fast_grid)
    T = len(bars["close"])
    device = torch.device("cpu")
    bars_t = {
        "ts": torch.arange(T, dtype=torch.int64, device=device),
        "o": torch.zeros(T, dtype=torch.float64, device=device),
        "h": torch.tensor(bars["high"], dtype=torch.float64, device=device),
        "l": torch.tensor(bars["low"], dtype=torch.float64, device=device),
        "c": torch.tensor(bars["close"], dtype=torch.float64, device=device),
        "v": torch.zeros(T, dtype=torch.float64, device=device),
        "mark": torch.tensor(mark, dtype=torch.int8, device=device),
    }
    params_t = {
        "fast": torch.tensor(fast_grid, dtype=torch.int64, device=device),
        "slow": torch.tensor(slow_grid, dtype=torch.int64, device=device),
    }
    state_t = MABatchedState(
        fast_count=torch.zeros(N, dtype=torch.int64, device=device),
        slow_count=torch.zeros(N, dtype=torch.int64, device=device),
        prev_diff=torch.zeros(N, dtype=torch.float64, device=device),
        has_prev=torch.zeros(N, dtype=torch.bool, device=device),
    )
    _, sig = MACrossoverStrategy.batched_step(
        state_t, bars_t, params_t, n_combos=N, n_bars=T)
    return sig.cpu().numpy()


def test_ma_crossover_batched_step_equivalence_small():
    """N=4, T=50, fast/slow 网格: batched_step vs per-combo step 循环 sig bit-equal"""
    bars = _synthetic_arr(days=2, start_ymd="20260101", seed=42)
    T = len(bars["close"])
    mark = np.array([0] * min(10, T // 5) + [1] * (T - min(10, T // 5)),
                    dtype=np.int8)
    fast_grid = [3, 5, 10, 21]
    slow_grid = [20, 30, 50, 60]

    sig_loop = _run_per_combo_loop(bars, mark, fast_grid, slow_grid)
    sig_batch = _run_batched(bars, mark, fast_grid, slow_grid)

    assert sig_loop.shape == sig_batch.shape
    assert np.array_equal(sig_loop, sig_batch), (
        f"diff count={np.sum(sig_loop != sig_batch)}; "
        f"first diff: {np.argwhere(sig_loop != sig_batch)[:3].tolist()}"
    )


def test_ma_crossover_batched_mark_zero_handling():
    """mark=0 段: batched_step 内 fast/slow count 累积, sig 全 0 (跟 step 一致)"""
    T = 30
    bars = {
        "open": np.zeros(T), "high": np.ones(T) * 100.0,
        "low": np.ones(T) * 100.0,
        "close": np.linspace(100, 110, T),
        "volume": np.zeros(T),
    }
    mark = np.zeros(T, dtype=np.int8)  # 全段预热
    sig_batch = _run_batched(bars, mark, [5], [20])
    assert sig_batch.shape == (1, T)
    assert np.all(sig_batch == 0), "mark=0 全段 sig 应为 0"


# ============ 4. sweep 路由 ============

def test_sweep_routing_threadpool_when_small_grid():
    """N < 32 时 sweep 走 ThreadPool (batched 阈值未达)"""
    bars = _synthetic_arr(days=10, start_ymd="20260101", seed=42)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000,
            "init_cash": 200000, "init_position": 0, "scale": 1.0,
            "buy_pct": 0.0, "sell_pct": 0.0, "params": {}}
    combos = []
    for fp in [3, 5]:
        for sp in [20, 30]:
            combos.append({"fast": fp, "slow": sp})
    # 4 combos < 32 阈值; 不管 device 都走 ThreadPool
    df = sweep_mod.sweep(bars, base, combos, splits=None, device="cpu",
                         n_workers=1, strategy_name="ma_crossover", verbose=False)
    assert len(df) == 4
    assert "score" in df.columns


def test_sweep_routing_threadpool_for_channel_deviation():
    """channel_deviation 不覆写 batched_step, 任何 device 都走 ThreadPool"""
    bars = _synthetic_arr(days=5, start_ymd="20260101", seed=42)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000,
            "init_cash": 200000, "init_position": 0, "scale": 1.0,
            "buy_pct": 0.0, "sell_pct": 0.0, "params": {}}
    combos = [{"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5, "tf1": 21}]
    # 即使 N=1 也走 ThreadPool (channel 不覆写 hook)
    df = sweep_mod.sweep(bars, base, combos, splits=None, device="auto",
                         n_workers=1, strategy_name="channel_deviation", verbose=False)
    assert len(df) == 1


def test_sweep_routing_uses_batched_when_threshold_met(monkeypatch):
    """N >= 32 + 真覆写 + (mock) device=gpu + gpu_available: sweep 走 batched 路径

    不强制真实 CUDA: 通过 monkeypatch batched_sweep.run_batched 让 sweep
    走\"batched 路径\"代码段并验证 metrics 被填充正确。
    """
    bars = _synthetic_arr(days=10, start_ymd="20260101", seed=42)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000,
            "init_cash": 200000, "init_position": 0, "scale": 1.0,
            "buy_pct": 0.0, "sell_pct": 0.0, "params": {}}
    combos = [{"fast": fp, "slow": sp}
              for fp in [3, 5, 10, 21]
              for sp in [20, 30, 40, 60, 80]]   # 20 combos; 阈值 32 仍未达

    # monkeypatch gpu_available -> True, device resolution -> gpu
    monkeypatch.setattr("evtrade.backends.gpu_available", lambda: True)
    from evtrade import backends as bd
    monkeypatch.setattr(bd, "resolve_device",
                        lambda req, ok: "gpu" if ok and req != "cpu" else "cpu")

    captured = {}

    def fake_run_batched(bars, base, params_list, *, strategy_cls, device,
                         splits=None, fee_bp=5.0, lam=1.0, min_trades=30,
                         max_mdd=1.0, split_ymd=None, _progress_print=None):
        captured["called"] = True
        captured["n_combos"] = len(params_list)
        captured["device"] = device
        captured["strategy_cls"] = strategy_cls
        # 走真实 ThreadPool 路径的 metrics (借 sweep._run_one 等价调用)
        # 注: sweep 实际行为不传 strategy_params={} (它走 defaults), 这里复刻
        # 同样的行为以保证 batched 路径跟 ThreadPool 路径输出一致
        from evtrade.core.sweep import run_one_vectorized
        warm = int(base["start"]) * 1_000_000
        metrics = []
        for ci, p in enumerate(params_list):
            m = run_one_vectorized(
                bars, base["period"], warm,
                strategy_name="ma_crossover",
                strategy_params={k: v for k, v in p.items() if k in ("fast", "slow")},
                init_cash=base["init_cash"],
                init_position=base["init_position"],
                trade_qty=p["trade_qty"],
                scale=p.get("scale", 1.0),
                buy_pct=p.get("buy_pct", 0.0),
                sell_pct=p.get("sell_pct", 0.0),
            )
            metrics.append(m)
        return [metrics]

    monkeypatch.setattr("evtrade.core.batched_sweep.run_batched", fake_run_batched)
    df = sweep_mod.sweep(bars, base, combos, splits=None, device="gpu",
                         n_workers=1, strategy_name="ma_crossover", verbose=False)
    # N=20 < 32 阈值 -> use_batched=False -> 仍走 ThreadPool, captured 不变
    assert "called" not in captured
    assert len(df) == 20

    # 现在升到 N>=32: 真走 batched 路径
    combos_big = [{"fast": fp, "slow": sp}
                  for fp in [3, 5, 10, 21]
                  for sp in [20, 30, 40, 60, 80, 100, 150, 200]]   # 32 combos
    captured.clear()
    df = sweep_mod.sweep(bars, base, combos_big, splits=None, device="gpu",
                         n_workers=1, strategy_name="ma_crossover", verbose=False)
    assert captured.get("called") is True
    assert captured.get("n_combos") == 32
    assert captured.get("strategy_cls") is MACrossoverStrategy
    assert len(df) == 32


def test_sweep_batched_propagates_exceptions_immediately(monkeypatch):
    """batched_step 抛异常 -> sweep 立即抛 (不延后, 不吞)"""
    bars = _synthetic_arr(days=10, start_ymd="20260101", seed=42)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000,
            "init_cash": 200000, "init_position": 0, "scale": 1.0,
            "buy_pct": 0.0, "sell_pct": 0.0, "params": {}}
    combos = [{"fast": fp, "slow": sp}
              for fp in [3, 5, 10, 21]
              for sp in [20, 30, 40, 60]]   # 16 combos 仍 < 32 阈值

    monkeypatch.setattr("evtrade.backends.gpu_available", lambda: True)
    from evtrade import backends as bd
    monkeypatch.setattr(bd, "resolve_device",
                        lambda req, ok: "gpu" if ok and req != "cpu" else "cpu")

    def fake_run_batched_boom(*a, **kw):
        raise RuntimeError("simulated batched failure")

    monkeypatch.setattr("evtrade.core.batched_sweep.run_batched",
                        fake_run_batched_boom)

    # N=16 < 32 阈值: 仍走 ThreadPool, 不应抛
    df = sweep_mod.sweep(bars, base, combos, splits=None, device="gpu",
                         n_workers=1, strategy_name="ma_crossover", verbose=False)
    assert len(df) == 16

    # N >= 32: batched 路径被触发 -> 抛 RuntimeError 同步透传
    combos_big = [{"fast": fp, "slow": sp}
                  for fp in [3, 5, 10, 21]
                  for sp in [20, 30, 40, 60, 80, 100, 150, 200]]
    with pytest.raises(RuntimeError, match="simulated batched failure"):
        sweep_mod.sweep(bars, base, combos_big, splits=None, device="gpu",
                         n_workers=1, strategy_name="ma_crossover", verbose=False)


def test_sweep_batched_fallback_to_cpu_when_no_cuda(monkeypatch):
    """gpu_available() == False -> use_batched False -> ThreadPool

    用 monkeypatch 强制 gpu_available() 返回 False, 不依赖实际环境.
    """
    # sweep 用的是 `from ..backends import gpu_available`, 必须 patch 源头
    monkeypatch.setattr("evtrade.backends.gpu_available", lambda: False)
    bars = _synthetic_arr(days=10, start_ymd="20260101", seed=42)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000,
            "init_cash": 200000, "init_position": 0, "scale": 1.0,
            "buy_pct": 0.0, "sell_pct": 0.0, "params": {}}
    combos = [{"fast": fp, "slow": sp}
              for fp in [3, 5, 10, 21]
              for sp in [20, 30, 40, 60, 80, 100, 150, 200]]   # 32 combos
    # 强制 gpu_available() False (即便本机有 CUDA, 此测试要求走 ThreadPool)
    df = sweep_mod.sweep(bars, base, combos, splits=None, device="auto",
                         n_workers=1, strategy_name="ma_crossover", verbose=False)
    assert len(df) == 32


def test_sweep_batched_oom_falls_back(monkeypatch):
    """CUDA OOM at runtime -> 自动 fallback ThreadPool + warning"""
    bars = _synthetic_arr(days=10, start_ymd="20260101", seed=42)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000,
            "init_cash": 200000, "init_position": 0, "scale": 1.0,
            "buy_pct": 0.0, "sell_pct": 0.0, "params": {}}
    combos = [{"fast": fp, "slow": sp}
              for fp in [3, 5, 10, 21]
              for sp in [20, 30, 40, 60, 80, 100, 150, 200]]

    monkeypatch.setattr("evtrade.backends.gpu_available", lambda: True)
    from evtrade import backends as bd
    monkeypatch.setattr(bd, "resolve_device",
                        lambda req, ok: "gpu" if ok and req != "cpu" else "cpu")

    def fake_run_batched_oom(*a, **kw):
        raise RuntimeError("CUDA out of memory: simulated")

    monkeypatch.setattr("evtrade.core.batched_sweep.run_batched",
                        fake_run_batched_oom)

    # 异常字符串含 \"out of memory\" -> 触发 fallback
    df = sweep_mod.sweep(bars, base, combos, splits=None, device="gpu",
                         n_workers=1, strategy_name="ma_crossover", verbose=False)
    assert len(df) == 32  # ThreadPool fallback 成功填充


# ============ 5. batched_step 无 instance state ============

def test_batched_step_no_instance_state():
    """batched_step 是 classmethod, 不引入 self._xxx 实例状态"""
    import re
    from pathlib import Path
    strategies_dir = Path(evtrade.__file__).parent / "strategies"
    pattern = re.compile(r"\bbatched_step\b.*\bself\._")
    violators = []
    for py in strategies_dir.glob("*.py"):
        if py.name == "vectorized_base.py":
            continue
        text = py.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            violators.append(f"{py.name}:{line_no} {m.group(0)!r}")
    assert not violators, (
        f"batched_step MUST NOT 引入 self._xxx (state 必须由 caller 持有): {violators}"
    )


# ============ 6. sweep batched 输出 schema 跟 ThreadPool 一致 ============

def test_sweep_batched_output_schema_matches_threadpool(monkeypatch):
    """同 grid, batched 路径(用 fake_run_batched 转走 run_one_vectorized)
    vs ThreadPool 路径: DataFrame 列集完全一致, score/S/pareto/filter_pass 一致
    """
    bars = _synthetic_arr(days=20, start_ymd="20260101", seed=42)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000,
            "init_cash": 200000, "init_position": 0, "scale": 1.0,
            "buy_pct": 0.0, "sell_pct": 0.0, "params": {}}
    combos = [{"fast": fp, "slow": sp}
              for fp in [3, 5, 10, 21]
              for sp in [20, 30, 40, 60, 80, 100, 150, 200]]   # 32 combos

    # 路径 A: ThreadPool (n_workers=1 串行)
    df_a = sweep_mod.sweep(bars, base, combos, splits=None, device="cpu",
                           n_workers=1, strategy_name="ma_crossover", verbose=False)

    # 路径 B: monkeypatch 强制走 batched (gpu mock + 32 combos 触发)
    monkeypatch.setattr("evtrade.backends.gpu_available", lambda: True)
    from evtrade import backends as bd
    monkeypatch.setattr(bd, "resolve_device",
                        lambda req, ok: "gpu" if ok and req != "cpu" else "cpu")

    def fake_run_batched_through_threadpool(bars, base, params_list, *,
                                            strategy_cls, device, splits=None,
                                            fee_bp=5.0, lam=1.0, min_trades=30,
                                            max_mdd=1.0, split_ymd=None,
                                            _progress_print=None):
        # 走真实 ThreadPool 路径的 metrics (借 sweep._run_one 等价调用)
        # 注: 复刻 sweep 的实际行为 (传 strategy_params={}) 以保证两路径输出一致
        from evtrade.core.sweep import run_one_vectorized
        warm = int(base["start"]) * 1_000_000
        metrics = []
        for p in params_list:
            m = run_one_vectorized(
                bars, base["period"], warm,
                strategy_name="ma_crossover",
                strategy_params={k: v for k, v in p.items() if k in ("fast", "slow")},
                init_cash=base["init_cash"],
                init_position=base["init_position"],
                trade_qty=p["trade_qty"],
                scale=p.get("scale", 1.0),
                buy_pct=p.get("buy_pct", 0.0),
                sell_pct=p.get("sell_pct", 0.0),
            )
            metrics.append(m)
        return [metrics]

    monkeypatch.setattr("evtrade.core.batched_sweep.run_batched",
                        fake_run_batched_through_threadpool)
    df_b = sweep_mod.sweep(bars, base, combos, splits=None, device="gpu",
                           n_workers=1, strategy_name="ma_crossover", verbose=False)

    # 列集一致
    assert set(df_a.columns) == set(df_b.columns)
    # 行数一致
    assert len(df_a) == len(df_b) == 32
    # score / ann_net_min / S / pareto / filter_pass 数值一致 (按 fast+slow 排序后比)
    sort_cols = ["fast", "slow"]
    df_a_s = df_a.sort_values(sort_cols).reset_index(drop=True)
    df_b_s = df_b.sort_values(sort_cols).reset_index(drop=True)
    for col in ["score", "ann_net_min", "ann_net_mean", "S", "pareto",
                "filter_pass", "n_trades", "sharpe_min", "sortino_min",
                "calmar_max", "x_mdd_max", "max_dd_days_max"]:
        if col in df_a.columns:
            a = df_a_s[col].astype(float).tolist()
            b = df_b_s[col].astype(float).tolist()
            assert np.allclose(a, b, atol=1e-12, rtol=0), (
                f"col {col} diff: {[(i, a[i], b[i]) for i in range(len(a)) if abs(a[i]-b[i])>1e-12][:3]}"
            )


def test_sweep_batched_score_ranking_unchanged(monkeypatch):
    """batched 路径 score 排序 == ThreadPool 路径 (按 fast+slow 排序后比)"""
    bars = _synthetic_arr(days=20, start_ymd="20260101", seed=42)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000,
            "init_cash": 200000, "init_position": 0, "scale": 1.0,
            "buy_pct": 0.0, "sell_pct": 0.0, "params": {}}
    combos = [{"fast": fp, "slow": sp}
              for fp in [3, 5, 10, 21]
              for sp in [20, 30, 40, 60, 80, 100, 150, 200]]   # 32 combos

    df_a = sweep_mod.sweep(bars, base, combos, splits=None, device="cpu",
                           n_workers=1, strategy_name="ma_crossover", verbose=False)
    df_a_s = df_a.sort_values(["fast", "slow"]).reset_index(drop=True)
    rank_a = df_a_s.sort_values("score", ascending=False).reset_index(drop=True)

    # 路径 B 同上 fake_run_batched_through_threadpool
    monkeypatch.setattr("evtrade.backends.gpu_available", lambda: True)
    from evtrade import backends as bd
    monkeypatch.setattr(bd, "resolve_device",
                        lambda req, ok: "gpu" if ok and req != "cpu" else "cpu")

    def fake_run_batched_through_threadpool(bars, base, params_list, *,
                                            strategy_cls, device, splits=None,
                                            fee_bp=5.0, lam=1.0, min_trades=30,
                                            max_mdd=1.0, split_ymd=None,
                                            _progress_print=None):
        from evtrade.core.sweep import run_one_vectorized
        warm = int(base["start"]) * 1_000_000
        metrics = []
        for p in params_list:
            m = run_one_vectorized(
                bars, base["period"], warm,
                strategy_name="ma_crossover",
                strategy_params={k: v for k, v in p.items() if k in ("fast", "slow")},
                init_cash=base["init_cash"],
                init_position=base["init_position"],
                trade_qty=p["trade_qty"],
                scale=p.get("scale", 1.0),
                buy_pct=p.get("buy_pct", 0.0),
                sell_pct=p.get("sell_pct", 0.0),
            )
            metrics.append(m)
        return [metrics]

    monkeypatch.setattr("evtrade.core.batched_sweep.run_batched",
                        fake_run_batched_through_threadpool)
    df_b = sweep_mod.sweep(bars, base, combos, splits=None, device="gpu",
                           n_workers=1, strategy_name="ma_crossover", verbose=False)
    rank_b = df_b.sort_values("score", ascending=False).reset_index(drop=True)

    # 按 fast+slow 比 score 排序后的 fast/slow 序列应一致 (top-N)
    assert (rank_a["fast"].tolist() == rank_b["fast"].tolist())
    assert (rank_a["slow"].tolist() == rank_b["slow"].tolist())
