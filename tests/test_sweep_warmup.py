"""sweep 主线程 numba kernel 预热行为

覆盖:
  - DSL 路径下 (dev_trigger), sweep 主线程预热调用 dsl_kernel(name).run_backtest
    在 ThreadPoolExecutor 提交 futures 之前
  - 预热异常不中断 sweep (try/except + log warning)
"""
from __future__ import annotations

import numpy as np


def _make_minimal_bars(n=64):
    """构造 sweep 最小可用 bars"""
    stime = np.arange(20260101000000, 20260101000000 + n * 60000, 60000,
                      dtype=np.int64)
    return {
        "stime": stime,
        "open":  np.full(n, 100.0, dtype=np.float64),
        "high":  np.full(n, 101.0, dtype=np.float64),
        "low":   np.full(n, 99.0, dtype=np.float64),
        "close": np.full(n, 100.5, dtype=np.float64),
        "volume": np.full(n, 1000, dtype=np.int64),
    }


def _dev_trigger_combo():
    """dev_trigger 策略 1 参数 (x)"""
    return {"period": "5m", "init_cash": 200000.0, "init_position": 200000.0,
            "trade_qty": 10000.0, "tf1": 21, "x": 0.5}


def test_sweep_does_not_crash_on_warmup_failure(monkeypatch):
    """预热异常被 try/except 捕获, sweep 继续走正常路径"""
    from evtrade.core import sweep as sweep_mod

    bars = _make_minimal_bars(64)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000.0,
            "init_cash": 200000.0, "init_position": 200000.0}
    combos = [_dev_trigger_combo()]

    from evtrade.core import kernel_dsl
    real_make_state = kernel_dsl.make_state_general
    calls = {"n": 0}

    def boom_once(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated warmup failure")
        # 后续调用 (run_one_dsl 内) 走真实路径
        return real_make_state(*a, **kw)

    monkeypatch.setattr(kernel_dsl, "make_state_general", boom_once)

    df = sweep_mod.sweep(bars, base, combos, split_ymd=None, n_workers=1,
                         device="cpu", strategy_name="dev_trigger",
                         verbose=False)
    # 预热被捕获 + 后续 run_one_dsl 走真实路径 -> 出 DataFrame
    assert df is not None
    assert len(df) == len(combos)
    assert calls["n"] >= 2  # 至少: 1 次 warm-up 抛 + 1 次 run_one_dsl 真


def test_sweep_runs_dsl_strategy_end_to_end():
    """dev_trigger DSL 策略 + 主线程预热正常路径, sweep 跑通并返回结果"""
    from evtrade.core import sweep as sweep_mod

    bars = _make_minimal_bars(64)
    base = {"start": "20260101", "period": "5m", "trade_qty": 10000.0,
            "init_cash": 200000.0, "init_position": 200000.0}
    combos = [_dev_trigger_combo()]

    df = sweep_mod.sweep(bars, base, combos, split_ymd=None, n_workers=1,
                         device="cpu", strategy_name="dev_trigger",
                         verbose=False)
    assert df is not None
    assert len(df) == len(combos)
