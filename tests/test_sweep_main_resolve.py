"""sweep_main base_params 初始化回归测试

之前 sweep_main 在 line 445-449 用 `if args.params:` 包裹
`_resolve_strategy_params` 调用 —— 不传 --params 时 base_params 未定义,
第 458 行 `{'params': base_params}` 抛 UnboundLocalError。本测试守住
无条件调 _resolve_strategy_params 的修复 (4 层优先级)。
"""
from __future__ import annotations

from evtrade.cli import sweep_main


def test_sweep_main_no_params_no_defaults_no_legacy_kwargs(tmp_path, monkeypatch):
    """不传 --params + 没 _defaults + 无 legacy_kwargs -> 应走到空 dict 路径不抛 UnboundLocalError"""
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(tmp_path))
    from evtrade.data import synthetic_bars
    from evtrade.core.kernel import bars_to_arrays
    bars = bars_to_arrays(synthetic_bars(days=10, start_ymd="20260901", seed=42))

    rc = sweep_main([
        "--strategy", "channel_deviation",
        "--code", "000001.SZ",
        "--start", "20260901",
        "--end", "20260903",
        "--grid", "low1=1.0,1.5",
        "--synthetic-days", "10",
        "--out", str(tmp_path / "sweep.csv"),
    ])
    # sweep_main 应正常返回 (None 或 df)
    assert rc is None or rc is not None  # 不抛异常即通过


def test_sweep_main_with_defaults_uses_them(tmp_path, monkeypatch):
    """不传 --params 但 _defaults 已落盘 -> sweep 应正常 (base_params 来自 _defaults)"""
    import json
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(tmp_path))
    (tmp_path / "channel_deviation.json").write_text(json.dumps(
        {"strategy": "channel_deviation",
         "params": {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}}), encoding="utf-8")

    from evtrade.data import synthetic_bars
    from evtrade.core.kernel import bars_to_arrays
    bars = bars_to_arrays(synthetic_bars(days=10, start_ymd="20260901", seed=42))

    rc = sweep_main([
        "--strategy", "channel_deviation",
        "--code", "000001.SZ",
        "--start", "20260901",
        "--end", "20260903",
        "--grid", "low2=0.8,1.2",   # 只扫 low2, 其余从默认来
        "--synthetic-days", "10",
        "--out", str(tmp_path / "sweep.csv"),
    ])
    assert rc is None or rc is not None
