"""跨字段 validator 测试 (spec: VectorizedStrategy.validators)

目的: 表达单字段 min/max 表达不了的硬约束 (如 channel_deviation 的 low1 > low2,
sweep 不再选出锁存失效的组合)。

覆盖:
  1. 策略类声明 validators 列表 (基类默认空)
  2. validators 在 _resolve_params 单字段检查通过后被调用
  3. channel_deviation 拒绝 low1 <= low2 / high1 <= high2
  4. CLI `--params` 入口触发 validator (端到端)
  5. sweep grid 入口预校验: 违反 validator 的 combo 被跳过 + warning
  6. sweep_results.csv 不含被跳过的 combo 行
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from evtrade.strategies import VectorizedStrategy, get_strategy


# ============ 1. 基类默认 validators 为空 ============

def test_base_class_validators_default_empty():
    """基类默认 validators 为空 list (不强制策略必须实现)"""
    assert VectorizedStrategy.validators == []
    assert isinstance(VectorizedStrategy.validators, list)


# ============ 2. channel_deviation 已声明 validator ============

def test_channel_deviation_declares_validator():
    """channel_deviation.validators MUST 包含 _validate_latch_order"""
    from evtrade.strategies.channel_deviation import (
        ChannelDeviationStrategy, _validate_latch_order,
    )
    assert _validate_latch_order in ChannelDeviationStrategy.validators


# ============ 3. channel_deviation 拒绝违反锁存约束的组合 ============

def test_channel_deviation_rejects_low1_le_low2():
    """low1 <= low2 抛 ValueError (锁存硬约束)"""
    with pytest.raises(ValueError, match="low1"):
        get_strategy("channel_deviation",
                     low1=1.0, low2=1.5,
                     high1=1.5, high2=0.5)


def test_channel_deviation_rejects_low1_eq_low2():
    """low1 == low2 也抛 ValueError (等于即锁存失效)"""
    with pytest.raises(ValueError, match="low1"):
        get_strategy("channel_deviation",
                     low1=1.0, low2=1.0,
                     high1=1.5, high2=0.5)


def test_channel_deviation_rejects_high1_le_high2():
    """high1 <= high2 抛 ValueError"""
    with pytest.raises(ValueError, match="high1"):
        get_strategy("channel_deviation",
                     low1=1.5, low2=1.0,
                     high1=0.5, high2=0.8)


def test_channel_deviation_accepts_valid_default():
    """默认 (1.5/1.0/1.5/0.5) 满足 low1>low2 / high1>high2, 不报错"""
    s = get_strategy("channel_deviation")
    assert s.low1 == 1.5 and s.low2 == 1.0
    assert s.high1 == 1.5 and s.high2 == 0.5


def test_channel_deviation_accepts_custom_valid():
    """自定义合法组合 (low1=2.0 > low2=1.5) 通过"""
    s = get_strategy("channel_deviation",
                     low1=2.0, low2=1.5,
                     high1=2.0, high2=0.8)
    assert s.low1 == 2.0 and s.low2 == 1.5


# ============ 4. CLI 入口触发 validator (端到端 smoke) ============

def test_cli_rejects_violating_params():
    """CLI --params low1:1.0;low2:1.5;high1:1.5;high2:0.5 触发 validator 报错"""
    import tempfile
    from io import StringIO
    from evtrade.cli import main as cli_main
    argv = ["backtest", "--strategy", "channel_deviation",
            "--device", "cpu", "--synthetic-days", "10",
            "--params", "low1:1.0;low2:1.5;high1:1.5;high2:0.5"]
    old_argv = sys.argv
    sys.argv = ["evtrade"] + argv
    buf = StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf
    try:
        with pytest.raises(ValueError, match="low1"):
            cli_main()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        sys.argv = old_argv


# ============ 5. sweep 入口预校验: 跳过违反 validator 的 combo ============

def test_sweep_skips_violating_combos():
    """sweep grid 含违反锁存约束的 low1/low2 组合 -> 预校验跳过, 其它合法组合继续"""
    from evtrade.core.data import synthetic_bars
    from evtrade.core.metrics import bars_to_arrays
    from evtrade.core.sweep import sweep

    bars = bars_to_arrays(synthetic_bars(days=15, start_ymd="20250101", seed=42))
    base = {"start": "20250101", "period": "5m",
            "trade_qty": 10000.0,
            "buy_pct": 0.0, "sell_pct": 0.0,
            "init_cash": 200000.0, "init_position": 200000.0,
            "params": {}}
    # low1=1.0, low2=1.5 违反 validator (一个); low1=1.5, low2=1.0 合法 (另一个)
    combos = [
        {"low1": 1.0, "low2": 1.5, "high1": 1.5, "high2": 0.5, "tf1": 21},
        {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5, "tf1": 21},
    ]
    df = sweep(bars, base, combos, split_ymd=None, n_workers=1,
               device="cpu", verbose=False, strategy_name="channel_deviation")
    # 只应有合法 combo 进入结果 (1 行)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["low1"] == 1.5 and row["low2"] == 1.0


def test_sweep_all_violating_raises():
    """全部 combo 违反 validator -> sweep 抛 ValueError (而不是输出空 CSV)"""
    from evtrade.core.data import synthetic_bars
    from evtrade.core.metrics import bars_to_arrays
    from evtrade.core.sweep import sweep

    bars = bars_to_arrays(synthetic_bars(days=10, start_ymd="20250101", seed=42))
    base = {"start": "20250101", "period": "5m",
            "trade_qty": 10000.0,
            "buy_pct": 0.0, "sell_pct": 0.0,
            "init_cash": 200000.0, "init_position": 200000.0,
            "params": {}}
    # 全部违反 validator
    combos = [
        {"low1": 1.0, "low2": 1.5, "high1": 1.5, "high2": 0.5, "tf1": 21},
        {"low1": 0.5, "low2": 1.0, "high1": 0.3, "high2": 0.8, "tf1": 21},
    ]
    with pytest.raises(ValueError, match="所有.*combo 都违反"):
        sweep(bars, base, combos, split_ymd=None, n_workers=1,
              device="cpu", verbose=False, strategy_name="channel_deviation")
