"""阶段 3: 策略参数 dict 通用化测试"""
from __future__ import annotations

import pytest

from evtrade.strategies import (get_strategy, get_strategy_param_spec,
                                 available_strategies)


def test_params_dict_form():
    """dict 形式传参"""
    s = get_strategy("channel_deviation",
                      params={"low1": 2.0, "low2": 1.0, "high1": 2.0, "high2": 0.8})
    assert s.low1 == 2.0
    assert s.low2 == 1.0
    assert s.high1 == 2.0
    assert s.high2 == 0.8


def test_params_kwargs_form():
    """kwargs 形式传参 (兼容旧 API)"""
    s = get_strategy("channel_deviation", low1=1.8, low2=1.0)
    assert s.low1 == 1.8
    assert s.low2 == 1.0
    # 默认值仍生效
    assert s.high1 == 1.5
    assert s.high2 == 0.5


def test_params_mixed_form():
    """dict + kwargs 混合: kwargs 优先 (填默认)"""
    s = get_strategy("channel_deviation",
                      params={"low1": 2.0},
                      low2=1.5)
    assert s.low1 == 2.0  # 来自 dict
    assert s.low2 == 1.5  # 来自 kwargs
    assert s.high1 == 1.5  # 默认


def test_params_default():
    """全默认"""
    s = get_strategy("channel_deviation")
    assert s.low1 == 1.5 and s.low2 == 1.0
    assert s.high1 == 1.5 and s.high2 == 0.5


def test_unknown_param_rejected():
    """未声明参数报错"""
    with pytest.raises(ValueError, match="未声明的参数"):
        get_strategy("channel_deviation", params={"bogus": 1.0})


def test_type_rejected():
    """类型错误报错"""
    with pytest.raises(TypeError, match="期望 float"):
        get_strategy("channel_deviation", params={"low1": "abc"})


def test_range_rejected():
    """超出 params_spec 范围报错"""
    with pytest.raises(ValueError, match="大于最大值"):
        get_strategy("channel_deviation", params={"low1": 200.0})


def test_param_spec_query():
    """get_strategy_param_spec 返回 schema"""
    spec = get_strategy_param_spec("channel_deviation")
    assert "low1" in spec
    assert spec["low1"]["default"] == 1.5
    assert spec["low1"]["type"] is float


def test_other_strategy_works():
    """非 channel_deviation 策略 (breakout) 同样走 dict 模式"""
    s = get_strategy("breakout", params={"lookback": 30, "breakout_pct": 0.002})
    assert s.lookback == 30
    assert s.breakout_pct == 0.002


def test_run_one_general_channel_deviation():
    """run_one_general 跑 channel_deviation (走参考引擎, 不依赖 kernel 固定参数)"""
    from evtrade.data import synthetic_bars
    from evtrade.core.kernel import bars_to_arrays
    from evtrade.core.sweep import run_one_general

    bars = bars_to_arrays(synthetic_bars(days=30, start_ymd="20241101", seed=42))
    m = run_one_general(bars, "5m", int("20241110") * 1_000_000,
                        200000.0, 200000.0, 10000.0,
                        strategy_name="channel_deviation",
                        strategy_params={"low1": 1.5, "low2": 1.0,
                                          "high1": 1.5, "high2": 0.5})
    assert "n_trades" in m
    assert "excess_pct" in m


def test_run_one_general_breakout():
    """run_one_general 跑 breakout (新策略, 完全通用)"""
    from evtrade.data import synthetic_bars
    from evtrade.core.kernel import bars_to_arrays
    from evtrade.core.sweep import run_one_general

    bars = bars_to_arrays(synthetic_bars(days=30, start_ymd="20241101", seed=42))
    m = run_one_general(bars, "5m", int("20241110") * 1_000_000,
                        200000.0, 200000.0, 10000.0,
                        strategy_name="breakout",
                        strategy_params={"lookback": 20, "breakout_pct": 0.001})
    assert "n_trades" in m
    assert "excess_pct" in m


def test_parse_grid_extra_keys():
    """parse_grid 支持 extra_keys (新策略 grid key)"""
    from evtrade.core.sweep import parse_grid
    combos = parse_grid(["lookback=10,20", "breakout_pct=0.001,0.002"],
                         extra_keys={"lookback", "breakout_pct"})
    assert len(combos) == 4
    assert all("lookback" in c and "breakout_pct" in c for c in combos)


def test_parse_grid_rejects_unknown_without_extra_keys():
    """parse_grid 默认仍拒绝未知 key"""
    from evtrade.core.sweep import parse_grid
    with pytest.raises(ValueError, match="不支持的网格参数"):
        parse_grid(["lookback=10"])


def test_sweep_breakout_with_grid():
    """端到端: sweep breakout + 自定义 grid"""
    from evtrade.data import synthetic_bars
    from evtrade.core.kernel import bars_to_arrays
    from evtrade.core.sweep import sweep, parse_grid

    bars = bars_to_arrays(synthetic_bars(days=60, seed=11))
    base = {"start": "20241120", "period": "5m",
            "trade_qty": 10000.0, "init_cash": 200000.0, "init_position": 200000.0,
            "low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5,
            "params": {"lookback": 20, "breakout_pct": 0.001}}
    combos = parse_grid(["lookback=15,20", "breakout_pct=0.001"],
                         extra_keys={"lookback", "breakout_pct"})
    df = sweep(bars, base, combos, splits=["20241210"], n_workers=2,
                strategy_name="breakout")
    # rows = combos 数 (单 split 时 train + test 共享 combos)
    assert len(df) == 2
    assert "lookback" in df.columns
    assert "breakout_pct" in df.columns
    # 多窗 (train_/test_/test2_) 都应存在
    assert "train_n_trades" in df.columns or "test_n_trades" in df.columns


def test_sweep_params_as_base_only():
    """--params 提供基础值, --grid 在其上扫描; 未指定 key 沿用基础值"""
    from evtrade.data import synthetic_bars
    from evtrade.core.kernel import bars_to_arrays
    from evtrade.core.sweep import sweep, parse_grid

    bars = bars_to_arrays(synthetic_bars(days=60, seed=42))
    base = {"start": "20241120", "period": "5m",
            "trade_qty": 10000.0, "init_cash": 200000.0, "init_position": 200000.0,
            # params 提供基础值, grid 只扫 lookback
            "params": {"lookback": 20, "breakout_pct": 0.001}}
    combos = parse_grid(["lookback=10,30"],
                         extra_keys={"lookback", "breakout_pct"})
    df = sweep(bars, base, combos, n_workers=2, strategy_name="breakout")
    # 只 2 行 (lookback=10 与 lookback=30)
    assert len(df) == 2
    # breakout_pct 沿用基础值 0.001
    assert df["breakout_pct"].tolist() == [0.001, 0.001]
    assert sorted(df["lookback"].tolist()) == [10.0, 30.0]
    # params 字段不重复
    assert "params" not in df.columns


def test_sweep_no_params_full_cartesian():
    """不传 --params, --grid 笛卡尔积"""
    from evtrade.data import synthetic_bars
    from evtrade.core.kernel import bars_to_arrays
    from evtrade.core.sweep import sweep, parse_grid

    bars = bars_to_arrays(synthetic_bars(days=60, seed=42))
    base = {"start": "20241120", "period": "5m",
            "trade_qty": 10000.0, "init_cash": 200000.0, "init_position": 200000.0,
            "params": {}}    # 无基础值
    combos = parse_grid(["lookback=10,20,30", "breakout_pct=0.0005,0.001"],
                         extra_keys={"lookback", "breakout_pct"})
    df = sweep(bars, base, combos, n_workers=2, strategy_name="breakout")
    # 笛卡尔积: 3 × 2 = 6 行
    assert len(df) == 6


def test_sweep_dev_trigger_uses_kernel_path():
    """DSL 策略 (dev_trigger) sweep 走 numba 特化内核 (步骤 3 通用路径)

    锁定: sweep 结果与直接调 run_one_dsl (同窗口同参数) 逐位一致;
    无 DSL 的策略才走参考引擎兜底。
    """
    from evtrade.data import synthetic_bars
    from evtrade.core.kernel import bars_to_arrays
    from evtrade.core.kernel_dsl import run_one_dsl
    from evtrade.core.sweep import sweep, parse_grid

    bars = bars_to_arrays(synthetic_bars(days=60, seed=11))
    base = {"start": "20241120", "period": "5m", "tf1": 21,
            "trade_qty": 10000.0, "init_cash": 200000.0, "init_position": 200000.0,
            "params": {"entry_dev": 0.5}}
    combos = parse_grid(["entry_dev=0.5,0.8"], extra_keys={"entry_dev"})
    df = sweep(bars, base, combos, splits=["20241210"], n_workers=2,
               strategy_name="dev_trigger", verbose=False)
    assert len(df) == 2
    assert "entry_dev" in df.columns
    assert "test_ann_net" in df.columns

    # test 窗 = 全量 bars + warmup 20241210: 抽 entry_dev=0.5 行与 run_one_dsl 对账
    row = df[df["entry_dev"] == 0.5].iloc[0]
    m = run_one_dsl(bars, "5m", int("20241210") * 1_000_000,
                    200000.0, 200000.0, 10000.0, tf1=21,
                    strategy_name="dev_trigger", strategy_params={"entry_dev": 0.5})
    assert int(row["test_n_trades"]) == m["n_trades"]
    assert float(row["test_final_equity"]) == m["final_equity"]
    assert float(row["test_excess_pct"]) == m["excess_pct"]
