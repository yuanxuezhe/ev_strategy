"""新策略开发模板测试 (DSL/numba/CUDA 已下线; 唯一契约 VectorizedStrategy)

抄一份, 改 key 与类名即可。step(state, bar, params) -> (state, int) 是策略与
framework 的唯一接口; 指标 (EMA/RSI/...) 在 step 内调用 evtrade.indicators.*;
state 由 engine 持有 (dataclass), 策略无 instance attr。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from evtrade.indicators import EMAState, ema_step
from evtrade.strategies import (
    VectorizedStrategy,
    available_strategies,
    get_strategy,
    get_strategy_param_spec,
    register_strategy,
)
import pytest


# ============ 1. 模板策略定义 (改这部分) ============

@dataclass
class TemplateDemoState:
    """策略持久状态: fast EMA + slow EMA (由 engine 跨调用持有)"""
    fast: EMAState = field(default_factory=EMAState)
    slow: EMAState = field(default_factory=EMAState)


@register_strategy("template_demo")
class TemplateDemoStrategy(VectorizedStrategy):
    """模板示例: 双均线偏离 (fast>slow+thr → BUY / fast<slow-thr → SELL)

    step(state, bar, params) -> (new_state, sig); sig ∈ {1, -1, 0} (BUY/SELL/hold)
    引擎 (Engine.on_bars / vectorized) 循环调 step, state 跨调用持续。
    """

    params_spec = {
        "fast":      {"default": 5,   "type": int,   "min": 2,  "max": 100},
        "slow":      {"default": 20,  "type": int,   "min": 5,  "max": 500},
        "threshold": {"default": 0.0, "type": float, "min": -1.0, "max": 1.0},
    }

    def init_state(self, params):
        return TemplateDemoState()

    def step(self, state, bar, params):
        if bar["mark"] == 0:                      # 预热段不产信号
            return state, 0
        fast_p, slow_p = int(params["fast"]), int(params["slow"])
        thr = float(params["threshold"])
        state.fast, fast = ema_step(state.fast, float(bar["c"]), fast_p)
        state.slow, slow = ema_step(state.slow, float(bar["c"]), slow_p)
        # fast / slow 未就绪: 不产信号
        if state.fast.count < fast_p or state.slow.count < slow_p:
            return state, 0
        diff = (fast - slow) / max(slow, 1e-9)
        sig = 1 if diff > thr else (-1 if diff < -thr else 0)
        return state, sig


# ============ 2. 验证清单 ============

def test_01_registered():
    """1. 注册可见"""
    assert "template_demo" in available_strategies()


def test_02_three_call_forms():
    """2. 三种构造形式都工作"""
    # 字典
    s1 = get_strategy("template_demo", params={"fast": 3, "slow": 10})
    assert s1.fast == 3 and s1.slow == 10
    # kwargs
    s2 = get_strategy("template_demo", fast=7, slow=15)
    assert s2.fast == 7 and s2.slow == 15
    # 混合
    s3 = get_strategy("template_demo", params={"fast": 8}, slow=20)
    assert s3.fast == 8 and s3.slow == 20


def test_03_defaults_filled():
    """3. 默认值填充完整"""
    s = get_strategy("template_demo")
    spec = get_strategy_param_spec("template_demo")
    for k in spec:
        assert k in s.params
        assert s.params[k] == spec[k]["default"]


def test_04_type_rejected():
    """4. 类型错误抛 ValueError (VectorizedStrategy._resolve_params 统一抛 ValueError)"""
    with pytest.raises(ValueError, match="期望 int"):
        get_strategy("template_demo", params={"fast": "abc"})


def test_05_range_rejected():
    """5. 范围错误抛 ValueError"""
    with pytest.raises(ValueError, match="大于 max"):
        get_strategy("template_demo", params={"fast": 1000})
    with pytest.raises(ValueError, match="小于 min"):
        get_strategy("template_demo", params={"fast": 1})


def test_06_unknown_rejected():
    """6. 未声明参数抛 ValueError"""
    with pytest.raises(ValueError, match="未声明的参数"):
        get_strategy("template_demo", params={"bogus_key": 1.0})


def test_07_one_run():
    """7. 跑一次合成数据: step 循环产出合法 sig, 不报错"""
    import numpy as np
    from evtrade.data import synthetic_bars
    from evtrade.core.vectorized_engine import _aggregate_buckets

    s = get_strategy("template_demo", fast=3, slow=10)
    raw = synthetic_bars(days=30, start_ymd="20241101", seed=42)
    # 1m bars -> 桶级
    bars = {"stime": np.array([int(b.stime) for b in raw], dtype=np.int64),
            "open":   np.array([b.open for b in raw], dtype=np.float64),
            "high":   np.array([b.high for b in raw], dtype=np.float64),
            "low":    np.array([b.low for b in raw], dtype=np.float64),
            "close":  np.array([b.close for b in raw], dtype=np.float64),
            "volume": np.array([b.volume for b in raw], dtype=np.float64)}
    buckets = _aggregate_buckets(bars, "5m", warmup_until=0)
    # engine 循环调 step (state 跨调用持续)
    state = s.init_state(s.params)
    sigs = []
    for i in range(len(buckets["ts"])):
        bar = {"ts": int(buckets["ts"][i]), "o": float(buckets["o"][i]),
               "h": float(buckets["h"][i]), "l": float(buckets["l"][i]),
               "c": float(buckets["c"][i]), "v": float(buckets["v"][i]),
               "mark": int(buckets["mark"][i])}
        state, sig = s.step(state, bar, s.params)
        sigs.append(sig)
    assert len(sigs) == len(buckets["ts"])
    assert set(sigs).issubset({-1, 0, 1})
    # 极端参数下可能全 0, 不强断


def test_08_replay_engine():
    """8. (可选) 向量化引擎 vs 逐 bar 引擎 对账 (用 channel_deviation 当稳定参照)"""
    from evtrade.data import synthetic_bars
    from evtrade.replay import reconcile

    bars = synthetic_bars(days=15, start_ymd="20241101", seed=42)
    warm = int("20241110") * 1_000_000
    rep = reconcile(bars, "5m", warm,
                    strategy_name="channel_deviation",
                    strategy_params={"low1": 1.5, "low2": 1.0,
                                     "high1": 1.5, "high2": 0.5},
                    init_cash=200000.0, init_position=200000.0,
                    trade_qty=10000.0, scale=1.0, verbose=False)
    assert rep["pass"] is True
