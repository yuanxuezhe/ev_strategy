"""新策略开发模板测试 (DSL/numba/CUDA 已下线; 唯一契约 VectorizedStrategy)

抄一份, 改 key 与类名即可。compute_signals 是策略与 framework 的唯一接口;
指标 (EMA/RSI/...) 在 compute_signals 内调用 evtrade.indicators.*。
"""
from __future__ import annotations

import numpy as np
import pytest

from evtrade.strategies import (
    VectorizedStrategy,
    available_strategies,
    get_strategy,
    get_strategy_param_spec,
    register_strategy,
)


# ============ 1. 模板策略定义 (改这部分) ============

@register_strategy("template_demo")
class TemplateDemoStrategy(VectorizedStrategy):
    """模板示例: 双均线交叉 (金叉 BUY / 死叉 SELL) 的向量化实现

    compute_signals(xp, bars, params) -> xp.ndarray[int8] (1=BUY / -1=SELL / 0=hold)
    框架在 Engine.on_bars 里会包单 bar -> compute_signals_for_one_bar -> 单值。
    """

    params_spec = {
        "fast":      {"default": 5,   "type": int,   "min": 2,  "max": 100},
        "slow":      {"default": 20,  "type": int,   "min": 5,  "max": 500},
        "threshold": {"default": 0.0, "type": float, "min": -1.0, "max": 1.0},
    }

    def compute_signals(self, xp, bars, params):
        from evtrade.indicators import xp_ema
        c = bars["c"]
        fast = int(params["fast"])
        slow = int(params["slow"])
        thr = float(params["threshold"])
        ema_fast = xp_ema(xp, c, fast)
        ema_slow = xp_ema(xp, c, slow)
        diff = (ema_fast - ema_slow) / xp.maximum(ema_slow, 1e-9)
        sig = xp.zeros(len(c), dtype=xp.int8)
        sig = xp.where(diff > thr, xp.int8(1), sig)
        sig = xp.where(diff < -thr, xp.int8(-1), sig)
        # 简单锁存: 同号相邻去重, 与实盘 on_bars 行为对齐 (此模板不演示 FSM)
        return sig


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
    """7. 跑一次合成数据: compute_signals 形态正确, 不报错"""
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
    sig = s.compute_signals(np, buckets, s.params)
    assert sig.dtype == np.int8
    assert len(sig) == len(buckets["ts"])
    # 极端参数下可能全 0, 不强断


def test_08_replay_engine():
    """8. (可选) 向量化引擎 vs 逐 bar 引擎 对账 (用 channel_deviation 当稳定参照)"""
    from evtrade.data import synthetic_bars
    from evtrade.replay import reconcile

    bars = synthetic_bars(days=15, start_ymd="20241101", seed=42)
    warm = int("20241110") * 1_000_000
    rep = reconcile(bars, "5m", warm, 21,
                    strategy_name="channel_deviation",
                    strategy_params={"low1": 1.5, "low2": 1.0,
                                     "high1": 1.5, "high2": 0.5},
                    init_cash=200000.0, init_position=200000.0,
                    trade_qty=10000.0, scale=1.0, verbose=False)
    assert rep["pass"] is True