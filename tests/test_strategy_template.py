"""新策略开发模板测试 (抄一份, 改 key 与类名即可)

演示: 一个最简单的"价格突破 N 根最高/最低"策略模板的测试。
新策略写完后, 复制这个文件, 改 class 名 + 注册 key 即可。
"""
from __future__ import annotations

import pytest

from evtrade.strategies import (StrategyBase, available_strategies,
                                 get_strategy, get_strategy_param_spec,
                                 register_strategy)


# ============ 1. 模板策略定义 (改这部分) ============

@register_strategy("template_demo")
class TemplateDemoStrategy(StrategyBase):
    """模板示例: 简单的双均线交叉 (金叉 BUY / 死叉 SELL)

    这里仅做参数 + 状态维护演示, 真实指标需从 cur["close"] 自行计算
    或从 indicators dict 拿 (如 indicators["ma_short"])。
    """

    params_spec = {
        "fast":      {"default": 5,   "type": int,   "min": 2,  "max": 100},
        "slow":      {"default": 20,  "type": int,   "min": 5,  "max": 500},
        "threshold": {"default": 0.0, "type": float, "min": -1.0, "max": 1.0},
    }

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params=params, **kwargs)
        self._bucket_ts = None
        self._acted = False
        # 简单均值状态
        self._closes: list[float] = []

    def _push(self, c: float):
        self._closes.append(c)
        n = max(self.fast, self.slow)
        if len(self._closes) > n:
            self._closes.pop(0)

    def check(self, cur, indicators=None, dw=None):
        if cur["ts"] != self._bucket_ts:
            self._bucket_ts = cur["ts"]
            self._acted = False
        self._push(cur["close"])
        if len(self._closes) < self.slow or self._acted:
            return None, {}

        fast_ma = sum(self._closes[-self.fast:]) / self.fast
        slow_ma = sum(self._closes[-self.slow:]) / self.slow
        diff = (fast_ma - slow_ma) / max(slow_ma, 1e-9)
        info = {"fast_ma": fast_ma, "slow_ma": slow_ma, "diff": diff}

        if diff > self.threshold:
            self._acted = True
            return "BUY", info
        if diff < -self.threshold:
            self._acted = True
            return "SELL", info
        return None, info


# ============ 2. 验证清单 (这部分的 8 项通常不用改) ============

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
    """4. 类型错误抛 TypeError"""
    with pytest.raises(TypeError, match="期望 int"):
        get_strategy("template_demo", params={"fast": "abc"})


def test_05_range_rejected():
    """5. 范围错误抛 ValueError"""
    with pytest.raises(ValueError, match="大于最大值"):
        get_strategy("template_demo", params={"fast": 1000})
    with pytest.raises(ValueError, match="小于最小值"):
        get_strategy("template_demo", params={"fast": 1})


def test_06_unknown_rejected():
    """6. 未声明参数抛 ValueError"""
    with pytest.raises(ValueError, match="未声明的参数"):
        get_strategy("template_demo", params={"bogus_key": 1.0})


def test_07_one_run():
    """7. 跑一次合成数据: 信号格式正确, 不报错"""
    from evtrade.data import synthetic_bars
    bars = synthetic_bars(days=30, start_ymd="20241101", seed=42)
    s = get_strategy("template_demo", fast=3, slow=10)
    n_sig = 0
    for b in bars:
        cur = {"ts": b.stime, "high": b.high, "low": b.low,
               "close": b.close, "open": b.open, "volume": b.volume,
               "code": b.code}
        sig, info = s.check(cur, {})
        if sig in ("BUY", "SELL"):
            n_sig += 1
            assert isinstance(info, dict) and len(info) > 0
    # 不断言 n_sig>0 (可能极端参数下不出信号), 只断言能跑通


def test_08_replay_engine():
    """8. (可选) 参考引擎回放对账"""
    from evtrade.data import synthetic_bars
    from evtrade.replay import reconcile

    bars = synthetic_bars(days=15, start_ymd="20241101", seed=42)
    warm = int("20241110") * 1_000_000
    rep = reconcile(bars, "5m", warm, 21, 1.5, 1.0, 1.5, 0.5,
                    init_cash=200000.0, init_position=200000.0,
                    trade_qty=10000.0, scale=1.0, verbose=False)
    # template_demo 不是 channel_deviation, 对账会跑但信号会差异
    # 这个测试只确认不崩; 真正的对账要写自己的策略级对账
    assert "pass" in rep
