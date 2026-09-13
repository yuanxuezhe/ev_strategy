"""新策略开发模板 (DSL/numba/CUDA 已下线; 唯一契约 VectorizedStrategy)

运行:  uv run python examples/strategy_template.py

抄这份, 改 key 与类名即可。step(state, bar, params) -> (state, int) 是策略与
framework 的唯一接口; 指标 (EMA / ATR / ...) 在 step 内调用 evtrade.indicators.*;
state 由 engine 持有 (dataclass), 策略无 instance attr。

注意: 此模板不调用 @register_strategy(), 避免污染全局注册表; 真正策略接入时
去掉注释即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from evtrade.indicators import EMAState, ema_step
from evtrade.primitives import Bar
from evtrade.strategies import VectorizedStrategy


# ============ 1. 策略持久 state ============

@dataclass
class TemplateDemoState:
    """策略持久状态: fast EMA + slow EMA"""
    fast: EMAState = field(default_factory=EMAState)
    slow: EMAState = field(default_factory=EMAState)


# ============ 2. 策略类 ============

class TemplateDemoStrategy(VectorizedStrategy):
    """模板示例: 双均线偏离 (fast>slow+thr → BUY / fast<slow-thr → SELL)"""

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


# ============ 3. 演示 (合成数据 + 手动驱动 step) ============

def main() -> None:
    """直接构造策略实例 + 合成 1m bar 流, 跑一遍 step 演示合约。

    不通过 CLI / Engine, 仅展示 'step(state, bar, params) -> (state, sig)' 接口。
    接入 framework 时: 用 get_strategy + VectorizedStrategy 注册装饰器, 然后跑
    backtest / sweep 即可。
    """
    strategy = TemplateDemoStrategy(params={"fast": 5, "slow": 20, "threshold": 0.001})
    state = strategy.init_state(strategy.params)

    # 构造 30 根合成 1m bar, c 价格做小步随机游走
    import random
    rng = random.Random(42)
    price = 10.0
    for minute in range(30):
        price += rng.gauss(0.0, 0.05)
        bar = Bar(stime=f"2026010109{minute:02d}00", code="DEMO",
                  open=price, high=price + 0.02, low=price - 0.02,
                  close=price, volume=100)
        bar_dict = {"ts": int(bar.stime), "o": bar.open, "h": bar.high,
                    "l": bar.low, "c": bar.close, "v": bar.volume,
                    "mark": 1 if minute >= 10 else 0}  # 前 10 根预热
        state, sig = strategy.step(state, bar_dict, strategy.params)
        if sig != 0:
            print(f"[{bar.stime}] sig={sig:+d}  close={bar.close:.4f}")


if __name__ == "__main__":
    main()
