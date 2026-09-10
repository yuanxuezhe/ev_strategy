"""channel_deviation.format_signal_line 演示: 逐桶打印通道偏离信息

运行:  uv run python examples/demo_format_signal_line.py
"""
import numpy as np
from datetime import datetime, timedelta

from evtrade.account import Account
from evtrade.core._harness import ListBarFeed
from evtrade.core.aggregator import BarAggregator
from evtrade.core.engine import Engine
from evtrade.core.timeutils import resolve_period_seconds
from evtrade.execution import SimulatedExecutor
from evtrade.primitives import Bar
from evtrade.strategies import get_strategy


def main() -> None:
    # 1) 合成 600 根 1m bar (含预热段 + 策略期)
    rng = np.random.default_rng(42)
    n = 600
    start = datetime(2024, 1, 1, 9, 30, 0)
    ret = rng.normal(0, 0.003, n)
    close = 10.0 * np.exp(np.cumsum(ret))
    bars = []
    for i in range(n):
        t = start + timedelta(minutes=i)
        c = float(close[i])
        h = c * (1 + abs(rng.normal(0, 0.001)))
        l = c * (1 - abs(rng.normal(0, 0.001)))
        bars.append(Bar(
            stime=t.strftime("%Y%m%d%H%M%S"), code="TEST",
            open=c, high=h, low=l, close=c, volume=1000,
        ))

    # 2) 组装 Engine 全链路 -> 触发 strategy.format_signal_line
    strategy = get_strategy("channel_deviation", params={"tf1": 21})
    account = Account(cash=200_000.0, position=200_000.0)
    executor = SimulatedExecutor(account, qty=10_000.0, verbose=True)
    feed = ListBarFeed(bars)

    def _noop(_bars: list) -> None:
        # Engine.run 会覆盖 aggregator.on_bars; 这里仅占位
        pass

    agg = BarAggregator(resolve_period_seconds("5m"), on_bars=_noop,
                        warmup_until="20240101093000")
    engine = Engine(feed, agg, strategy, executor, tf1=21, verbose=True)
    engine.run()


if __name__ == "__main__":
    main()