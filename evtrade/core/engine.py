"""逐 bar 步进引擎 (实盘/对账通用)

run(): feed → aggregator → on_bars → strategy.step(state, bar) → 累计 sig

策略在 step(state, bar, params) 内推 EMA / FSM state; state 由 engine 持有
跨调用持续; framework 不传任何指标。

Engine 不假设 info 键集; 信号行打印走 strategy.format_signal_line hook。
2026-09-13 重构: framework 不再调 Executor / trade_decision / Account;
策略 step 自负责撮合与记账 (framework 仅驱动 step + 累计 sig/state)。
"""
from __future__ import annotations

from ..core.aggregator import BarAggregator
from ..strategies.vectorized_base import VectorizedStrategy


def _bucket_bar(rec: dict) -> dict:
    """将 aggregator 回调的 1m bar dict 转 strategy 期望的桶级 dict"""
    return {
        "ts": rec["ts"],
        "o": rec["open"], "h": rec["high"],
        "l": rec["low"],  "c": rec["close"],
        "v": rec["volume"], "mark": rec.get("mark", 1),
    }


class Engine:
    """逐 bar 步进引擎 (framework 仅驱动 step + 累计 sig/state)"""

    def __init__(self, feed, strategy: VectorizedStrategy,
                 aggregator: BarAggregator | None = None,
                 verbose: bool = True, **legacy):
        # feed: 任意 stream() -> Iterator[Bar] 的 bar 流对象 (鸭子类型)
        self.feed = feed
        self.strategy = strategy
        # 兼容旧调用: aggregator 可显式传入, 否则按需构造
        if aggregator is None:
            from ..core.aggregator import BarAggregator as _Agg
            aggregator = _Agg()
        self.aggregator = aggregator
        self.verbose = verbose
        # 桶级信号轨迹
        self.bucket_signals: list[int] = []
        # 上一根 1m bar 的 cur 快照 (检测桶切换)
        self._last_cur: dict | None = None
        # 策略持久 state (strategy-step-only); engine 持有跨调用持续
        self._state = strategy.init_state(strategy.params)
        # 让 aggregator 的回调指向自己
        self.aggregator.on_bars = self.on_bars

    def _process_bucket(self, rec: dict):
        """桶 finalized 后: 算一次信号 + 打印"""
        if rec.get("mark", 1) != 1:
            return
        self._state, sig_int = self.strategy.step(
            self._state, _bucket_bar(rec), self.strategy.params)
        self.bucket_signals.append(int(sig_int))
        if sig_int != 0 and self.verbose:
            info = getattr(self.strategy, "_last_info", None)
            print(self.strategy.format_signal_line(rec["ts"], sig_int, info),
                  flush=True)

    def on_bars(self, bars: list[dict]):
        """聚合器回调: 桶 CLOSE 时驱动策略一次 (用 finalized OHLCV)"""
        cur = bars[-1]
        if self._last_cur is not None and self._last_cur["ts"] != cur["ts"]:
            self._process_bucket(self._last_cur)
        self._last_cur = cur

    def run(self):
        total = 0
        try:
            for bar in self.feed.stream():
                self.aggregator.update(bar)
                total += 1
            self.aggregator.flush()
            # flush 后手动触发最后一个桶的信号 (flush 不产生"下一桶切换")
            if self._last_cur is not None:
                self._process_bucket(self._last_cur)
        except KeyboardInterrupt:
            print("\n已停止")
        if self.verbose:
            print(f"\n完成, 共处理 {total} 根 1m bar")
        return total