"""逐 bar 回测/实盘引擎 (实盘/对账用)

run(): feed → aggregator → on_bars → strategy.step(state, bar) → executor

策略在 step(state, bar, params) 内推 EMA / FSM state; state 由 engine 持有
跨调用持续; framework 不传任何指标。

Engine 不假设 info 键集; 信号行打印走 strategy.format_signal_line hook。
"""

from ..core.aggregator import BarAggregator
from ..execution.base import Executor
from ..feeds.base import Feed
from ..primitives import fmt, sig_to_side
from ..strategies.vectorized_base import VectorizedStrategy
from .config import INIT_CASH, INIT_POSITION, TRADE_QTY


def _bucket_bar(rec: dict) -> dict:
    """将 aggregator 回调的 1m bar dict 转 strategy 期望的桶级 dict"""
    return {
        "ts": rec["ts"],
        "o": rec["open"], "h": rec["high"],
        "l": rec["low"],  "c": rec["close"],
        "v": rec["volume"], "mark": rec.get("mark", 1),
    }


class Engine:
    """回测/实盘统一引擎 (framework 不假定指标; VectorizedStrategy 唯一契约)"""

    def __init__(self, feed: Feed, aggregator: BarAggregator,
                 strategy: VectorizedStrategy, executor: Executor,
                 verbose: bool = True, **legacy):
        self.feed = feed
        self.aggregator = aggregator
        self.strategy = strategy
        self.executor = executor
        self.verbose = verbose
        # 桶级信号轨迹 (供 replay/reconcile 对账用)
        self.bucket_signals: list[int] = []
        # 上一根 1m bar 的 cur 快照 (检测桶切换)
        self._last_cur: dict | None = None
        # 策略持久 state (strategy-step-only); engine 持有跨调用持续
        self._state = strategy.init_state(strategy.params)
        # 让 aggregator 的回调指向自己
        self.aggregator.on_bars = self.on_bars

    def _process_bucket(self, rec: dict):
        """桶 finalized 后: 算一次信号 + 可选下单/打印"""
        if rec.get("mark", 1) != 1:
            return
        price = float(rec["close"])
        self.executor.update_price(price)
        self._state, sig_int = self.strategy.step(
            self._state, _bucket_bar(rec), self.strategy.params)
        self.bucket_signals.append(int(sig_int))
        if sig_int != 0:
            signal = sig_to_side(sig_int)
            self.executor.trade(signal, price, rec["ts"])
            if self.verbose:
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

    def print_summary(self):
        """回测结束: 策略总资产 vs 不操作基线, 算盈亏"""
        acc = self.executor.account
        price = acc.last_price
        final_total = acc.equity(price)
        baseline = acc.baseline_equity(price)
        diff = final_total - baseline
        pct = (diff / baseline * 100) if baseline else 0

        print("\n" + "=" * 60)
        print("回测盈亏汇总")
        print("=" * 60)
        print(f"期末价 (最后一根close) : {fmt(price)}")
        print(f"交易次数              : {len(acc.trades)}")
        print(f"期初资金 / 期初持仓    : {acc.init_cash:.0f} / {acc.init_position:.0f}股")
        print(f"期末资金 / 期末持仓    : {acc.cash:.2f} / {acc.position:.0f}股")
        print(f"期末持仓市值           : {acc.position * price:.2f}")
        print(f"策略总资产 (资金+市值) : {final_total:.2f}")
        print(f"不操作基线 (资金+市值) : {baseline:.2f}")
        print(f"盈亏差额 (策略-基线)   : {diff:+.2f}")
        print(f"盈亏比例              : {pct:+.2f}%")
        print("=" * 60)


def build_engine(feed, *, period, warmup_until=None,
                 strategy=None, strategy_name=None, strategy_params=None,
                 init_cash=INIT_CASH, init_position=INIT_POSITION,
                 trade_qty=TRADE_QTY, scale=1.0,
                 buy_pct=0.0, sell_pct=0.0, all_in=False,
                 verbose=False, executor=None,
                 **legacy) -> Engine:
    """装配 Feed → Aggregator → Strategy → Executor → Engine

    strategy: 可直接传 VectorizedStrategy 实例; 否则按 strategy_name +
              strategy_params 构造 (get_strategy)
    executor: 可直接传 Executor 实例; 否则新建 Account + SimulatedExecutor
    period:   "5m" 字符串或周期秒数 int
    """
    from ..execution.account import Account
    from ..timeutils import resolve_period_seconds
    from ..strategies import get_strategy

    if strategy is None:
        strategy = get_strategy(strategy_name, params=strategy_params or {})
    account = Account(cash=init_cash, position=init_position)
    if executor is None:
        from ..execution.base import SimulatedExecutor
        executor = SimulatedExecutor(account, qty=trade_qty, verbose=verbose,
                                     scale=scale, buy_pct=buy_pct,
                                     sell_pct=sell_pct, all_in=all_in)
    period_seconds = (resolve_period_seconds(period)
                      if isinstance(period, str) else period)
    aggregator = BarAggregator(period_seconds, on_bars=None,
                               warmup_until=warmup_until)
    return Engine(feed, aggregator, strategy, executor, verbose=verbose)