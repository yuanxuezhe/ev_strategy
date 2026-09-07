from __future__ import annotations
"""回测/实盘统一引擎 (自 mysql_analyze_demo.py 原样迁移; 参考实现, 差分测试基准)

================================================================
✅  可改层模块  ✅  (但与 kernel 等价, 改时小心)
================================================================
本文件是参考引擎 (与 evtrade.kernel 等价, 由 tests/test_differential.py
与 tests/test_replay.py 锁定)。可改的部分:

  - print_summary() 输出格式: 任意改, 不影响差分
  - Engine.on_bars 内追加风控检查 (止损/止盈):
      在 "if signal:" 之前加 cur.H/L/C 与账户 position 的检查,
      若不满足风控则置 signal=None。

**不要改**的部分:
  - _sync_ema 增量推送逻辑 (与 kernel step 第 2 步等价)
  - mark=0 时只累积指标不驱动策略 (kernel 第 4 步等价)
  - 成交时点 = 信号当根 close, 价格 = cur["close"]
================================================================
"""

from ..frozen.aggregator import BarAggregator
from ..frozen.incremental_indicators import EMAChannel
from ..frozen.models import fmt
from ..execution.base import Executor
from ..feeds.base import Feed
from ..strategies.base import StrategyBase
from .config import INIT_CASH, INIT_POSITION, TF1, TRADE_QTY


# ============ 引擎 (连接 Feed → Aggregator → 策略 → 执行) ============

class Engine:
    """回测/实盘统一引擎

    run(): 从 feed 取 bar 喂 aggregator; aggregator 回调 on_bars;
           on_bars 内: 预热(mark=0)跳过, 否则算指标+策略, 信号触发 executor。
    """

    def __init__(self, feed: Feed, aggregator: BarAggregator,
                 strategy: StrategyBase, executor: Executor,
                 tf1: int = TF1, verbose=True):
        self.feed = feed
        self.aggregator = aggregator
        self.strategy = strategy
        self.executor = executor
        self.tf1 = tf1
        self.verbose = verbose
        # 增量 EMA 通道轨 (替代每根 O(n) 重算的 ema_channel)
        self.ema_ch = EMAChannel(tf1)
        self._pushed = 0        # 已 push 的闭合桶数 (对应 aggregator.bars 长度)
        # 让 aggregator 的回调指向自己
        self.aggregator.on_bars = self.on_bars

    def _sync_ema(self, closed_bars: list[dict]):
        """把新增的闭合桶 push 进 EMA 状态 (O(1)/桶)"""
        n = len(closed_bars)
        while self._pushed < n:
            b = closed_bars[self._pushed]
            self.ema_ch.push(b["high"], b["low"])
            self._pushed += 1

    def on_bars(self, bars: list[dict]):
        """聚合器回调: 计算指标 + 策略信号 + 执行 + 打印

        mark=0 (预热): 只 push 闭合桶进 EMA 状态 (指标预热), 不驱动策略。
        mark=1 (策略期): 增量算指标 + 驱动策略 + 打印。
        """
        closed = bars[:-1]           # 已闭合桶
        cur = bars[-1]               # 当前未闭合桶
        # 同步新增闭合桶到 EMA 状态 (预热期也需累积, 保证 mark=1 时指标就绪)
        self._sync_ema(closed)

        if cur.get("mark", 1) == 0:
            return   # 预热: 只累积指标历史, 不进入策略

        price = float(cur["close"])
        self.executor.update_price(price)

        # 增量 EMA: 基于已闭合序列 + 当前桶最新 H/L, O(1)
        up, dw = self.ema_ch.channel(cur["high"], cur["low"])
        signal, info = self.strategy.check(cur, up, dw)

        # 信号触发 -> 统一下单处理
        if signal:
            self.executor.trade(signal, price, cur["ts"])

        if not self.verbose:
            return

        # 信号行的字段与格式由策略自己负责 (Engine 不假设 info 的键)
        if signal:
            print(self.strategy.format_signal_line(cur, up, dw, signal, info),
                  flush=True)

    def run(self):
        total = 0
        try:
            for bar in self.feed.stream():
                self.aggregator.update(bar)
                total += 1
            self.aggregator.flush()
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


# ============ 装配工厂 (cli._run_ref / sweep.run_one_general 共用) ============

def build_engine(feed, *, period, warmup_until=None,
                 strategy=None, strategy_name=None, strategy_params=None,
                 init_cash=INIT_CASH, init_position=INIT_POSITION,
                 trade_qty=TRADE_QTY, scale=1.0,
                 buy_pct=0.0, sell_pct=0.0, all_in=False,
                 tf1=TF1, verbose=False, executor=None) -> Engine:
    """装配 Feed → Aggregator → Strategy → Executor → Engine

    参考引擎的四处装配 (_run_ref / run_one_general / replay_engine / 测试)
    本是同一套 Account+Executor+Strategy+Aggregator+Engine, 本工厂只统一
    "装配", 不统一"汇总" (各调用方的绩效口径不同)。

      - strategy 未给时按 strategy_name + strategy_params 构造 (get_strategy)
      - executor 未给时新建 Account + SimulatedExecutor (verbose 跟随 verbose)
      - period 接受 "5m" 字符串或周期秒数 int
      - warmup_until 为 stime 字符串阈值 (BarAggregator 约定), None 表示不预热
    """
    from ..frozen.account import Account
    from ..frozen.timeutils import resolve_period_seconds
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
    return Engine(feed, aggregator, strategy, executor, tf1=tf1, verbose=verbose)
