from __future__ import annotations
"""回测/实盘统一引擎 (自 mysql_analyze_demo.py 原样迁移; 参考实现, 差分测试基准)"""

from .aggregator import BarAggregator
from .config import TF1
from .execution import Executor
from .feeds import Feed
from .indicators import EMAChannel
from .models import fmt
from .strategy import ChannelDeviationStrategy


# ============ 引擎 (连接 Feed → Aggregator → 策略 → 执行) ============

class Engine:
    """回测/实盘统一引擎

    run(): 从 feed 取 bar 喂 aggregator; aggregator 回调 on_bars;
           on_bars 内: 预热(mark=0)跳过, 否则算指标+策略, 信号触发 executor。
    """

    def __init__(self, feed: Feed, aggregator: BarAggregator,
                 strategy: ChannelDeviationStrategy, executor: Executor,
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
        self.executor.account.last_price = price

        # 增量 EMA: 基于已闭合序列 + 当前桶最新 H/L, O(1)
        up, dw = self.ema_ch.channel(cur["high"], cur["low"])
        signal, info = self.strategy.check(cur, up, dw)

        # 信号触发 -> 统一下单处理
        if signal:
            self.executor.trade(signal, price, cur["ts"])

        if not self.verbose:
            return

        low_dev = info.get("low_dev")
        high_dev = info.get("high_dev")
        low_dev_h = info.get("low_dev_h")
        high_dev_l = info.get("high_dev_l")
        low_hit_prev = info.get("low_hit_prev")
        high_hit_prev = info.get("high_hit_prev")

        prefix = f"{signal} >>> " if signal else "             "
        line = (f"{prefix}[{cur['ts']}] {cur['code']} | O:{cur['open']} H:{cur['high']} "
                f"L:{cur['low']} C:{cur['close']} | vol:{cur['volume']} x{cur['count']} | "
                f"UP={fmt(up)} DW={fmt(dw)} | "
                f"low_dev(L/DW)={fmt(low_dev)}% high_dev(H/UP)={fmt(high_dev)}% | "
                f"low_dev_h(H/DW)={fmt(low_dev_h)}% high_dev_l(L/UP)={fmt(high_dev_l)}% | "
                f"low_hit_prev={fmt(low_hit_prev)} high_hit_prev={fmt(high_hit_prev)}")
        if signal:
            print(line, flush=True)

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
