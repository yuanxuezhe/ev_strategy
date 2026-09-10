"""ChainedFeed: 串联多个 Feed (实盘标准用法)

先历史预热 (含 warmup_until 之前的指标累积), 再接实时源。
对 Engine 完全透明 —— 只看到一个连续的 Bar 流。
"""
from ._registry import register_feed
from .base import Feed


@register_feed("chained")
class ChainedFeed(Feed):
    """串联多个行情源: 先历史预热, 再接实时 (实盘用)

    例: ChainedFeed(MySQLBacktestFeed(...今天), LiveFeed(code))
    """

    def __init__(self, *feeds: Feed):
        self.feeds = feeds

    def stream(self):
        for feed in self.feeds:
            yield from feed.stream()