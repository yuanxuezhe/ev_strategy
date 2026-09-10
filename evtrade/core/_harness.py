"""共享 fixture: 内部测试/回测用的 bar 流 adapter

调用方:
  - ListBarFeed: Bar list -> Bar 流 (passthrough), 被 replay.replay_engine / 测试 / examples 使用

Engine 消费任意 `stream() -> Iterator[Bar]` 的对象 (鸭子类型);
回测数据加载唯一入口是 core/data.load_bars / synthetic_bars。
"""
from __future__ import annotations

from typing import Iterator


class ListBarFeed:
    """Bar 列表 -> Bar 流 (passthrough)"""

    def __init__(self, bars: list):
        self.bars = bars

    def stream(self) -> Iterator:
        yield from self.bars
