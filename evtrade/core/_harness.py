"""共享 fixture: 内部测试/回测用的 feed adapter

调用方:
  - ListBarFeed:   Bar list -> Bar 流 (passthrough), 被 replay.replay_engine 使用
  - NumpyDictFeed: numpy OHLCV dict -> Bar 流; 公共 fixture, 供测试 / 脚本构造合成数据
                   (sweep 现在直接吃 numpy dict, 不再经过此 feed)
"""
from __future__ import annotations

from typing import Iterator


class NumpyDictFeed:
    """numpy OHLCV dict -> Bar 对象流

    输入: dict 含键 stime/open:high:low:close:volume, 长度均为 n
    输出: 每次 yield 一个 Bar (与 mysql_history feed 同口径)
    """

    def __init__(self, bars: dict):
        self.bs = bars

    def stream(self) -> Iterator:
        from ..primitives import Bar
        bs = self.bs
        n = len(bs["stime"])
        for i in range(n):
            yield Bar(
                stime=str(int(bs["stime"][i])),
                code="SYN",
                open=float(bs["open"][i]),
                high=float(bs["high"][i]),
                low=float(bs["low"][i]),
                close=float(bs["close"][i]),
                volume=int(bs["volume"][i]),
            )


class ListBarFeed:
    """Bar 列表 -> Bar 流 (passthrough)"""

    def __init__(self, bars: list):
        self.bars = bars

    def stream(self) -> Iterator:
        yield from self.bars
