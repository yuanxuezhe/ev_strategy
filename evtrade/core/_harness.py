"""共享 fixture: 内部测试/回测用的 feed 与 adapter

之前分散在多处:
  - evtrade/core/sweep.py     :: class _ArrFeed   (numpy dict -> Bar stream)
  - evtrade/core/replay.py    :: class _Feed       (Bar list -> Bar stream)
  - tests/test_differential.py :: class ListFeed  (Bar list -> Bar stream)
  - scripts/benchmark.py       :: class ListFeed  (Bar list -> Bar stream)

抽到本模块统一提供:
  - NumpyDictFeed: numpy OHLCV dict -> Bar stream (替代 sweep._ArrFeed)
  - ListBarFeed:   Bar list -> Bar stream        (替代 replay._Feed / ListFeed)

调用方约定:
  - 生产代码 sweep / replay 应改用本模块, 减少局部重复定义;
  - 测试与脚本代码不强求改 (它们是 leaf, 替换风险 > 收益), 后续单独 PR 清理。
"""
from __future__ import annotations

from typing import Iterator


class NumpyDictFeed:
    """numpy OHLCV dict -> Bar 对象流

    输入: dict 含键 stime/open:high:low:close:volume, 长度均为 n
    输出: 每次 yield 一个 frozen/models.Bar (与 mysql_history feed 同口径)
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
