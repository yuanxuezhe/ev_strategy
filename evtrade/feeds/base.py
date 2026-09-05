from __future__ import annotations
"""Feed 抽象基类

================================================================
✅  可改层 (feeds 子包)  ✅
================================================================
所有行情源的统一契约: stream() 从旧到新 yield Bar(stime, code, OHLCV)。
14 位 stime 字符串 + 单调升序是隐式不变量。
"""
from typing import Iterator

from ..frozen.models import Bar


class Feed:
    """行情源抽象: stream() 按时间从旧到新 yield Bar"""

    def stream(self) -> Iterator[Bar]:
        raise NotImplementedError
