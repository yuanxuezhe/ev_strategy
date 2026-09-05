from __future__ import annotations
"""Feed 注册表: 按 key 获取 Feed 实例

================================================================
✅  可改层 (feeds 子包)  ✅
================================================================
新增 Feed: 在子包任意模块加 @register_feed("xxx") 类,
会自动进入 _FEEDS 表, CLI 可用 --feed xxx 调用。

当前已注册:
  - mysql_history: MySQLBacktestFeed (历史回测)
  - chained: ChainedFeed (实盘拼接)
"""
from typing import Callable, Type

from .base import Feed

_FEEDS: dict[str, Type[Feed]] = {}


def register_feed(name: str) -> Callable[[Type[Feed]], Type[Feed]]:
    """@register_feed("mysql_history") 装饰器"""
    def deco(cls: Type[Feed]) -> Type[Feed]:
        if name in _FEEDS:
            raise ValueError(f"Feed 名 {name!r} 已注册为 {_FEEDS[name].__name__}")
        _FEEDS[name] = cls
        return cls
    return deco


def get_feed(name: str, **kwargs) -> Feed:
    """按 key 构造 Feed 实例"""
    if name not in _FEEDS:
        raise ValueError(f"未知 Feed {name!r}; 可用: {list(_FEEDS)}")
    return _FEEDS[name](**kwargs)


def available_feeds() -> list[str]:
    return sorted(_FEEDS.keys())


# ============ 触发 @register_feed 装饰器副作用 ============
from . import mysql_history  # noqa: F401, E402  (注册 MySQLBacktestFeed)
from . import chained  # noqa: F401, E402  (注册 ChainedFeed)
