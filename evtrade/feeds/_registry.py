from __future__ import annotations
"""Feed 注册表 (独立模块, 避免循环导入)

新增 Feed: 在子包任意模块加 @register_feed("xxx") 类。
"""
from typing import Callable, Type

from .base import Feed

_FEEDS: dict[str, Type[Feed]] = {}


def register_feed(name: str) -> Callable[[Type[Feed]], Type[Feed]]:
    def deco(cls: Type[Feed]) -> Type[Feed]:
        if name in _FEEDS:
            raise ValueError(f"Feed 名 {name!r} 已注册为 {_FEEDS[name].__name__}")
        _FEEDS[name] = cls
        return cls
    return deco


def get_feed(name: str, **kwargs) -> Feed:
    if name not in _FEEDS:
        raise ValueError(f"未知 Feed {name!r}; 可用: {list(_FEEDS)}")
    return _FEEDS[name](**kwargs)


def available_feeds() -> list[str]:
    return sorted(_FEEDS.keys())
