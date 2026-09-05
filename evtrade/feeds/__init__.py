from __future__ import annotations
"""feeds 子包: 行情源入口(历史 + 实时)

公开 API:
  Feed                  基类
  MySQLBacktestFeed     历史回测 (mysql_history)
  ChainedFeed           串联多个 Feed (chained)
  get_feed(name, **kw)  按 key 构造 (注册表驱动)
  available_feeds()     所有可用 key
  register_feed(name)   装饰器 (新增 Feed 时用)

用法:
  from evtrade.feeds import get_feed
  feed = get_feed("mysql_history", code="159992.SZ", start_ymd="20250101", end_ymd="20260903")
"""
# 1) 导入子模块, 触发 @register_feed 装饰器副作用
from .base import Feed
from .mysql_history import MySQLBacktestFeed  # noqa: F401  注册 mysql_history
from .chained import ChainedFeed                # noqa: F401  注册 chained
# 2) 暴露 registry API
from ._registry import get_feed, available_feeds, register_feed

__all__ = [
    "Feed", "MySQLBacktestFeed", "ChainedFeed",
    "get_feed", "available_feeds", "register_feed",
]
