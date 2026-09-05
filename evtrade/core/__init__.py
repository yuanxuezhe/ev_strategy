from __future__ import annotations
"""core 子包: 主调度层

公开 API:
  main(argv)  统一 CLI 入口 (委托 evtrade.cli.main)

未来扩展方向:
  - 把 kernel / engine / sweep / replay / permutation 整体迁入
  - 提供统一的 run(feed, strategy, executor, params) -> Result 接口
"""
from .runner import main

__all__ = ["main"]
