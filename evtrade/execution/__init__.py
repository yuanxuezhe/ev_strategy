"""execution 子包: 下单撮合

公开 API: trade_decision (单一成交决策) / Executor / SimulatedExecutor
实盘接入: 实现 Executor.trade, 接券商 API + 成交回报
"""
from .base import Executor, SimulatedExecutor, trade_decision

__all__ = ["trade_decision", "Executor", "SimulatedExecutor"]
