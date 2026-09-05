"""execution 子包: 下单撮合

公开 API: Executor / SimulatedExecutor / BrokerExecutor
实盘接入: 实现 BrokerExecutor.trade, 接券商 API + 成交回报
"""
from .base import BrokerExecutor, Executor, SimulatedExecutor

__all__ = ["Executor", "SimulatedExecutor", "BrokerExecutor"]
