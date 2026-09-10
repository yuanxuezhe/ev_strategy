from __future__ import annotations
"""统一行情结构与格式化 (跨模块通用基础类型)

Bar 是所有模块共用的统一结构 (Feed 产 / Aggregator 消 / Engine 中转)。
stime 字段必须保持 14 位 YYYYMMDDHHmmss 字符串, 字典序 == 时间序。
"""

from dataclasses import dataclass


@dataclass
class Bar:
    """统一行情 bar (所有行情源转换成此结构, 解耦数据来源)"""
    stime: str
    code: str
    open: float
    high: float
    low: float
    close: float
    volume: int


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "----"


def sig_to_side(sig: int) -> str:
    """策略信号 int -> 字符串下单方向 (1=BUY, -1=SELL, 0=空)"""
    return {1: "BUY", -1: "SELL"}.get(sig, "")
