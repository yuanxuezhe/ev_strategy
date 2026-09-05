from __future__ import annotations
"""统一行情结构与格式化 (自 mysql_analyze_demo.py 原样迁移)"""

from dataclasses import dataclass


# ============ 统一数据结构 ============

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
