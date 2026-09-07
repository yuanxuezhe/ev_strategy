from __future__ import annotations
"""统一行情结构与格式化 (跨模块通用基础类型)

================================================================
✅  可改层  ✅  (基础数据类型, 无差分锁定)
================================================================
Bar 是所有模块共用的统一结构 (Feed 产 / Aggregator 消 / Engine 中转)。
stime 字段必须保持 14 位 YYYYMMDDHHmmss 字符串, 字典序 == 时间序。

原 frozen/models.py (2026-09 重构迁移至此); 旧 evtrade.models 路径通过
__init__.py 的 sys.modules shim 仍可用。
================================================================
"""

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
