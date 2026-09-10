from __future__ import annotations
"""周期桶时间戳计算 (支持任意 m/h/d 周期)

桶算法 (epoch 锚定版, 通用实现, 支持任意 m/h/d 周期):
  * compute_bucket_general / bucket_ts_encoded ——
      e  = bar 的 naive-epoch 秒 (stime 即北京墙钟, 午夜天然对齐 86400 倍数)
      e0 = (e // P) × P                          (P = 周期秒数)
      桶 ts = e0 对应时刻 (恰在边界) 或 e0+P (前开后闭, 右端点标注)
    对 90m/7m/3d 等给出连续、无重叠、确定性的分区 (不整除天长的周期边界会
    相对钟面漂移, 属周期本身的性质, 与加密交易所做法一致)。

另有 14 位整数时间戳 <-> epoch 秒的纯整数历法 (标量 + numpy 向量两套)。
"""

import re
from datetime import datetime, timedelta

import numpy as np

_EPOCH0 = datetime(1970, 1, 1)
_PERIOD_RE = re.compile(r"^(\d+)(m|h|d)$")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}


def resolve_period_seconds(period: str) -> int:
    """周期字符串 -> 秒数。接受任意 '数字+m/h/d': 5m/7m/90m/2h/6h/3d/7d ..."""
    m = _PERIOD_RE.match(str(period).strip())
    if not m:
        raise ValueError(f"周期格式非法: {period!r} (应为如 5m / 90m / 2h / 3d)")
    v, u = int(m.group(1)), m.group(2)
    if v <= 0:
        raise ValueError(f"周期数值必须为正: {period!r}")
    return v * _UNIT_SECONDS[u]


def compute_bucket_general(stime: str, period_seconds: int) -> str:
    """通用合并桶时间戳 (前开后闭, 右端点标注), 支持任意 m/h/d 周期。

    stime 即北京墙钟时间, naive-epoch 午夜天然对齐 86400 倍数, 直接以其为锚:
    e0 = (e // P) × P; 恰在边界 → ts = e0, 否则 → ts = e0 + P。
    """
    dt = datetime.strptime(stime, "%Y%m%d%H%M%S")
    e = int((dt - _EPOCH0).total_seconds())
    e0 = (e // period_seconds) * period_seconds
    r = e - e0
    ts = _EPOCH0 + timedelta(seconds=e0 + (0 if r == 0 else period_seconds))
    return ts.strftime("%Y%m%d%H%M%S")


# ============ 14 位整数时间戳 <-> epoch 秒 (纯整数历法) ============

def _days_from_civil(y: int, m: int, d: int) -> int:
    """公历日期 -> 自 1970-01-01 的天数 (Howard Hinnant 算法)"""
    y = y - (1 if m <= 2 else 0)
    era = y // 400
    yoe = y - era * 400                                  # [0, 399]
    mp = (m + 9) % 12                                    # 3月=0
    doy = (153 * mp + 2) // 5 + d - 1                    # [0, 365]
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy        # [0, 146096]
    return era * 146097 + doe - 719468


def _civil_from_days(z: int):
    """自 1970-01-01 的天数 -> 公历日期 (y, m, d)"""
    z = z + 719468
    era = z // 146097
    doe = z - era * 146097                               # [0, 146096]
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + (3 if mp < 10 else -9)
    return y + (1 if m <= 2 else 0), m, d


def encoded_to_epoch(t: int) -> int:
    """YYYYMMDDHHmmss 整数 -> epoch 秒"""
    y = t // 10000000000
    mo = (t // 100000000) % 100
    d = (t // 1000000) % 100
    h = (t // 10000) % 100
    mi = (t // 100) % 100
    s = t % 100
    return _days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s


def epoch_to_encoded(e: int) -> int:
    """epoch 秒 -> YYYYMMDDHHmmss 整数"""
    days = e // 86400
    sod = e % 86400
    h = sod // 3600
    mi = (sod % 3600) // 60
    s = sod % 60
    y, mo, d = _civil_from_days(days)
    return ((((y * 100 + mo) * 100 + d) * 100 + h) * 100 + mi) * 100 + s


def bucket_ts_encoded(t: int, period_seconds: int) -> int:
    """合并桶时间戳 (前开后闭, 标注右端点), 任意 m/h/d 周期通用 (整数版)

    与 compute_bucket_general 同式 (字符串版), 给 vectorized 引擎用。
    """
    e = encoded_to_epoch(t)
    e0 = (e // period_seconds) * period_seconds
    if e == e0:
        return epoch_to_encoded(e0)
    return epoch_to_encoded(e0 + period_seconds)


# ============ 向量历法 (numpy int64 数组版, 与上方标量版同 Hinnant 真源) ============

def encoded_to_epoch_np(t: np.ndarray) -> np.ndarray:
    """YYYYMMDDHHmmss int64 数组 -> epoch 秒数组 (encoded_to_epoch 的向量化)"""
    y = t // 10_000_000_000
    mo = (t // 100_000_000) % 100
    d = (t // 1_000_000) % 100
    h = (t // 10_000) % 100
    mi = (t // 100) % 100
    s = t % 100
    yy = y - (mo <= 2)
    era = yy // 400
    yoe = yy - era * 400
    mp = (mo + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return (era * 146097 + doe - 719468) * 86400 + h * 3600 + mi * 60 + s


def epoch_to_encoded_np(e: np.ndarray) -> np.ndarray:
    """epoch 秒数组 -> YYYYMMDDHHmmss int64 数组 (epoch_to_encoded 的向量化)"""
    days = e // 86400
    sod = e % 86400
    h = sod // 3600
    mi = (sod % 3600) // 60
    s = sod % 60
    z = days + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + np.where(mp < 10, 3, -9)
    y2 = y + (m <= 2)
    return ((((y2 * 100 + m) * 100 + d) * 100 + h) * 100 + mi) * 100 + s
