from __future__ import annotations
"""周期桶时间戳计算与分段区间 (支持任意 m/h/d 周期)

两套桶算法:
  * compute_bucket (字符串/字段取整版) —— 原始实现, 仅对 1m/5m/15m/30m/1h/4h/1d
    这类"能对齐日历字段"的周期正确; 保留作为历史语义基准 (等价性测试用)。
    对 90m/7m/3d 等任意周期它要么分区错误 (>60m 的分钟字段无法对 90 取整),
    要么直接崩溃 (3d 在月初会拼出 day=00 的非法日期)。
  * compute_bucket_general (epoch 锚定版) —— 通用实现, 支持任意 m/h/d 周期:
      e  = bar 的 naive-epoch 秒 (stime 即北京墙钟, 午夜天然对齐 86400 倍数)
      e0 = (e // P) × P                          (P = 周期秒数)
      桶 ts = e0 对应时刻 (恰在边界) 或 e0+P (前开后闭, 右端点标注)
    对老 7 周期与 compute_bucket 逐例等价 (tests/test_timeutils.py 锁定);
    对 90m/7m/3d 等给出连续、无重叠、确定性的分区 (不整除天长的周期边界会
    相对钟面漂移, 属周期本身的性质, 与加密交易所做法一致)。
"""

import re
from datetime import datetime, timedelta
from functools import lru_cache

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
    语义与 compute_bucket 在老 7 周期上逐例一致 (能整除天长的周期, 边界恰好
    落在同一钟面位置)。
    """
    dt = datetime.strptime(stime, "%Y%m%d%H%M%S")
    e = int((dt - _EPOCH0).total_seconds())
    e0 = (e // period_seconds) * period_seconds
    r = e - e0
    ts = _EPOCH0 + timedelta(seconds=e0 + (0 if r == 0 else period_seconds))
    return ts.strftime("%Y%m%d%H%M%S")


@lru_cache(maxsize=200000)
def _bucket_plus_period(floor: str, delta: timedelta) -> str:
    """floor + 周期, datetime 进位规避 60/24/32 溢出 (缓存: 同周期桶 ts 重复)"""
    return (datetime.strptime(floor, "%Y%m%d%H%M%S") + delta).strftime("%Y%m%d%H%M%S")


def compute_bucket(stime: str, value: int, pos: int, delta: timedelta) -> str:
    """合并桶时间戳 (前开后闭, 标注右端点 close)

    floor = (num // value) * value, 后面补 0
    on_boundary (整除且后位全0) -> floor           (区间 (prev, floor] 右端点)
    否则                        -> floor + 周期    (区间 (floor, floor+周期] 右端点)
    +周期用 datetime 进位, 规避 60/24/32 溢出 (lru_cache 加速重复桶)
    """
    num = int(stime[pos:pos + 2])
    floor = stime[:pos] + f"{(num // value) * value:02d}" + "0" * (len(stime) - pos - 2)
    if (num % value == 0) and all(c == "0" for c in stime[pos + 2:]):
        return floor
    return _bucket_plus_period(floor, delta)


def daterange(start_ymd: str, end_ymd: str, step_days: int = 7):
    """从旧到新按 step_days 天生成 [seg_start, seg_end] 闭区间 (YYYYMMDDHHmmss)

    每段含 step_days 天: [dayN 00:00:00, dayN+step-1 23:59:59]
    末段对齐到 end_ymd 的 23:59:59, 不超界。
    例: start=20250101, step=7 -> [20250101000000, 20250107235959]
        下一段                  -> [20250108000000, 20250115235959]
    """
    fmt = "%Y%m%d"
    cur = datetime.strptime(start_ymd, fmt)
    last = datetime.strptime(end_ymd, fmt)
    while cur <= last:
        seg_end_day = min(cur + timedelta(days=step_days - 1), last)
        yield (cur.strftime("%Y%m%d") + "000000",
               seg_end_day.strftime("%Y%m%d") + "235959")
        cur = seg_end_day + timedelta(days=1)


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
