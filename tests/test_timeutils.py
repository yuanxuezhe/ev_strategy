from __future__ import annotations
"""桶时间戳算法测试:
  A. 任意周期: 内核整数版 vs compute_bucket_general (datetime 参考实现)
  B. 老 7 周期: 通用算法 vs 原始"字段取整"算法 (行为不变性的等价性证明)
"""

import random
from datetime import date, datetime, timedelta

from evtrade.config import PERIODS
from evtrade.kernel import (_days_from_civil, bucket_ts_encoded,
                            epoch_to_encoded, encoded_to_epoch,
                            resolve_period_seconds)
from evtrade.timeutils import compute_bucket, compute_bucket_general


def _sample_stimes(rng):
    """覆盖 年界/月界/闰日/日界 + 随机偏移 (含非整分秒)"""
    bases = ["20231230113059", "20240101000000", "20240228235959",
             "20240229000000", "20250105093500", "20251231235959",
             "20260101000001", "20260903145959"]
    out = list(bases)
    for b in bases:
        d = datetime.strptime(b, "%Y%m%d%H%M%S")
        for _ in range(60):
            off = rng.randint(-86400 * 2, 86400 * 2)
            out.append((d + timedelta(seconds=off)).strftime("%Y%m%d%H%M%S"))
    return out


LEGACY_PERIODS = list(PERIODS.keys())
ARBITRARY_PERIODS = ["2m", "7m", "13m", "45m", "90m", "120m", "2h", "5h",
                     "7h", "12h", "3d", "7d"]


def test_bucket_matches_general_reference_all_periods():
    """内核整数版 vs datetime 参考实现: 老 7 周期 + 任意周期 全部一致"""
    rng = random.Random(7)
    stimes = _sample_stimes(rng)
    for period in LEGACY_PERIODS + ARBITRARY_PERIODS:
        P = resolve_period_seconds(period)
        for s in stimes:
            ref = compute_bucket_general(s, P)
            got = str(bucket_ts_encoded(int(s), P))
            assert got == ref, f"period={period} stime={s} ref={ref} got={got}"


def test_bucket_legacy_equivalence_on_original_periods():
    """老 7 周期: 通用算法 == 原始字段取整算法 (行为不变性证明)"""
    rng = random.Random(7)
    stimes = _sample_stimes(rng)
    for period, (unit, value, pos, delta) in PERIODS.items():
        for s in stimes:
            legacy = compute_bucket(s, value, pos, delta)
            got = str(bucket_ts_encoded(int(s), resolve_period_seconds(period)))
            assert got == legacy, f"period={period} stime={s} legacy={legacy} got={got}"


def test_resolve_period_seconds():
    assert resolve_period_seconds("5m") == 300
    assert resolve_period_seconds("90m") == 5400
    assert resolve_period_seconds("2h") == 7200
    assert resolve_period_seconds("3d") == 259200
    assert resolve_period_seconds("1m") == 60
    import pytest
    for bad in ("5x", "m", "0m", "-3h", "5.5m", ""):
        with pytest.raises(ValueError):
            resolve_period_seconds(bad)


def test_days_from_civil_matches_datetime():
    """Hinnant 历法 vs datetime, 2000~2030 逐日核对"""
    epoch_ord = date(1970, 1, 1).toordinal()
    cur = date(2000, 1, 1)
    end = date(2030, 1, 1)
    one = timedelta(days=1)
    while cur < end:
        assert _days_from_civil(cur.year, cur.month, cur.day) == cur.toordinal() - epoch_ord
        cur += one


def test_epoch_roundtrip():
    """14位整数 <-> epoch 秒 往返一致"""
    rng = random.Random(3)
    for _ in range(3000):
        e = rng.randint(0, 2 ** 31)
        t = int(epoch_to_encoded(e))
        assert encoded_to_epoch(t) == e


def test_bucket_semantics_samples():
    """语义抽查: 5m 右端点 / 1d 次日零点 / 90m 跨小时连续分区 / 2h 对齐"""
    P5 = resolve_period_seconds("5m")
    assert str(bucket_ts_encoded(20250105100730, P5)) == "20250105101000"
    assert str(bucket_ts_encoded(20250105100500, P5)) == "20250105100500"
    assert str(bucket_ts_encoded(20250105100459, P5)) == "20250105100500"
    P1D = resolve_period_seconds("1d")
    assert str(bucket_ts_encoded(20250105093500, P1D)) == "20250106000000"
    # 90m: 以午夜为锚, 09:30 属于 (09:00, 10:30]
    P90 = resolve_period_seconds("90m")
    assert str(bucket_ts_encoded(20250106093000, P90)) == "20250106103000"
    assert str(bucket_ts_encoded(20250106102900, P90)) == "20250106103000"
    assert str(bucket_ts_encoded(20250106103000, P90)) == "20250106103000"
    assert str(bucket_ts_encoded(20250106103100, P90)) == "20250106120000"
    # 2h: 对齐钟面
    P2H = resolve_period_seconds("2h")
    assert str(bucket_ts_encoded(20250106132500, P2H)) == "20250106140000"
