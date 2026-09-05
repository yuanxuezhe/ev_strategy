from __future__ import annotations
"""行情加载: MySQL 单次全量拉取 + 本地 npz 缓存 + 合成数据生成器

================================================================
✅  可改层模块  ✅
================================================================
本文件是内核路径的数据加载层 (与参考引擎的 MySQLBacktestFeed 平行):

  - load_bars(code, start, end): MySQL 拉数 + npz 缓存 (默认开 cache)
      缓存键 = (code, warmup_start, end); 命中后零数据库开销
  - synthetic_bars(days, start_ymd, seed): 确定性合成数据 (无库体验/测试)

新增数据源 (Tushare / AkShare / CSV / Parquet):
  1. 实现 fetch_xxx() 返回 numpy 数组 {stime, open, high, low, close, volume}
  2. 在 load_bars 里加分支 (例如 if source == 'tushare': ...)
  3. CLI 加 --data-source 参数
================================================================
"""

import os
from datetime import datetime, timedelta

import numpy as np

from .config import DB_URL, TABLE
from ..frozen.models import Bar

BAR_KEYS = ("stime", "open", "high", "low", "close", "volume")


def load_bars(code: str, start_ymd: str, end_ymd: str, warmup_days: int = 365,
              cache_dir: str = None, use_cache: bool = True,
              db_url: str = None, verbose: bool = True) -> dict:
    """加载 1m 行情 -> {stime:int64, open/high/low/close/volume:float64}

    stime 为 YYYYMMDDHHmmss 整数 (升序)。缓存键 = (code, 预热起点, end)。
    """
    warmup_start = (datetime.strptime(start_ymd, "%Y%m%d")
                    - timedelta(days=warmup_days)).strftime("%Y%m%d")
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{code}_{warmup_start}_{end_ymd}.npz")
        if use_cache and os.path.exists(cache_file):
            z = np.load(cache_file)
            arrays = {k: z[k] for k in BAR_KEYS}
            if verbose:
                print(f"  -- 缓存命中: {cache_file} ({len(arrays['stime'])} 根)", flush=True)
            return arrays

    df = _fetch(code, warmup_start, end_ymd, db_url or DB_URL)
    if len(df) == 0:
        raise RuntimeError(
            f"数据库无数据: code={code} 区间 [{warmup_start}, {end_ymd}] "
            f"(请检查 EVTRADE_DB_URL/网络, 或改用 --synthetic-days 合成数据)")
    arrays = {
        "stime": df["stime"].to_numpy(dtype="int64"),
        "open": df["open"].to_numpy(dtype="float64"),
        "high": df["high"].to_numpy(dtype="float64"),
        "low": df["low"].to_numpy(dtype="float64"),
        "close": df["close"].to_numpy(dtype="float64"),
        "volume": df["volume"].to_numpy(dtype="float64"),
    }
    if cache_file:
        np.savez_compressed(cache_file, **arrays)
        if verbose:
            print(f"  -- 缓存已写: {cache_file}", flush=True)
    if verbose:
        print(f"  -- 加载 {len(arrays['stime'])} 根 1m bar "
              f"[{warmup_start} ~ {end_ymd}] (含预热)", flush=True)
    return arrays


def _fetch(code: str, start_ymd: str, end_ymd: str, db_url: str):
    """单次全量查询 (代替参考实现的分段查询; 一次 fetchall 后在本地处理)"""
    import pandas as pd
    from sqlalchemy import create_engine
    seg_start = start_ymd + "000000"
    seg_end = end_ymd + "235959"
    sql = (f"SELECT stock_code, stime, open, high, low, close, volume "
           f"FROM {TABLE} WHERE stime >= '{seg_start}' AND stime <= '{seg_end}' "
           f"AND stock_code = '{code}' ORDER BY stime ASC")
    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if len(df):
        df = df.sort_values("stime").reset_index(drop=True)
    return df


def arrays_to_bars(arrays: dict, code: str = "SYN") -> list[Bar]:
    """内核数组 -> list[Bar] (参考引擎/测试用)"""
    return [Bar(stime=str(int(arrays["stime"][i])), code=code,
                open=float(arrays["open"][i]), high=float(arrays["high"][i]),
                low=float(arrays["low"][i]), close=float(arrays["close"][i]),
                volume=int(arrays["volume"][i]))
            for i in range(len(arrays["stime"]))]


# ============ 合成数据 (确定性, 无库环境测试/演示) ============

def synthetic_bars(days: int = 30, start_ymd: str = "20250101", seed: int = 42,
                   s0: float = 10.0, jump_prob: float = 0.02,
                   odd_seconds_prob: float = 0.1) -> list[Bar]:
    """合成 1m bar: 均值回复随机游走 + 偶发跳空 (制造通道偏离), 少量非整分秒
    (练习桶边界逻辑)。A股时段 09:30-11:29 / 13:00-14:59, 每天 240 根, 含周末
    (纯测试用途, 保证跨日/跨月边界覆盖)。
    """
    rng = np.random.default_rng(seed)
    # 日内分钟槽 (升序): 09:30-11:29 + 13:00-14:59
    slots = ([f"09{m:02d}" for m in range(30, 60)]
             + [f"10{m:02d}" for m in range(0, 60)]
             + [f"11{m:02d}" for m in range(0, 30)]
             + [f"13{m:02d}" for m in range(0, 60)]
             + [f"14{m:02d}" for m in range(0, 60)])
    d0 = datetime.strptime(start_ymd, "%Y%m%d")
    bars: list[Bar] = []
    price = s0
    for day in range(days):
        ds = (d0 + timedelta(days=day)).strftime("%Y%m%d")
        for mi in slots:
            sec = "30" if rng.random() < odd_seconds_prob else "00"
            ret = 0.0005 * (s0 - price) / s0 + rng.normal(0.0, 0.003)
            if rng.random() < jump_prob:
                ret += (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(0.015, 0.04)
            o = price
            c = price * (1.0 + ret)
            hi = max(o, c) * (1.0 + abs(rng.normal(0.0, 0.001)))
            lo = min(o, c) * (1.0 - abs(rng.normal(0.0, 0.001)))
            vol = int(rng.integers(100, 20000))
            bars.append(Bar(stime=ds + mi + sec, code="SYN",
                            open=float(o), high=float(hi), low=float(lo),
                            close=float(c), volume=vol))
            price = c
    return bars
