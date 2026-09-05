from __future__ import annotations
"""行情源 Feed (自 mysql_analyze_demo.py 原样迁移)"""

import time
from datetime import datetime, timedelta
from typing import Iterator

from sqlalchemy import create_engine, text

from .config import DB_URL, TABLE
from .models import Bar
from .timeutils import daterange


# ============ 行情源 Feed (统一流入接口) ============

class Feed:
    """行情源抽象: stream() 按时间从旧到新 yield Bar"""

    def stream(self) -> Iterator[Bar]:
        raise NotImplementedError


class MySQLBacktestFeed(Feed):
    """MySQL 历史行情源 (分段查询, 闭区间, 按证券代码筛选)

    包含预热: start 之前 warmup_days 天的行情也拉取 (供指标预热, 由 aggregator mark=0 标记)。
    """

    def __init__(self, code: str, start_ymd: str, end_ymd: str,
                 step_days: int = 7, warmup_days: int = 365,
                 delay: float = 0, verbose=True):
        self.code = code
        self.start_ymd = start_ymd
        self.end_ymd = end_ymd
        self.step_days = step_days
        self.warmup_days = warmup_days
        self.delay = delay
        self.verbose = verbose
        # 预热起点 = 策略起点 - warmup_days
        self.warmup_start = (datetime.strptime(start_ymd, "%Y%m%d")
                             - timedelta(days=warmup_days)).strftime("%Y%m%d")

    @property
    def warmup_until(self) -> str:
        """mark 阈值: stime < start_ymd 000000 为预热"""
        return self.start_ymd + "000000"

    def stream(self) -> Iterator[Bar]:
        import pandas as pd
        engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
        total = 0
        with engine.connect() as conn:
            for seg_start, seg_end in daterange(self.warmup_start, self.end_ymd, self.step_days):
                sql = (f"SELECT stock_code, stime, open, high, low, close, volume "
                       f"FROM {TABLE} WHERE stime >= '{seg_start}' AND stime <= '{seg_end}' "
                       f"AND stock_code = '{self.code}' "
                       f"ORDER BY stime ASC")
                # 批量取数 (pd.read_sql 一次性 fetchall, 比逐行 conn.execute 快数倍)
                df = pd.read_sql(sql, conn)
                seg_n = 0
                for rec in df.itertuples(index=False):
                    yield Bar(stime=rec.stime, code=rec.stock_code,
                              open=float(rec.open), high=float(rec.high),
                              low=float(rec.low), close=float(rec.close),
                              volume=int(rec.volume))
                    seg_n += 1
                    if self.delay:
                        time.sleep(self.delay)
                total += seg_n
                if self.verbose:
                    print(f"  -- 段 {seg_start[:8]}~{seg_end[:8]} 处理 {seg_n} 根, 累计 {total}",
                          flush=True)


class ChainedFeed(Feed):
    """串联多个行情源: 先历史预热, 再接实时 (实盘用)

    例: ChainedFeed(MySQLBacktestFeed(...今天), LiveFeed(code))
    """

    def __init__(self, *feeds: Feed):
        self.feeds = feeds

    def stream(self) -> Iterator[Bar]:
        for feed in self.feeds:
            yield from feed.stream()
