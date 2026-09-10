"""MySQLBacktestFeed: 历史回测数据源

MySQL 分段查询, 含预热窗口, verbose 打印进度。
参数与 evtrade.data._fetch 等价, 但用 yield 而非一次 fetchall,
适合实盘节奏 (--no-sleep 关闭 delay 即可全速)。
"""
import time
from datetime import datetime, timedelta

from sqlalchemy import create_engine

from ..core.config import DB_URL, TABLE
from ..primitives import Bar
from ..core.timeutils import daterange
from ._registry import register_feed
from .base import Feed


@register_feed("mysql_history")
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
        self.warmup_start = (datetime.strptime(start_ymd, "%Y%m%d")
                             - timedelta(days=warmup_days)).strftime("%Y%m%d")

    @property
    def warmup_until(self) -> str:
        return self.start_ymd + "000000"

    def stream(self):
        import pandas as pd
        engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
        total = 0
        with engine.connect() as conn:
            for seg_start, seg_end in daterange(self.warmup_start, self.end_ymd, self.step_days):
                sql = (f"SELECT stock_code, stime, open, high, low, close, volume "
                       f"FROM {TABLE} WHERE stime >= '{seg_start}' AND stime <= '{seg_end}' "
                       f"AND stock_code = '{self.code}' "
                       f"ORDER BY stime ASC")
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