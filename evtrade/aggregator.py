from __future__ import annotations
"""增量周期合并器 (自 mysql_analyze_demo.py 原样迁移 + 任意周期支持)"""

from typing import Optional, Union

from .models import Bar
from .timeutils import compute_bucket_general


# ============ 聚合器 (独立于指标和策略) ============

class BarAggregator:
    """增量周期合并 (独立于指标)

    每根 bar -> update(); 更新当前桶, 桶切换时闭合旧桶;
    每次更新后以周期K线集合调用 on_bars(merged_bars), 集合末位为当前桶最新合并bar。

    period_cfg: 兼容两种写法
      * 传统元组 PERIODS[period] = (单位, 数值, stime起始位, timedelta) —— 取其 timedelta
      * 直接给周期秒数 int (如 resolve_period_seconds("90m"))
      统一走 compute_bucket_general (支持任意 m/h/d 周期; 对老 7 周期与原
      字段取整算法逐例等价, 见 tests/test_timeutils.py)。

    warmup_until: stime 字符串阈值; stime < 该值标记 mark=0 (预热, 仅聚合+指标累积),
                  stime >= 该值标记 mark=1 (驱动策略)。同一桶内 mark 取最新一根的值。
    """

    def __init__(self, period_cfg: Union[tuple, int], on_bars, warmup_until=None):
        if isinstance(period_cfg, (int, float)):
            self.period_seconds = int(period_cfg)
        else:
            self.period_seconds = int(period_cfg[3].total_seconds())
        self.on_bars = on_bars
        self.warmup_until = warmup_until
        self.bars: list[dict] = []          # 已闭合桶
        self.cur: Optional[dict] = None     # 当前未闭合桶

    def update(self, bar: Bar):
        ts = compute_bucket_general(bar.stime, self.period_seconds)
        mark = 0 if (self.warmup_until and bar.stime < self.warmup_until) else 1

        # 桶切换: 闭合旧桶 (快照入列, cur 重置)
        bucket_switched = self.cur is not None and ts != self.cur["ts"]
        if bucket_switched:
            self.bars.append(self.cur)
            self.cur = None

        # 新桶: 首根 bar 初始化
        if self.cur is None:
            self.cur = {"ts": ts, "code": bar.code,
                        "open": bar.open, "high": bar.high,
                        "low": bar.low, "close": bar.close,
                        "volume": bar.volume, "count": 1, "mark": mark}
        else:
            # 同桶: 用最新 bar 更新 (open 首根, high/low 极值, close 最新, vol 累加)
            self.cur["high"] = max(self.cur["high"], bar.high)
            self.cur["low"] = min(self.cur["low"], bar.low)
            self.cur["close"] = bar.close
            self.cur["volume"] += bar.volume
            self.cur["count"] += 1
            self.cur["mark"] = mark   # mark 跟随最新一根

        # 周期K线集合 = 已闭合 + 当前桶(末位=最新行情)
        # 用 append/pop 复用列表, 避免每根 O(n) 复制 self.bars + [self.cur]
        self.bars.append(self.cur)
        try:
            self.on_bars(self.bars)
        finally:
            self.bars.pop()

    def flush(self):
        """收尾闭合最后一个桶"""
        if self.cur is not None:
            self.bars.append(self.cur)
            self.cur = None
        self.on_bars(self.bars)
