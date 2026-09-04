from __future__ import annotations
"""
minute_bars 周期合并 + 通道偏离策略 (回测/实盘统一架构)

分层:
  Feed (行情源)     ──yield Bar──▶ Engine
                                   │
                              BarAggregator (周期合并, 独立于策略)
                                   │ on_bars(周期K线集合)
                                   ▼
                         ChannelDeviationStrategy + ema_channel(指标)
                                   │ signal
                                   ▼
                              Executor (模拟/真实下单) → Account (资金持仓)

切换实盘/回测: 换 Feed + 换 Executor, 其余不变。
用法: python mysql_analyze_demo.py --period 5m --start 20250101 --end 20260903 --no-sleep
依赖: pip install pymysql sqlalchemy
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator
from sqlalchemy import create_engine, text

DB_URL = "mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4"
TABLE = "minute_bars"
INTERVAL = 0.1  # 100ms

# 周期: (单位, 数值, stime 起始位 0-indexed, timedelta)
# stime = YYYYMMDDHHmmss: 0-3年 4-5月 6-7日 8-9时 10-11分 12-13秒
PERIODS = {
    "1m":  ("m", 1,  10, timedelta(minutes=1)),
    "5m":  ("m", 5,  10, timedelta(minutes=5)),
    "15m": ("m", 15, 10, timedelta(minutes=15)),
    "30m": ("m", 30, 10, timedelta(minutes=30)),
    "1h":  ("h", 1,  8,  timedelta(hours=1)),
    "4h":  ("h", 4,  8,  timedelta(hours=4)),
    "1d":  ("d", 1,  6,  timedelta(days=1)),
}

TF1 = 21  # 通道轨 EMA 周期


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


from functools import lru_cache


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


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "----"


# ============ 聚合器 (独立于指标和策略) ============

class BarAggregator:
    """增量周期合并 (独立于指标)

    每根 bar -> update(); 更新当前桶, 桶切换时闭合旧桶;
    每次更新后以周期K线集合调用 on_bars(merged_bars), 集合末位为当前桶最新合并bar。

    warmup_until: stime 字符串阈值; stime < 该值标记 mark=0 (预热, 仅聚合+指标累积),
                  stime >= 该值标记 mark=1 (驱动策略)。同一桶内 mark 取最新一根的值。
    """

    def __init__(self, period_cfg, on_bars, warmup_until=None):
        self.value, self.pos, self.delta = period_cfg[1], period_cfg[2], period_cfg[3]
        self.on_bars = on_bars
        self.warmup_until = warmup_until
        self.bars: list[dict] = []          # 已闭合桶
        self.cur: dict | None = None        # 当前未闭合桶

    def update(self, bar: Bar):
        ts = compute_bucket(bar.stime, self.value, self.pos, self.delta)
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


# ============ 指标 (纯函数) ============

def ema(values: list[float], p: int):
    """EMA: 前 p 个 SMA 做 seed, 之后 EMA = price*k + EMA_prev*(1-k), k=2/(p+1)

    保留为纯函数 (用于一次性计算/校验)。热路径用 IncrementalEMA 增量化。
    """
    if len(values) < p:
        return None
    k = 2 / (p + 1)
    e = sum(values[:p]) / p          # SMA seed
    for v in values[p:]:
        e = v * k + e * (1 - k)
    return e


class IncrementalEMA:
    """增量 EMA 状态 (O(1)/桶, 替代每根 O(n) 重算)

    维护已闭合桶序列的 EMA:
      - 前 p-1 个桶: 累加 sum, 未出 EMA
      - 第 p 个桶: seed = sum/p, 出首个 EMA
      - 之后每个闭合桶: EMA = price*k + EMA_prev*(1-k), k=2/(p+1)
    当前未闭合桶: 基于已闭合 EMA 做一次临时递推 (不修改状态)。
    """

    def __init__(self, p: int):
        self.p = p
        self.k = 2 / (p + 1)
        self.sum = 0.0          # 前 p 个值的累加 (seed 用)
        self.count = 0          # 已闭合桶数
        self.ema = None         # 已闭合序列的最后一个 EMA (count>=p 时有值)

    def push(self, value: float):
        """闭合一个新桶, O(1) 更新 EMA 状态"""
        if self.count < self.p:
            self.sum += value
            self.count += 1
            if self.count == self.p:
                self.ema = self.sum / self.p   # SMA seed
        else:
            self.ema = value * self.k + self.ema * (1 - self.k)
            self.count += 1

    def current(self, pending_value: float):
        """带当前未闭合桶最新值的 EMA (不修改状态, O(1))

        pending_value: 当前桶最新 H 或 L。
        返回: 若已闭合 count>=p-1 则基于 pending 递推一次得到当前 EMA; 否则 None。
        """
        if self.count < self.p - 1:
            return None
        if self.count == self.p - 1:
            # 闭合了 p-1 个, 加当前 pending 凑齐 p 个 -> SMA seed
            return (self.sum + pending_value) / self.p
        # count >= p: 已有 ema, 用 pending 递推一次
        return pending_value * self.k + self.ema * (1 - self.k)


class EMAChannel:
    """通达信蓝色通道轨 (增量版)

    UP1 := EMA(H, TF1) 上轨, DW1 := EMA(L, TF1) 下轨
    桶闭合时 push(h, l); 每根 bar 调 channel(cur_h, cur_l) 得当前 (UP, DW), O(1)。
    """

    def __init__(self, tf1: int = 21):
        self.tf1 = tf1
        self.up_ema = IncrementalEMA(tf1)
        self.dw_ema = IncrementalEMA(tf1)

    def push(self, high: float, low: float):
        """桶闭合时调用"""
        self.up_ema.push(high)
        self.dw_ema.push(low)

    def channel(self, cur_high: float, cur_low: float):
        """返回当前 (UP, DW), 含未闭合桶最新值, O(1)。数据不足返回 (None, None)"""
        up = self.up_ema.current(cur_high)
        dw = self.dw_ema.current(cur_low)
        return up, dw


def ema_channel(bars: list[dict], tf1: int = 21):
    """通达信蓝色通道轨 (纯函数版, 向后兼容/校验用)

    UP1 := EMA(H, TF1)   上轨
    DW1 := EMA(L, TF1)   下轨
    入参: bars 周期K线集合 (末位=当前桶最新), tf1 EMA 周期
    出参: (UP, DW), 数据不足返回 (None, None)
    """
    if len(bars) < tf1:
        return None, None
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    return ema(highs, tf1), ema(lows, tf1)


# ============ 策略 (有状态, 独立于行情源和账户) ============

class ChannelDeviationStrategy:
    """通道偏离回撤策略 (有状态)

    偏离定义 (百分比):
      low_dev  = (DW - L) / DW * 100   # L 低于下轨的程度 (>0 为下偏), 用于 low1 极端偏离判定
      high_dev = (H - UP) / UP * 100   # H 高于上轨的程度 (>0 为上偏), 用于 high1 极端偏离判定
      low_dev_h  = (DW - H) / DW * 100 # H 相对下轨的回撤偏离, 用于 low2 BUY 触发
      high_dev_l = (L - UP) / UP * 100 # L 相对上轨的回撤偏离, 用于 high2 SELL 触发

    逻辑:
      当 low_dev  > low1%  -> 标记 low_hit  (下轨极端偏离, 用最低价 L)
      当 high_dev > high1% -> 标记 high_hit (上轨极端偏离, 用最高价 H)
      下一根 low_dev_h  < low2%  -> BUY  (最高价 H 回升到下轨上方, 整根bar回通道内)
      下一根 high_dev_l < high2% -> SELL (最低价 L 回落到上轨下方, 整根bar回通道内)

    信号方向:
      BUY  = 超跌反弹
      SELL = 超涨回落
    """

    def __init__(self, low1=1.5, low2=1, high1=1.5, high2=0.5):
        self.low1, self.low2 = low1, low2
        self.high1, self.high2 = high1, high2
        self.low_hit = False      # 下轨极端偏离锁存标记
        self.high_hit = False     # 上轨极端偏离锁存标记
        # 同一根周期桶内每个状态只允许一次状态转换 (置位 或 触发解除, 二选一)
        self._bucket_ts = None
        self._low_acted = False
        self._high_acted = False

    def check(self, cur: dict, up, dw):
        """返回 (signal, info) signal 为 'BUY'/'SELL'/None

        锁存逻辑 (hysteresis, 需 low1 > low2 / high1 > high2):
          下轨: low_dev > low1 -> low_hit=True (置位, 保持)
                low_hit 且 low_dev_h < low2 -> BUY 并 low_hit=False (H 回升过下轨, 解除)
          上轨: high_dev > high1 -> high_hit=True (置位, 保持)
                high_hit 且 high_dev_l < high2 -> SELL 并 high_hit=False (L 回落过上轨, 解除)
        即标记一旦置位, 不会因 low_dev 跌破 low1 而清除, 必须等到回撤 < low2 才触发并解除。

        单桶单操作: 同一根周期K线桶内, low_hit / high_hit 各自只允许一次状态转换
          (置位true 或 触发信号并置false), 二者互斥, 防止桶内 H/L 大幅波动反复触发。
          桶切换(ts变化)时重置本桶操作锁。
        """
        if up is None or dw is None or up == 0 or dw == 0:
            return None, {}
        # 桶切换: 重置本桶操作锁
        if cur["ts"] != self._bucket_ts:
            self._bucket_ts = cur["ts"]
            self._low_acted = False
            self._high_acted = False

        low_dev = (dw - cur["low"]) / dw * 100
        high_dev = (cur["high"] - up) / up * 100
        low_dev_h = (dw - cur["high"]) / dw * 100    # BUY 触发用: H vs DW
        high_dev_l = (cur["low"] - up) / up * 100    # SELL 触发用: L vs UP
        info = {"low_dev": low_dev, "high_dev": high_dev,
                "low_dev_h": low_dev_h, "high_dev_l": high_dev_l,
                "low_hit_prev": self.low_hit, "high_hit_prev": self.high_hit}

        signal = None
        # 信号触发 + 解除锁存: 基于上一根的 hit 标记, 当前根回撤入场 (本桶未操作过才触发)
        if self.low_hit and low_dev_h < self.low2 and not self._low_acted:
            signal = "BUY"
            self.low_hit = False
            self._low_acted = True
        elif self.high_hit and high_dev_l < self.high2 and not self._high_acted:
            signal = "SELL"
            self.high_hit = False
            self._high_acted = True

        # 进入极端偏离: 置位 (已置位则保持; 本桶已触发信号则不再置位)
        if low_dev > self.low1 and not self._low_acted:
            self.low_hit = True
            self._low_acted = True
        if high_dev > self.high1 and not self._high_acted:
            self.high_hit = True
            self._high_acted = True
        return signal, info


# ============ 账户 (资金/持仓, 独立于下单方式) ============

class Account:
    """资金与持仓管理 (回测/实盘共用)"""

    def __init__(self, cash: float, position: float):
        self.init_cash = cash
        self.init_position = position
        self.cash = cash
        self.position = position
        self.trades: list[dict] = []
        self.last_price = 0.0

    def apply(self, side: str, qty: float, price: float, ts: str):
        """记账 (不含下单逻辑, 由 Executor 调用)"""
        if side == "BUY":
            self.cash -= qty * price
            self.position += qty
        elif side == "SELL":
            self.cash += qty * price
            self.position -= qty
        self.trades.append({"ts": ts, "side": side, "qty": qty, "price": price})

    def equity(self, price: float = None) -> float:
        """总资产 = 资金 + 持仓市值"""
        p = price if price is not None else self.last_price
        return self.cash + self.position * p

    def baseline_equity(self, price: float = None) -> float:
        """不操作基线 = 期初资金 + 期初持仓 * 期末价"""
        p = price if price is not None else self.last_price
        return self.init_cash + self.init_position * p


# ============ 执行器 (下单抽象: 模拟 / 真实) ============

class Executor:
    """下单接口: 策略只调 trade(signal), 不关心是模拟还是真实委托"""

    def trade(self, signal: str, price: float, ts: str) -> bool:
        raise NotImplementedError


class SimulatedExecutor(Executor):
    """回测模拟下单: 以信号当根 close 成交, 资金/持仓不足则买满/卖完"""

    def __init__(self, account: Account, qty: float, verbose=True):
        self.account = account
        self.qty = qty
        self.verbose = verbose

    def trade(self, signal: str, price: float, ts: str) -> bool:
        acc = self.account
        if signal == "BUY":
            qty = min(self.qty, acc.cash / price) if price > 0 else 0
            if qty <= 0:
                return False
            acc.apply("BUY", qty, price, ts)
            if self.verbose:
                print(f"        >> BUY  {qty:.0f}股 @ {price:.4f}  花费 {qty*price:.2f}  "
                      f"剩余资金 {acc.cash:.2f} 持仓 {acc.position:.0f}", flush=True)
            return True
        elif signal == "SELL":
            qty = min(self.qty, acc.position)
            if qty <= 0:
                return False
            acc.apply("SELL", qty, price, ts)
            if self.verbose:
                print(f"        >> SELL {qty:.0f}股 @ {price:.4f}  收入 {qty*price:.2f}  "
                      f"剩余资金 {acc.cash:.2f} 持仓 {acc.position:.0f}", flush=True)
            return True
        return False


class BrokerExecutor(Executor):
    """实盘下单: 调券商 API (占位, 接入时实现)"""

    def __init__(self, account: Account, qty: float, broker=None):
        self.account = account
        self.qty = qty
        self.broker = broker  # 券商 API 客户端

    def trade(self, signal: str, price: float, ts: str) -> bool:
        # TODO: 接入真实券商下单 API
        # order_id = self.broker.place_order(signal, self.qty, ...)
        # 成交回报后调 self.account.apply(...) 记账
        print(f"        >> [LIVE] {signal} {self.qty:.0f}股 @ {price:.4f} (未接入券商)",
              flush=True)
        return False


# ============ 行情源 Feed (统一流入接口) ============

class Feed:
    """行情源抽象: stream() 按时间从旧到新 yield Bar"""

    def stream(self) -> Iterator[Bar]:
        raise NotImplementedError


def daterange(start_ymd: str, end_ymd: str, step_days: int = 7):
    """从旧到新按 step_days 天生成 [seg_start, seg_end] 闭区间 (YYYYMMDDHHmmss)

    每段含 step_days 天: [dayN 00:00:00, dayN+step-1 23:59:59]
    末段对齐到 end_ymd 的 23:59:59, 不超界。
    例: start=20250101, step=7 -> [20250101000000, 20250107235959]
        下一段           -> [20250108000000, 20250115235959]
    """
    fmt = "%Y%m%d"
    cur = datetime.strptime(start_ymd, fmt)
    last = datetime.strptime(end_ymd, fmt)
    while cur <= last:
        seg_end_day = min(cur + timedelta(days=step_days - 1), last)
        yield (cur.strftime("%Y%m%d") + "000000",
               seg_end_day.strftime("%Y%m%d") + "235959")
        cur = seg_end_day + timedelta(days=1)


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


# ============ 引擎 (连接 Feed → Aggregator → 策略 → 执行) ============

class Engine:
    """回测/实盘统一引擎

    run(): 从 feed 取 bar 喂 aggregator; aggregator 回调 on_bars;
           on_bars 内: 预热(mark=0)跳过, 否则算指标+策略, 信号触发 executor。
    """

    def __init__(self, feed: Feed, aggregator: BarAggregator,
                 strategy: ChannelDeviationStrategy, executor: Executor,
                 tf1: int = TF1, verbose=True):
        self.feed = feed
        self.aggregator = aggregator
        self.strategy = strategy
        self.executor = executor
        self.tf1 = tf1
        self.verbose = verbose
        # 增量 EMA 通道轨 (替代每根 O(n) 重算的 ema_channel)
        self.ema_ch = EMAChannel(tf1)
        self._pushed = 0        # 已 push 的闭合桶数 (对应 aggregator.bars 长度)
        # 让 aggregator 的回调指向自己
        self.aggregator.on_bars = self.on_bars

    def _sync_ema(self, closed_bars: list[dict]):
        """把新增的闭合桶 push 进 EMA 状态 (O(1)/桶)"""
        n = len(closed_bars)
        while self._pushed < n:
            b = closed_bars[self._pushed]
            self.ema_ch.push(b["high"], b["low"])
            self._pushed += 1

    def on_bars(self, bars: list[dict]):
        """聚合器回调: 计算指标 + 策略信号 + 执行 + 打印

        mark=0 (预热): 只 push 闭合桶进 EMA 状态 (指标预热), 不驱动策略。
        mark=1 (策略期): 增量算指标 + 驱动策略 + 打印。
        """
        closed = bars[:-1]           # 已闭合桶
        cur = bars[-1]               # 当前未闭合桶
        # 同步新增闭合桶到 EMA 状态 (预热期也需累积, 保证 mark=1 时指标就绪)
        self._sync_ema(closed)

        if cur.get("mark", 1) == 0:
            return   # 预热: 只累积指标历史, 不进入策略

        price = float(cur["close"])
        self.executor.account.last_price = price

        # 增量 EMA: 基于已闭合序列 + 当前桶最新 H/L, O(1)
        up, dw = self.ema_ch.channel(cur["high"], cur["low"])
        signal, info = self.strategy.check(cur, up, dw)

        # 信号触发 -> 统一下单处理
        if signal:
            self.executor.trade(signal, price, cur["ts"])

        if not self.verbose:
            return

        low_dev = info.get("low_dev")
        high_dev = info.get("high_dev")
        low_dev_h = info.get("low_dev_h")
        high_dev_l = info.get("high_dev_l")
        low_hit_prev = info.get("low_hit_prev")
        high_hit_prev = info.get("high_hit_prev")

        prefix = f"{signal} >>> " if signal else "             "
        line = (f"{prefix}[{cur['ts']}] {cur['code']} | O:{cur['open']} H:{cur['high']} "
                f"L:{cur['low']} C:{cur['close']} | vol:{cur['volume']} x{cur['count']} | "
                f"UP={fmt(up)} DW={fmt(dw)} | "
                f"low_dev(L/DW)={fmt(low_dev)}% high_dev(H/UP)={fmt(high_dev)}% | "
                f"low_dev_h(H/DW)={fmt(low_dev_h)}% high_dev_l(L/UP)={fmt(high_dev_l)}% | "
                f"low_hit_prev={fmt(low_hit_prev)} high_hit_prev={fmt(high_hit_prev)}")
        if signal:
            print(line, flush=True)

    def run(self):
        total = 0
        try:
            for bar in self.feed.stream():
                self.aggregator.update(bar)
                total += 1
            self.aggregator.flush()
        except KeyboardInterrupt:
            print("\n已停止")
        if self.verbose:
            print(f"\n完成, 共处理 {total} 根 1m bar")
        return total

    def print_summary(self):
        """回测结束: 策略总资产 vs 不操作基线, 算盈亏"""
        acc = self.executor.account
        price = acc.last_price
        final_total = acc.equity(price)
        baseline = acc.baseline_equity(price)
        diff = final_total - baseline
        pct = (diff / baseline * 100) if baseline else 0

        print("\n" + "=" * 60)
        print("回测盈亏汇总")
        print("=" * 60)
        print(f"期末价 (最后一根close) : {fmt(price)}")
        print(f"交易次数              : {len(acc.trades)}")
        print(f"期初资金 / 期初持仓    : {acc.init_cash:.0f} / {acc.init_position:.0f}股")
        print(f"期末资金 / 期末持仓    : {acc.cash:.2f} / {acc.position:.0f}股")
        print(f"期末持仓市值           : {acc.position * price:.2f}")
        print(f"策略总资产 (资金+市值) : {final_total:.2f}")
        print(f"不操作基线 (资金+市值) : {baseline:.2f}")
        print(f"盈亏差额 (策略-基线)   : {diff:+.2f}")
        print(f"盈亏比例              : {pct:+.2f}%")
        print("=" * 60)


# ============ 默认配置 ============

INIT_CASH = 200000.0       # 期初资金 20万
INIT_POSITION = 200000.0   # 期初持仓 20万股
TRADE_QTY = 10000.0        # 每次信号交易股数


# ============ 入口 ============

def main():
    ap = argparse.ArgumentParser(description="minute_bars 周期合并 + 通达信通道轨 (回测/实盘统一)")
    ap.add_argument("--period", default="5m", choices=list(PERIODS.keys()))
    ap.add_argument("--start", default="20250101", help="策略起始日期 YYYYMMDD")
    ap.add_argument("--end", default="20260903", help="策略结束日期 YYYYMMDD")
    ap.add_argument("--step-days", type=int, default=7, help="分段查询天数(闭区间)")
    ap.add_argument("--tf1", type=int, default=TF1, help="通道轨 EMA 周期")
    ap.add_argument("--no-sleep", action="store_true", help="去掉每根 bar 的 sleep, 全速回测")
    ap.add_argument("--trade-qty", type=float, default=TRADE_QTY, help="每次信号交易股数")
    ap.add_argument("--code", default="159992.SZ", help="证券代码 (如 159992.SZ / 513120.SH)")
    ap.add_argument("--low1", type=float, default=1.5, help="下轨极端偏离阈值(百分比)")
    ap.add_argument("--low2", type=float, default=1.0, help="下轨回撤触发阈值(百分比)")
    ap.add_argument("--high1", type=float, default=1.5, help="上轨极端偏离阈值(百分比)")
    ap.add_argument("--high2", type=float, default=0.5, help="上轨回撤触发阈值(百分比)")
    args = ap.parse_args()

    delay = 0 if args.no_sleep else INTERVAL

    # 1. 行情源 (回测: MySQL 历史分段, 含预热)
    feed = MySQLBacktestFeed(code=args.code, start_ymd=args.start, end_ymd=args.end,
                             step_days=args.step_days, delay=delay, verbose=True)

    # 2. 账户 + 执行器 (回测: 模拟下单)
    account = Account(cash=INIT_CASH, position=INIT_POSITION)
    executor = SimulatedExecutor(account, qty=args.trade_qty, verbose=True)

    # 3. 聚合器 (周期合并 + warmup 标记)
    aggregator = BarAggregator(PERIODS[args.period], on_bars=None,
                               warmup_until=feed.warmup_until)

    # 4. 策略
    strategy = ChannelDeviationStrategy(low1=args.low1, low2=args.low2,
                                        high1=args.high1, high2=args.high2)

    # 5. 引擎 (注入以上组件)
    engine = Engine(feed, aggregator, strategy, executor, tf1=args.tf1, verbose=True)

    print(f"证券: {args.code}  周期: {args.period}  策略日期: {args.start}~{args.end}  "
          f"预热起点: {feed.warmup_start}  分段: {args.step_days}天/段(闭区间)  "
          f"TF1={args.tf1}  sleep={'OFF' if args.no_sleep else 'ON'}  "
          f"low1/low2={args.low1}/{args.low2} high1/high2={args.high1}/{args.high2}  "
          f"Ctrl+C 停止\n")
    engine.run()
    engine.print_summary()


if __name__ == "__main__":
    main()
