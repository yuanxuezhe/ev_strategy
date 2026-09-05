from __future__ import annotations
"""
evtrade — minute_bars 周期合并 + 通道偏离策略 (回测/实盘统一架构)

由原单文件 mysql_analyze_demo.py 拆分而来 (纯移动, 逻辑不变):

  config.py       常量与默认配置 (DB_URL 支持环境变量 EVTRADE_DB_URL 覆盖)
  models.py       Bar 统一行情结构 / fmt
  timeutils.py    compute_bucket (周期桶时间戳) / daterange (分段区间)
  aggregator.py   BarAggregator 增量周期合并器
  indicators.py   EMA / IncrementalEMA / EMAChannel (通达信蓝色通道轨)
  strategy.py     ChannelDeviationStrategy 通道偏离回撤策略
  account.py      Account 资金/持仓记账
  execution.py    Executor / SimulatedExecutor / BrokerExecutor
  feeds.py        Feed / MySQLBacktestFeed / ChainedFeed
  engine.py       Engine 回测/实盘统一引擎 (以上为"参考实现", 也是差分测试基准)

新增:
  kernel.py       numba 流式决策内核 —— 与参考实现逐行等价 (差分测试锁定),
                  回测批量灌入 / 实盘逐根 step / 参数扫描 / GPU 移植共用同一套递推
  data.py         行情加载: MySQL 单次全量拉取 + 本地 npz 缓存 + 合成数据生成器
  sweep.py        参数并发扫描 (线程池 + nogil 内核) + 绩效汇总 + walk-forward 验证
  gpu.py          GPU 可用性探测与 CUDA 移植说明
  cli.py          命令行入口 (backtest / sweep)

用法:
  python mysql_analyze_demo.py --period 5m --start 20250101 --end 20260903 --no-sleep
  python -m evtrade backtest --engine kernel --period 5m ...
  python -m evtrade sweep --grid low1=1.0,1.5,2.0 --grid high2=0.3,0.5 ...

切换实盘/回测: 换 Feed + 换 Executor, 其余不变。
依赖: pip install pymysql sqlalchemy numpy numba
"""

from .models import Bar, fmt
from .timeutils import compute_bucket, compute_bucket_general, daterange, resolve_period_seconds
from .aggregator import BarAggregator
from .indicators import EMAChannel, IncrementalEMA, ema, ema_channel
from .strategy import ChannelDeviationStrategy
from .account import Account
from .execution import BrokerExecutor, Executor, SimulatedExecutor
from .feeds import ChainedFeed, Feed, MySQLBacktestFeed
from .engine import Engine

__all__ = [
    "Bar", "fmt",
    "compute_bucket", "daterange", "BarAggregator",
    "ema", "IncrementalEMA", "EMAChannel", "ema_channel",
    "ChannelDeviationStrategy", "Account",
    "Executor", "SimulatedExecutor", "BrokerExecutor",
    "Feed", "MySQLBacktestFeed", "ChainedFeed", "Engine",
]
