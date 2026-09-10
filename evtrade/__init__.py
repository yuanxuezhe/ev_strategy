"""evtrade — 策略回测 / 参数扫描 / 行情回放 (CPU+GPU 统一向量化路径)

包结构:
  顶层     __init__.py / __main__.py / cli.py / backends.py
  core/      aggregator / timeutils / engine / vectorized_engine / tsbucket
             / sweep / replay / permutation / metrics / data / config
  execution/ account / base (Executor / SimulatedExecutor)
  strategies/ vectorized_base + 已注册策略 (channel_deviation / ma_crossover)
  indicators/ ema (step 增量版 + numpy 批量参考版)

用法:
  python -m evtrade backtest --device auto --strategy channel_deviation ...
  python -m evtrade sweep --grid low1=1.0,1.5 --strategy channel_deviation ...
  python -m evtrade replay --log live_demo.log --against-ref
  python -m pytest tests -q

切换实盘/回测: 换 bar 流 + 换 Executor, 其余不变。
依赖: pip install pymysql sqlalchemy numpy torch (torch CPU/CUDA 统一后端)
"""

# ---- 核心数据模型 / 时间桶 / 桶合并 / 记账 ----
from .primitives import Bar, fmt
from .core.timeutils import (
    compute_bucket_general,
    resolve_period_seconds, encoded_to_epoch, epoch_to_encoded,
    bucket_ts_encoded, _days_from_civil,
)
from .core.aggregator import BarAggregator
from .execution.account import Account

# ---- 兼容旧 import 路径 (evtrade.<module>) ----
from .core import (
    data as _data_mod, metrics as _metrics_mod,
    aggregator as _aggregator_mod, replay as _replay_mod,
)
from . import primitives as _primitives_mod
from .execution import account as _account_mod, base as _execution_mod

import sys as _sys
for _name, _mod in [
    ("evtrade.data", _data_mod), ("evtrade.metrics", _metrics_mod),
    ("evtrade.aggregator", _aggregator_mod), ("evtrade.replay", _replay_mod),
    ("evtrade.primitives", _primitives_mod),
    ("evtrade.account", _account_mod), ("evtrade.execution", _execution_mod),
]:
    _sys.modules.setdefault(_name, _mod)

# ---- 引擎 / 调度 ----
from .core.engine import Engine
from .core.sweep import sweep, parse_grid, GRID_KEYS, run_one_vectorized
from .core.replay import (
    replay_vectorized, replay_engine, reconcile,
    append_bar, read_bars_log, write_bars_log,
)
from .core.permutation import permutation_test
from .core.metrics import (
    bars_to_arrays, summarize,
)
from .core.config import DB_URL, TABLE, INIT_CASH, INIT_POSITION, TRADE_QTY

# ---- execution ----
from .execution.base import Executor, SimulatedExecutor

# ---- strategies ----
from .strategies import (
    VectorizedStrategy,
    get_strategy, available_strategies, register_strategy,
    get_strategy_class, get_strategy_param_spec,
)
from .strategies.channel_deviation import ChannelDeviationStrategy
from .strategies.ma_crossover import MACrossoverStrategy

# ---- backends / vectorized ----
from .backends import get_xp, gpu_available, resolve_device
from .core.vectorized_engine import run_vectorized

# ---- indicators (仅 EMA: step 增量版 + numpy 批量参考版) ----
from .indicators import (
    ema, ema_channel,
    EMAState, EMAChannelState,
    ema_step, ema_channel_step,
)

__all__ = [
    "Bar", "fmt",
    "compute_bucket_general",
    "resolve_period_seconds", "encoded_to_epoch", "epoch_to_encoded",
    "bucket_ts_encoded", "_days_from_civil",
    "BarAggregator", "Account",
    "VectorizedStrategy",
    "ChannelDeviationStrategy", "MACrossoverStrategy",
    "get_strategy", "available_strategies", "register_strategy",
    "get_strategy_class", "get_strategy_param_spec",
    "ema", "ema_channel",
    "EMAState", "EMAChannelState",
    "ema_step", "ema_channel_step",
    "Executor", "SimulatedExecutor",
    "Engine",
    "run_vectorized", "run_one_vectorized", "get_xp", "gpu_available", "resolve_device",
    "bars_to_arrays", "summarize",
    "sweep", "parse_grid", "GRID_KEYS",
    "replay_vectorized", "replay_engine", "reconcile",
    "append_bar", "read_bars_log", "write_bars_log",
    "permutation_test",
    "DB_URL", "TABLE", "INIT_CASH", "INIT_POSITION", "TRADE_QTY",
]
