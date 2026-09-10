"""evtrade — 策略回测 / 参数扫描 / 行情回放 (CPU+GPU 统一向量化路径)

包结构:
  顶层     __init__.py / __main__.py / cli.py / backends.py
  core/      aggregator / timeutils / engine / vectorized_engine / gpu
             / sweep / replay / permutation / metrics / data / config / capability
  execution/ account / base (Executor / SimulatedExecutor / BrokerExecutor)
  feeds/     base / mysql_history / chained / _registry
  strategies/ vectorized_base + 已注册策略 (channel_deviation / ma_crossover)
  indicators/ ema / atr / rsi / boll (xp 算子 + 纯 Python 增量版)

用法:
  python -m evtrade backtest --device auto --strategy channel_deviation ...
  python -m evtrade sweep --grid low1=1.0,1.5 --strategy channel_deviation ...
  python -m evtrade replay --log live_demo.log --against-ref
  python -m pytest tests -q

切换实盘/回测: 换 Feed + 换 Executor, 其余不变。
依赖: pip install pymysql sqlalchemy numpy (cupy 可选 GPU)
"""

# ---- 核心数据模型 / 时间桶 / 桶合并 / 记账 ----
from .primitives import Bar, fmt
from .core.timeutils import (
    compute_bucket, compute_bucket_general, daterange,
    resolve_period_seconds, encoded_to_epoch, epoch_to_encoded,
    bucket_ts_encoded, _days_from_civil,
)
from .core.aggregator import BarAggregator
from .execution.account import Account

# ---- 兼容旧 import 路径 (evtrade.<module>) ----
from .core import (
    data as _data_mod, config as _config_mod, timeutils as _timeutils_mod,
    aggregator as _aggregator_mod, engine as _engine_mod, sweep as _sweep_mod,
    replay as _replay_mod, permutation as _permutation_mod, gpu as _gpu_mod,
    vectorized_engine as _vec_mod, metrics as _metrics_mod,
)
from . import primitives as _primitives_mod
from .execution import account as _account_mod, base as _execution_mod

import sys as _sys
for _name, _mod in [
    ("evtrade.data", _data_mod), ("evtrade.config", _config_mod),
    ("evtrade.engine", _engine_mod), ("evtrade.sweep", _sweep_mod),
    ("evtrade.replay", _replay_mod), ("evtrade.permutation", _permutation_mod),
    ("evtrade.gpu", _gpu_mod), ("evtrade.aggregator", _aggregator_mod),
    ("evtrade.timeutils", _timeutils_mod),
    ("evtrade.primitives", _primitives_mod),
    ("evtrade.account", _account_mod), ("evtrade.execution", _execution_mod),
    ("evtrade.vectorized_engine", _vec_mod), ("evtrade.metrics", _metrics_mod),
]:
    _sys.modules.setdefault(_name, _mod)

# ---- 兼容旧 evtrade.kernel (转发到 core.timeutils / core.metrics) ----
import types as _types
_kernel_stub = _types.ModuleType("evtrade.kernel")
_kernel_stub.bucket_ts_encoded = _timeutils_mod.bucket_ts_encoded
_kernel_stub.encoded_to_epoch = _timeutils_mod.encoded_to_epoch
_kernel_stub.epoch_to_encoded = _timeutils_mod.epoch_to_encoded
_kernel_stub._days_from_civil = _timeutils_mod._days_from_civil
_kernel_stub.resolve_period_seconds = _timeutils_mod.resolve_period_seconds
_kernel_stub.summarize = _metrics_mod.summarize
_kernel_stub.trades_to_list = _metrics_mod.trades_to_list
_kernel_stub.bars_to_arrays = _metrics_mod.bars_to_arrays
_sys.modules["evtrade.kernel"] = _kernel_stub

# ---- 引擎 / 调度 ----
from .core.engine import Engine, build_engine
from .core.gpu import gpu_info
from .core.sweep import sweep, parse_grid, GRID_KEYS, run_one_vectorized
from .core.replay import (
    replay_vectorized, replay_engine, reconcile,
    append_bar, read_bars_log, write_bars_log,
)
from .core.permutation import permutation_test
from .core.metrics import (
    bars_to_arrays, summarize, trades_to_list,
)
from .core.config import DB_URL, TABLE, TF1, INIT_CASH, INIT_POSITION, TRADE_QTY

# ---- execution ----
from .execution.base import Executor, SimulatedExecutor

# ---- feeds ----
from .feeds import (
    ChainedFeed, Feed, MySQLBacktestFeed,
    get_feed, available_feeds, register_feed,
)

# ---- strategies ----
from .strategies import (
    VectorizedStrategy,
    get_strategy, available_strategies, register_strategy,
    get_strategy_class, get_strategy_param_spec,
)
from .strategies.channel_deviation import ChannelDeviationStrategy
from .strategies.ma_crossover import MACrossoverStrategy

# ---- backends / vectorized ----
from .backends import get_xp, gpu_available
from .core.vectorized_engine import run_vectorized

# ---- indicators ----
from .indicators import (
    ema, ema_channel, atr, rsi, bollinger, sma, true_range,
    xp_ema, xp_ema_channel,
    xp_ema_torch, xp_ema_channel_torch,
    EMAState, EMAChannelState, ATRState, RSIState, SMAState, BollState,
    ema_step, ema_channel_step, atr_step, rsi_step, sma_step, boll_step,
)


__all__ = [
    "Bar", "fmt",
    "compute_bucket", "compute_bucket_general", "daterange",
    "resolve_period_seconds", "encoded_to_epoch", "epoch_to_encoded",
    "bucket_ts_encoded", "_days_from_civil",
    "BarAggregator", "Account",
    "VectorizedStrategy",
    "ChannelDeviationStrategy", "MACrossoverStrategy",
    "get_strategy", "available_strategies", "register_strategy",
    "get_strategy_class", "get_strategy_param_spec",
    "ema", "ema_channel", "atr", "rsi", "bollinger", "sma", "true_range",
    "xp_ema", "xp_ema_channel",
    "xp_ema_torch", "xp_ema_channel_torch",
    "EMAState", "EMAChannelState",
    "ATRState", "RSIState", "SMAState", "BollState",
    "ema_step", "ema_channel_step",
    "atr_step", "rsi_step", "sma_step", "boll_step",
    "Executor", "SimulatedExecutor",
    "Feed", "MySQLBacktestFeed", "ChainedFeed",
    "get_feed", "available_feeds", "register_feed",
    "Engine", "build_engine",
    "run_vectorized", "run_one_vectorized", "get_xp", "gpu_available",
    "bars_to_arrays", "summarize", "trades_to_list",
    "gpu_info",
    "sweep", "parse_grid", "GRID_KEYS",
    "replay_vectorized", "replay_engine", "reconcile",
    "append_bar", "read_bars_log", "write_bars_log",
    "permutation_test",
    "DB_URL", "TABLE", "TF1", "INIT_CASH", "INIT_POSITION", "TRADE_QTY",
]
