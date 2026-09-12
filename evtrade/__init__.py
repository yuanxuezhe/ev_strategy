from __future__ import annotations
"""evtrade — 策略回测 / 参数扫描 (CPU+GPU 统一向量化路径)

2026-09-13 重构: framework 不持有资金/持仓/撮合/PnL/收益概念;
execution/ 子包 + core/metrics + core/replay + core/permutation + core/config 已下线;
策略自管 cash/position/trades/equity 等业务概念。

包结构:
  顶层     __init__.py / __main__.py / cli.py / backends.py
  core/      aggregator / timeutils / engine / vectorized_engine / tsbucket
             / sweep / batched_sweep / data
  strategies/ vectorized_base + 已注册策略 (channel_deviation / ma_crossover /
              filtered_mr)
  indicators/ ema (step 增量版 + numpy 批量参考版 + torch 批量版)

用法:
  python -m evtrade backtest --device auto --strategy channel_deviation ...
  python -m evtrade sweep --grid low1=1.0,1.5 --strategy channel_deviation ...

依赖: pip install pymysql sqlalchemy numpy torch (torch CPU/CUDA 统一后端)
"""

# ---- 核心数据模型 / 时间桶 / 桶合并 ----
from .primitives import Bar, fmt
from .core.timeutils import (
    compute_bucket_general,
    resolve_period_seconds, encoded_to_epoch, epoch_to_encoded,
    bucket_ts_encoded, _days_from_civil,
)
from .core.aggregator import BarAggregator

# ---- 引擎 / 调度 ----
from .core.engine import Engine
from .core.sweep import sweep, parse_grid, GRID_KEYS, run_one_vectorized
from .core.data import DB_URL, TABLE

# ---- strategies ----
from .strategies import (
    VectorizedStrategy,
    get_strategy, available_strategies, register_strategy,
    get_strategy_class, get_strategy_param_spec,
)
from .strategies.channel_deviation import ChannelDeviationStrategy
from .strategies.ma_crossover import MACrossoverStrategy
from .strategies.filtered_mr import FilteredMRStrategy

# ---- backends / vectorized ----
from .backends import get_xp, gpu_available, resolve_device
from .core.vectorized_engine import run_vectorized

# ---- indicators (仅 EMA: step 增量版 + numpy 批量参考版 + torch 批量版) ----
from .indicators import (
    ema, ema_channel,
    EMAState, EMAChannelState,
    ema_step, ema_channel_step,
    torch_ema,
)

__all__ = [
    "Bar", "fmt",
    "compute_bucket_general",
    "resolve_period_seconds", "encoded_to_epoch", "epoch_to_encoded",
    "bucket_ts_encoded", "_days_from_civil",
    "BarAggregator",
    "VectorizedStrategy",
    "ChannelDeviationStrategy", "MACrossoverStrategy", "FilteredMRStrategy",
    "get_strategy", "available_strategies", "register_strategy",
    "get_strategy_class", "get_strategy_param_spec",
    "ema", "ema_channel",
    "EMAState", "EMAChannelState",
    "ema_step", "ema_channel_step",
    "torch_ema",
    "Engine",
    "run_vectorized", "run_one_vectorized",
    "get_xp", "gpu_available", "resolve_device",
    "sweep", "parse_grid", "GRID_KEYS",
    "DB_URL", "TABLE",
]