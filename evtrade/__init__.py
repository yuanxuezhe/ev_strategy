from __future__ import annotations
"""evtrade — minute_bars 周期合并 + 通道偏离策略 (回测/实盘统一架构)

================================================================
📦 包结构 (2026-09-06 子包化, 顶层仅 2 个文件)
================================================================

顶层只放:
  __init__.py   本文件: 顶层 API 统一导出 (向后兼容)
  __main__.py   python -m evtrade 入口
  cli.py        CLI 入口 (回测 / 扫描 / 回放 三个子命令)

子包:
  frozen/       ⚠️ 冻结层 (kernel 锁定, 改了触发 72 项测试)
                - aggregator.py   BarAggregator 增量桶合并
                - timeutils.py    compute_bucket_general (任意周期)
                - models.py       Bar / fmt
                - account.py      Account 记账
                - strategy.py     ChannelDeviationStrategy (冻结版)
                - incremental_indicators.py  IncrementalEMA / EMAChannel

  core/         ✅ 主调度层
                - kernel.py       numba 流式决策内核 (冻结, 与 frozen 同源等价)
                - engine.py       参考引擎 (供 replay 对账)
                - gpu.py          CUDA 内核 (与 kernel 等价)
                - sweep.py        并发参数扫描 + 鲁棒评分
                - replay.py       录制回放对账
                - permutation.py  MC 置换检验
                - config.py       全局常量
                - data.py         MySQL 拉数 + npz 缓存 + 合成数据

  execution/    ✅ 下单撮合
                - base.py         Executor / SimulatedExecutor / BrokerExecutor

  feeds/        ✅ 行情入口
                - base.py         Feed 基类
                - mysql_history.py   MySQLBacktestFeed
                - chained.py      ChainedFeed
                - _registry.py    get_feed / register_feed

  strategies/   ✅ 策略目录
                - base.py         StrategyBase + register_strategy
                - channel_deviation.py  DSL 版 (实验)
                - example_breakout.py   示例

  indicators/   ✅ 指标目录 (纯函数库, jupyter 友好)
                - ema / atr / rsi / boll

================================================================
🟢🟡🔴 修改频次分类 (每个文件 docstring 顶部也标了)
================================================================
🟢 冻结 (kernel 锁定, 改了触发全红):
    frozen/aggregator / timeutils / models / account / strategy /
    incremental_indicators
    core/kernel / engine / gpu / replay

🟡 可改 (用户面 / 参数面):
    cli.py  core/config / data / sweep  execution/base
    feeds/*  strategies/*  indicators/*

🟠 慎改 (公式锁定, kbs/13 文档约束):
    core/sweep (score 公式)  core/permutation (日块置换)

================================================================
📜 用法
================================================================
  python -m evtrade backtest --engine kernel --period 5m ...
  python -m evtrade sweep --grid low1=1.0,1.5,2.0 --grid high2=0.3,0.5,0.8 ...
  python -m evtrade replay --log live_demo.log --against-ref
  python -m pytest tests -q

切换实盘/回测: 换 Feed + 换 Executor, 其余不变。
依赖: pip install pymysql sqlalchemy numpy numba
"""

# ============================================================
# 顶层 API 统一导出 (向后兼容, 用户代码不变)
# ============================================================
import sys as _sys

# ---- frozen 冻结层 ----
from .primitives import Bar, fmt
from .core.timeutils import compute_bucket, compute_bucket_general, daterange, resolve_period_seconds
from .core.aggregator import BarAggregator
from .frozen.account import Account
from .strategies.channel_deviation import ChannelDeviationStrategy
from .frozen.incremental_indicators import EMAChannel, IncrementalEMA, ema, ema_channel

# ---- 向后兼容 shim: 让 `from evtrade.data import ...` / `from evtrade.config import ...` 继续可用 ----
# 测试文件 (tests/test_*.py) 用顶层路径, 顶层 `__init__.py` 把它们映射到真实位置
from .core import data as _data_mod
from .core import config as _config_mod
from .core import kernel as _kernel_mod
from .core import engine as _engine_mod
from .core import sweep as _sweep_mod
from .core import replay as _replay_mod
from .core import permutation as _permutation_mod
from .core import gpu as _gpu_mod
from . import primitives as _primitives_mod
from .core import timeutils as _timeutils_mod, aggregator as _aggregator_mod
from .frozen import (
    account as _account_mod,
    incremental_indicators as _incr_indicators_mod,
)
from .strategies import channel_deviation as _strategy_mod  # 旧 evtrade.strategy shim (向后兼容)
from .execution import base as _execution_mod
_sys.modules.setdefault("evtrade.data", _data_mod)
_sys.modules.setdefault("evtrade.config", _config_mod)
_sys.modules.setdefault("evtrade.kernel", _kernel_mod)
_sys.modules.setdefault("evtrade.engine", _engine_mod)
_sys.modules.setdefault("evtrade.sweep", _sweep_mod)
_sys.modules.setdefault("evtrade.replay", _replay_mod)
_sys.modules.setdefault("evtrade.permutation", _permutation_mod)
_sys.modules.setdefault("evtrade.gpu", _gpu_mod)
_sys.modules.setdefault("evtrade.aggregator", _aggregator_mod)
_sys.modules.setdefault("evtrade.timeutils", _timeutils_mod)
_sys.modules.setdefault("evtrade.primitives", _primitives_mod)
_sys.modules.setdefault("evtrade.models", _primitives_mod)  # 向后兼容旧路径
_sys.modules.setdefault("evtrade.account", _account_mod)
_sys.modules.setdefault("evtrade.strategy", _strategy_mod)
_sys.modules.setdefault("evtrade.execution", _execution_mod)
_sys.modules.setdefault("evtrade._incremental_indicators", _incr_indicators_mod)

# ---- core 主调度 ----
from .core.kernel import (
    KernelState, make_state, run_backtest, run_backtest_trace, step,
    bucket_table, summarize, trades_to_list, bars_to_arrays,
    bucket_ts_encoded, encoded_to_epoch, epoch_to_encoded, _days_from_civil,
)
from .core.kernel_dsl import (
    build_dsl_kernel, dsl_kernel, make_state_general, run_one_dsl,
    strategy_has_dsl,
)
from .core.engine import Engine
from .core.gpu import gpu_info, cuda_sweep_window, cuda_sweep_window_generic
from .core.sweep import sweep, parse_grid, GRID_KEYS
from .core.replay import replay_kernel, replay_engine, reconcile, append_bar, read_bars_log, write_bars_log
from .core.permutation import permutation_test
from .core.config import DB_URL, TABLE, INTERVAL, TF1, INIT_CASH, INIT_POSITION, TRADE_QTY

# ---- execution ----
from .execution.base import BrokerExecutor, Executor, SimulatedExecutor

# ---- feeds / strategies / indicators 子包 ----
from .feeds import (
    ChainedFeed, Feed, MySQLBacktestFeed,
    get_feed, available_feeds, register_feed,
)
from .strategies import (
    StrategyBase,
    get_strategy, available_strategies, register_strategy,
    render_numba_state_body, render_cuda_device_function, DSLCtx, dsl_check,
)
from .indicators import (
    ema as ema_fn, atr, rsi, bollinger, sma, true_range,
)


__all__ = [
    # 数据模型
    "Bar", "fmt",
    # 时间桶
    "compute_bucket", "compute_bucket_general", "daterange", "resolve_period_seconds",
    # 桶合并 + 记账
    "BarAggregator", "Account",
    # 策略
    "ChannelDeviationStrategy",
    # 指标 (增量 / 冻结层)
    "ema", "IncrementalEMA", "EMAChannel", "ema_channel",
    # 指标 (纯函数 / 子包)
    "ema_fn", "atr", "rsi", "bollinger", "sma", "true_range",
    # 执行器
    "Executor", "SimulatedExecutor", "BrokerExecutor",
    # 行情
    "Feed", "MySQLBacktestFeed", "ChainedFeed",
    "get_feed", "available_feeds", "register_feed",
    # 策略目录
    "StrategyBase", "get_strategy", "available_strategies", "register_strategy",
    "render_numba_state_body", "render_cuda_device_function", "DSLCtx", "dsl_check",
    # 引擎 / 内核
    "Engine",
    "KernelState", "make_state", "run_backtest", "run_backtest_trace", "step",
    "bucket_table", "summarize", "trades_to_list", "bars_to_arrays",
    "bucket_ts_encoded", "encoded_to_epoch", "epoch_to_encoded", "_days_from_civil",
    # DSL 特化内核 (任意 DSL 策略的 numba/CUDA 路径)
    "build_dsl_kernel", "dsl_kernel", "make_state_general", "run_one_dsl",
    "strategy_has_dsl",
    # GPU
    "gpu_info", "cuda_sweep_window", "cuda_sweep_window_generic",
    # 扫描 / 评分
    "sweep", "parse_grid", "GRID_KEYS",
    # 回放 / 置换
    "replay_kernel", "replay_engine", "reconcile",
    "append_bar", "read_bars_log", "write_bars_log",
    "permutation_test",
    # 配置
    "DB_URL", "TABLE", "INTERVAL", "TF1", "INIT_CASH", "INIT_POSITION", "TRADE_QTY",
]
