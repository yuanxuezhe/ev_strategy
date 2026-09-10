# kbs 知识库 · 通道偏离策略 + 统一 CPU/GPU 回测框架

本知识库是对项目源码的系统性中文分析文档，覆盖架构、数据结构、核心算法、策略逻辑、组件细节与运行指南。
**2026-09-09 起项目再次重构为统一 CPU/GPU 路径**（删 DSL 渲染层 / numba 流式内核 / NVRTC CUDA 编译，
策略唯一入口 = `VectorizedStrategy.step(state, bar, params)` + `init_state`；CPU/GPU 统一由 PyTorch 后端承担，
2026-09-10 pytorch-unified-strategy），详见 12、14、15 号文档。

> **2026-09-10 同步** (change `consolidate-simplify-core`):
> - 删 `evtrade/feeds/` 子包（数据加载内联 `core/data.py::load_bars`/`synthetic_bars`；bar 流 = 自定义 `.stream()` 对象，参考 `core/_harness.ListBarFeed`）；
> - 指标库只留 EMA（删 `atr/boll/rsi` 模块；`indicators/ema.py` = 增量版 `ema_step`/`ema_channel_step` + numpy 批量参考 `ema`/`ema_channel`）；
> - 删 `core/capability.py`（并入 `backends.resolve_device`）、`core/gpu.py`（更名 `core/tsbucket.py`，去 `gpu_info`）；
> - 删 `BrokerExecutor`（成交决策收口 `execution/base.py::trade_decision`，SimulatedExecutor + vectorized 共用）；
> - 删 CLI 死 flag `--engine`/`--no-sleep`/`--step-days`/`--show-bars`/`--bars-out` + 各指标单列 flag（改 `--params`）；共享选项进单一父 parser（argparse `parents=`）；
> - `config.py` 删 `PERIODS`/`INTERVAL`；`timeutils.py` 删 `compute_bucket`(bare)/`daterange`（留 `compute_bucket_general`/`resolve_period_seconds`）；`metrics.py` 删 `trades_to_list`、`engine.py` 删 `build_engine`/`print_summary`。
>
> **2026-09-09 同步**:
> - 删 `evtrade/strategies/{base,dsl,example_*.py}` 共 4 文件；策略基类合并为 `VectorizedStrategy` (`vectorized_base.py`)；
> - 删 `evtrade/core/{kernel,kernel_dsl}.py`、`core/gpu.cuda_sweep_window_generic`、`core/incremental_indicators.py`；
> - 删 `pyproject.toml` 的 `numba` 依赖；cupy 降为 optional extra（2026-09-10 进一步删 cupy，统一 torch）；
> - CLI 收口 `--device {cpu, gpu, auto}`（替代旧 `--engine {kernel, ref, vectorized}`）；
> - `kbs/14` 重写为"统一策略契约"，原"DSL 三端转译"章节整体废弃；
> - 引擎语义改为**桶 CLOSE**（桶切换时用上一桶 finalized OHLCV 驱动策略一次），与 vectorized 路径完全对齐。
> - `evtrade/indicators/` 为纯 Python 增量版 + numpy 批量参考版；无 numba，无 CUDA_DEVICE_*。

## 一句话简介

从 MySQL `minute_bars` 表读取 1 分钟行情（`core/data.py::load_bars`），增量合并为多周期 K 线（任意 m/h/d 周期），
用 EMA 通道轨（通达信蓝轨）的"极端偏离 → 回撤确认"逻辑产生 BUY/SELL 信号，
回测与实盘共用同一套 Engine，仅通过替换 bar 流（行情源）与 Executor（下单器）切换。

## 文档索引

| 文档 | 内容 | 适合谁读 |
|---|---|---|
| [01-项目总览.md](01-项目总览.md) | 项目定位、核心特性、技术栈、快速上手 | 所有人（先读这个） |
| [02-系统架构.md](02-系统架构.md) | 分层架构、数据流、回测/实盘切换原则（CPU/GPU 统一向量化） | 所有人 |
| [03-核心数据结构.md](03-核心数据结构.md) | Bar、周期桶 dict、`resolve_period_seconds` 周期秒数、stime 格式 | 开发者 |
| [04-周期合并机制.md](04-周期合并机制.md) | `compute_bucket_general` 桶算法、桶生命周期、warmup 机制 | 开发者（本项目最核心的机制之一） |
| [05-指标计算-EMA通道.md](05-指标计算-EMA通道.md) | EMA 定义、增量版 `ema_step`/`ema_channel_step` + numpy 批量参考 `ema`/`ema_channel` | 开发者 |
| [06-交易策略详解.md](06-交易策略详解.md) | 通道偏离回撤策略、锁存逻辑、单桶单操作、`step` 主逻辑 | 策略研究 / 开发者 |
| [07-账户与执行器.md](07-账户与执行器.md) | Account 记账、`trade_decision` 单一成交决策、Simulated Executor + 自定义实盘 Executor | 开发者 |
| [08-行情源Feed.md](08-行情源Feed.md) | 行情数据加载（`load_bars`/`synthetic_bars`）、bar 流契约、预热窗口 | 开发者 |
| [09-引擎Engine与主流程.md](09-引擎Engine与主流程.md) | Engine 装配、`on_bars` 桶 CLOSE 语义、`metrics.summarize` 盈亏汇总 | 开发者 |
| [10-配置参数与运行指南.md](10-配置参数与运行指南.md) | CLI 参数全表（含 `--device {cpu,gpu,auto}`）、典型命令、输出解读、FAQ | 使用者 |
| [11-扩展指南.md](11-扩展指南.md) | 新增 bar 流 / 执行器 / 策略 / 周期的做法（含 `VectorizedStrategy` 模板） | 二次开发者 |
| [12-重构与性能内核.md](12-重构与性能内核.md) | evtrade 包结构、vectorized 引擎、CPU/GPU 统一路径、旧 numba/CUDA/DSL 退场记录 | 所有人（先读 01 再读这个） |
| [13-绩效评估与鲁棒选参框架.md](13-绩效评估与鲁棒选参框架.md) | 超额曲线口径、metrics 30 字段（含 x_mdd）、滚动 WFO、邻域衰减 S、复合 score、帕累托、蒙特卡洛置换检验 | 选参/实盘前必读 |
| [14-策略DSL与三端转译.md](14-策略DSL与三端转译.md) | 旧 DSL→三端转译（numba/CUDA/`state_spec` 三端投影）**已下线**（2026-09）；现为单一 `step` 契约 + `@dataclass` state | 想了解历史 / 归档 |
| [15-PyTorch统一策略.md](15-PyTorch统一策略.md) | PyTorch 后端 (pytorch-unified-strategy, 2026-09-10)：`backends.get_xp` 统一 CPU/GPU、策略代码一份双端跑、`ema_step` 增量 + 逐桶 step 批量 | 写新策略 + 调参 + 性能优化 |
| [使用说明.md](使用说明.md) | **所有参数意思 + 完整命令行 + 网格扫参** + 输出解读 | 操作手册, 跑前/看结果前查这个 |

## 建议阅读路径

- **新人上手**：01 → 02 → 10（跑起来）→ 06（理解策略）
- **修改/排查合并逻辑**：03 → 04 → 09
- **改策略或调参**：05 → 06 → 10 → 14（让新策略 CPU/GPU 双端可用）
- **接实盘**：02 → 07 → 08 → 11

## 源码地图

现行代码已拆为 `evtrade/` 包，组件与文档的对应关系：

| 模块 | 内容 | 相关文档 |
|---|---|---|
| `evtrade/primitives.py` | `Bar`、`fmt` | 03 |
| `evtrade/core/config.py` | DB_URL（`EVTRADE_DB_URL` 可覆盖）/ 默认资金（`INIT_CASH`/`INIT_POSITION`/`TRADE_QTY`） | 01、10 |
| `evtrade/core/timeutils.py` | `compute_bucket_general`、`bucket_ts_encoded`、`resolve_period_seconds`、历法（标量/向量） | 04 |
| `evtrade/core/aggregator.py` | `BarAggregator`（INT period_seconds） | 04 |
| `evtrade/indicators/ema.py` | 增量版 `ema_step`/`ema_channel_step` + numpy 批量参考 `ema`/`ema_channel`（无 numba / 无 CUDA） | 05 |
| `evtrade/strategies/vectorized_base.py` | ★ **唯一策略基类** `VectorizedStrategy` + 注册表 + `params_spec` + `format_signal_line` hook | 06、11、14 |
| `evtrade/strategies/channel_deviation.py` | `ChannelDeviationStrategy`（`ema_channel_step` 增量 + Python FSM） | 06 |
| `evtrade/strategies/ma_crossover.py` | `MACrossoverStrategy`（EMA 双均线示例） | 14 |
| `evtrade/execution/account.py`、`evtrade/execution/base.py` | `Account`、`Executor`/`SimulatedExecutor` + `trade_decision` 单一成交决策 | 07 |
| `evtrade/core/data.py` | MySQL 拉取（`load_bars`）+ npz 缓存 + 合成数据（`synthetic_bars`） | 08、12 |
| `evtrade/core/engine.py` | ★ `Engine`（逐 bar 路径，桶 CLOSE 语义，供实盘/对账） | 09 |
| `evtrade/core/vectorized_engine.py` | ★ `run_vectorized`（批量向量化路径，CPU/GPU 统一） | 12、13 |
| `evtrade/core/metrics.py` | `bars_to_arrays` / `summarize`（30 字段，含 x_mdd / cagr_excess） | 12、13 |
| `evtrade/core/sweep.py` / `evtrade/cli.py` / `evtrade/core/tsbucket.py` | 扫描 / CLI / 桶预计算缓存（`precompute_ts_mark` + LRU）；`backends.py` 统一 CPU/GPU 后端 | 10、12、13 |
| `evtrade/core/replay.py` | 行情回放 + 对账（vectorized vs Engine.on_bars） | 10、13 |
| `tests/` | 26 个测试文件（CPU/GPU 容差 + vectorized-vs-Engine 对账 + metrics 字段集 + reconcile） | 12 |

> 现行代码已无 `mysql_analyze_demo.py` 兼容入口，统一用 `python -m evtrade`（子命令 `backtest`/`sweep`/`replay`/`params`）。
> 旧单文件实现 / numba 流式内核 / DSL 渲染管线均已下线（详见 12 号文档 §1）。

## 术语表

| 术语 | 含义 |
|---|---|
| bar / K线 | 一根行情记录（O/H/L/C + 成交量） |
| 1m bar | 数据库 `minute_bars` 中的原始 1 分钟行情 |
| 周期桶（bucket） | 若干根 1m bar 合并成的一根目标周期 K 线（如 5m） |
| 桶 ts | 桶的时间戳；本系统采用**前开后闭区间，标注右端点**（即桶内最后一根 1m bar 的时刻所在周期边界） |
| 闭合桶 | 已收完的桶（后续 1m bar 属于新桶了） |
| 当前桶（cur） | 尚未收完、正在被最新 1m bar 更新的桶 |
| UP / DW | 通道上轨 = EMA(H, 21)，下轨 = EMA(L, 21)（通达信蓝色通道轨） |
| 偏离（dev） | 价格相对通道轨的百分比距离，共 4 种定义，见 06 文档 |
| 锁存（latch / hysteresis） | "极端偏离"标记一旦置位保持不变，直到"回撤确认"触发信号并解除 |
| mark | 桶的标志位：0=预热（只累积指标），1=策略期（驱动策略） |
| warmup | 策略起始日之前的预热数据，用于让 EMA 有足够历史 |
| 不操作基线 | 期初资金 + 期初持仓 × 期末价；与策略期末总资产对比得盈亏 |
| `step(state, bar, params) -> (state, sig)` | 策略唯一抽象入口；`bar` = 单桶 `{ts,o,h,l,c,v,mark}`；返回 `(new_state, sig)`，`sig ∈ {-1,0,1}` |
| `init_state(params) -> state` | state 初值（无状态策略默认 `None`；stateful 策略覆写返 `@dataclass`）；state 由 engine 持有跨调用 |
| `state` | 持久状态（推荐 `@dataclass`）；策略**无 instance attr**，由引擎在 `step` 调用间持有 |
| `xp` | `backends.get_xp(device) -> torch.device`（`"cpu"→cpu`、`"gpu"→cuda`）；策略代码不直接 import 后端 |
| `VectorizedStrategy` | 唯一策略基类，定义 `step` 抽象方法 + `params_spec` schema 校验 + `format_signal_line` 默认 hook |
| `format_signal_line` | 策略展示 hook（Engine.on_bars 在 verbose 时打印）；默认仅打 OHLCV + 信号；channel_deviation 自定义展示 up/dw/dev |
| `reconcile` | vectorized vs Engine.on_bars 桶级信号 + 逐笔成交 bitwise 对账（`evtrade.core.replay.reconcile`） |
| `bucket_diff_cap` | `reconcile` 的差异容忍上限；默认 8（lenient），`EVT_RECONCILE_STRICT=1` 时 cap=0（strict） |

## 快速运行

```bash
pip install pymysql sqlalchemy pandas numpy torch  # 核心依赖 (torch 统一 CPU/GPU 后端; 无 numba)

# 单次回测 (CPU)
python -m evtrade backtest --strategy channel_deviation --device cpu \
    --period 5m --start 20250101 --end 20260903

# 单次回测 (GPU; 无 CUDA 时 --device gpu 会抛 ValueError, 用 --device auto 自动探测)
python -m evtrade backtest --strategy channel_deviation --device auto \
    --period 5m --start 20250101 --end 20260903

# 参数扫描: 滚动 WFO 多窗 + 费率 + 邻域衰减评分 + 蒙特卡洛 (选参标准流程, 见 13)
python -m evtrade sweep --strategy channel_deviation --device auto \
    --data-cache cache \
    --grid "low1=1.0,1.5,2.0" --grid "high2=0.3,0.5,0.8" \
    --splits 20260101,20260401,20260701 --fee-bp 5 --mc 300

# 实盘一致性验收: vectorized vs Engine.on_bars 逐桶 + 逐笔对账 (见 10/13)
python -m evtrade replay --log live_demo.log --strategy channel_deviation \
    --device cpu --against-ref   # 输出 PASS/FAIL + 分歧统计
```

前提：能访问 MySQL `192.168.10.2:33066` 的 `evtrade` 库（详见 08 文档，注意凭据为敏感信息，
可用环境变量 `EVTRADE_DB_URL` 覆盖）。更多用法见 10、12、13 号文档。
