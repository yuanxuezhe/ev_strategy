# kbs 知识库 · 通道偏离策略 + 统一 CPU/GPU 回测框架

本知识库是对项目源码的系统性中文分析文档，覆盖架构、数据结构、核心算法、策略逻辑、组件细节与运行指南。
**2026-09-09 起项目再次重构为统一 CPU/GPU 路径**（删 DSL 渲染层 / numba 流式内核 / NVRTC CUDA 编译，
策略唯一入口 = `VectorizedStrategy.compute_signals(xp, bars, params)`；性能内核由 numpy + cupy 承担），
详见 12、14 号文档。

> **2026-09-09 同步**:
> - 删 `evtrade/strategies/{base,dsl,example_*.py}` 共 4 文件；策略基类合并为 `VectorizedStrategy` (`vectorized_base.py`)；
> - 删 `evtrade/core/{kernel,kernel_dsl}.py`、`core/gpu.cuda_sweep_window_generic`、`core/incremental_indicators.py`；
> - 删 `pyproject.toml` 的 `numba` 依赖；cupy 降为 optional extra；
> - CLI 收口 `--device {cpu, gpu, auto}`（替代旧 `--engine {kernel, ref, vectorized}`）；
> - `kbs/14` 重写为"统一策略契约"，原"DSL 三端转译"章节整体废弃；
> - 引擎语义改为**桶 CLOSE**（桶切换时用上一桶 finalized OHLCV 驱动策略一次），与 vectorized 路径完全对齐。
> - `evtrade/indicators/` 改写为纯 Python 增量版 + xp 算子批量版；无 numba，无 CUDA_DEVICE_*。

## 一句话简介

从 MySQL `minute_bars` 表读取 1 分钟行情，增量合并为多周期 K 线（任意 m/h/d 周期），
用 EMA 通道轨（通达信蓝轨）的"极端偏离 → 回撤确认"逻辑产生 BUY/SELL 信号，
回测与实盘共用同一套 Engine，仅通过替换 Feed（行情源）与 Executor（下单器）切换。

## 文档索引

| 文档 | 内容 | 适合谁读 |
|---|---|---|
| [01-项目总览.md](01-项目总览.md) | 项目定位、核心特性、技术栈、快速上手 | 所有人（先读这个） |
| [02-系统架构.md](02-系统架构.md) | 分层架构、数据流、回测/实盘切换原则（CPU/GPU 统一向量化） | 所有人 |
| [03-核心数据结构.md](03-核心数据结构.md) | Bar、周期桶 dict、PERIODS 配置、stime 格式 | 开发者 |
| [04-周期合并机制.md](04-周期合并机制.md) | compute_bucket 算法、桶生命周期、warmup 机制 | 开发者（本项目最核心的机制之一） |
| [05-指标计算-EMA通道.md](05-指标计算-EMA通道.md) | EMA 定义、xp 算子批量版 + 纯 Python 增量版 | 开发者 |
| [06-交易策略详解.md](06-交易策略详解.md) | 通道偏离回撤策略、锁存逻辑、单桶单操作、`compute_signals` 主逻辑 | 策略研究 / 开发者 |
| [07-账户与执行器.md](07-账户与执行器.md) | Account 记账、Simulated/Broker Executor | 开发者 |
| [08-行情源Feed.md](08-行情源Feed.md) | Feed 抽象、分段查询、预热窗口、ChainedFeed | 开发者 |
| [09-引擎Engine与主流程.md](09-引擎Engine与主流程.md) | Engine 装配、`on_bars` 桶 CLOSE 语义、盈亏汇总 | 开发者 |
| [10-配置参数与运行指南.md](10-配置参数与运行指南.md) | CLI 参数全表（含 `--device {cpu,gpu,auto}`）、典型命令、输出解读、FAQ | 使用者 |
| [11-扩展指南.md](11-扩展指南.md) | 新增行情源 / 执行器 / 策略 / 周期的做法（含 `VectorizedStrategy` 模板） | 二次开发者 |
| [12-重构与性能内核.md](12-重构与性能内核.md) | evtrade 包结构、vectorized 引擎、xp 算子、CPU/GPU 统一路径、旧 numba/CUDA 退场记录 | 所有人（先读 01 再读这个） |
| [13-绩效评估与鲁棒选参框架.md](13-绩效评估与鲁棒选参框架.md) | 超额曲线口径、metrics 16 字段、滚动 WFO、邻域衰减 S、复合 score、帕累托、蒙特卡洛置换检验 | 选参/实盘前必读 |
| [14-统一策略契约.md](14-策略DSL与三端转译.md) | VectorizedStrategy 契约、`compute_signals(xp,bars,params)` 主入口、CPU/GPU 双端统一实现、新策略开发步骤（新写，旧 DSL→三端转译章节已废） | 新策略开发必读 |
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
| `evtrade/core/config.py` | DB_URL / PERIODS / TF1 / 默认资金 | 01、10 |
| `evtrade/core/timeutils.py` | `compute_bucket`、`bucket_ts_encoded`、`daterange` | 04 |
| `evtrade/core/aggregator.py` | `BarAggregator` | 04 |
| `evtrade/indicators/{ema,atr,boll,rsi}.py` | 纯 Python 增量版 + xp 算子批量版（无 numba / 无 CUDA_DEVICE_*） | 05 |
| `evtrade/strategies/vectorized_base.py` | ★ **唯一策略基类** `VectorizedStrategy` + 注册表 + `params_spec` + `format_signal_line` hook | 06、11、14 |
| `evtrade/strategies/channel_deviation.py` | `ChannelDeviationStrategy`（xp 算子 + Python FSM 共享） | 06 |
| `evtrade/strategies/ma_crossover.py` | `MACrossoverStrategy`（纯数组算子示例） | 14 |
| `evtrade/execution/account.py`、`evtrade/execution/base.py` | `Account`、`Executor` 两实现 | 07 |
| `evtrade/feeds/` | `Feed` 抽象 + `mysql_history` / `chained` / `_registry` | 08 |
| `evtrade/core/engine.py` | ★ `Engine`（逐 bar 路径，桶 CLOSE 语义，供实盘/对账） | 09 |
| `evtrade/core/vectorized_engine.py` | ★ `run_vectorized`（批量向量化路径，CPU/GPU 统一） | 12、13 |
| `evtrade/core/metrics.py` | `bars_to_arrays` / `trades_to_list` / `summarize`（16 字段） | 12、13 |
| `evtrade/core/data.py` | MySQL 拉取 + npz 缓存 + 合成数据 | 12 |
| `evtrade/core/sweep.py` / `evtrade/cli.py` / `evtrade/core/gpu.py` | 扫描 / CLI / GPU 基础设施（cupy 探测 + 预计算缓存；无 NVRTC 编译） | 10、12、13 |
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
| `compute_signals(xp, bars, params)` | 策略唯一抽象入口；`xp` = numpy 或 cupy；返回 `ndarray[int8]`，长度 = `len(bars["ts"])` |
| `compute_signals_for_one_bar(xp, bar, params)` | framework 单 bar 包装（默认包成单元素数组 → `compute_signals[0]`）；子类可覆写维护 instance 状态 |
| `xp` | 抽象后端（`numpy` 或 `cupy`）；策略代码禁止直接 `import numpy / cupy`，必须通过 `xp` 参数 |
| `VectorizedStrategy` | 唯一策略基类，定义 `compute_signals` 抽象方法 + `params_spec` schema 校验 + `format_signal_line` 默认 hook |
| `format_signal_line` | 策略展示 hook（Engine.on_bars 在 verbose 时打印）；默认仅打 OHLCV + 信号；channel_deviation 自定义展示 up/dw/dev |
| `reconcile` | vectorized vs Engine.on_bars 桶级信号 + 逐笔成交 bitwise 对账（`evtrade.core.replay.reconcile`） |
| `bucket_diff_cap` | `reconcile` 的差异容忍上限；默认 8（lenient），`EVT_RECONCILE_STRICT=1` 时 cap=0（strict） |

## 快速运行

```bash
pip install pymysql sqlalchemy pandas numpy      # 核心依赖 (无 numba)
pip install cupy-cuda12x                         # 可选: GPU 后端 (Linux)

# 单次回测 (CPU)
python -m evtrade backtest --strategy channel_deviation --device cpu \
    --period 5m --start 20250101 --end 20260903 --no-sleep

# 单次回测 (GPU, cupy 不可用时自动回退 cpu)
python -m evtrade backtest --strategy channel_deviation --device gpu \
    --period 5m --start 20250101 --end 20260903 --no-sleep

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
