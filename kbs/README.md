# kbs 知识库 · minute_bars 周期合并 + 通道偏离策略

本知识库是对项目源码的系统性中文分析文档，覆盖架构、数据结构、核心算法、策略逻辑、组件细节与运行指南。
**2026-09-05 起项目已重构为 `evtrade/` 包**（单文件拆分 + numba 流式内核 + 并发参数扫描，见 12 号文档）。
01~11 号文档的行号（如 `mysql_analyze_demo.py:70`）对应 git 历史中的原始单文件版本（commit 5585371），
逻辑与现行代码一致。生成日期：2026-09-04，更新：2026-09-08。

> **2026-09-08 同步**：`state_spec` 声明化、kernel/gpu 按 `state_spec` 工厂化、DSL→CUDA 投影迁至 `strategies/dsl.py`、
> 框架去特殊化（`bucket_table` 去 EMA 轨 / per-bar dict 契约 / 策略展示 hook）已补入 02、06、11、12、14 号文档与下方源码地图。
> `mysql_analyze_demo.py` 兼容入口已移除，统一用 `python -m evtrade`。

## 一句话简介

从 MySQL `minute_bars` 表读取 1 分钟行情，增量合并为多周期 K 线（1m/5m/15m/30m/1h/4h/1d），
用 EMA 通道轨（通达信蓝轨）的"极端偏离 → 回撤确认"逻辑产生 BUY/SELL 信号，
回测与实盘共用同一套 Engine，仅通过替换 Feed（行情源）与 Executor（下单器）切换。

## 文档索引

| 文档 | 内容 | 适合谁读 |
|---|---|---|
| [01-项目总览.md](01-项目总览.md) | 项目定位、核心特性、技术栈、快速上手 | 所有人（先读这个） |
| [02-系统架构.md](02-系统架构.md) | 分层架构、数据流、回测/实盘切换原则 | 所有人 |
| [03-核心数据结构.md](03-核心数据结构.md) | Bar、周期桶 dict、PERIODS 配置、stime 格式 | 开发者 |
| [04-周期合并机制.md](04-周期合并机制.md) | compute_bucket 算法、桶生命周期、warmup 机制 | 开发者（本项目最核心的机制之一） |
| [05-指标计算-EMA通道.md](05-指标计算-EMA通道.md) | EMA 定义、IncrementalEMA 增量状态机、EMAChannel | 开发者 |
| [06-交易策略详解.md](06-交易策略详解.md) | 通道偏离回撤策略、锁存逻辑、单桶单操作 | 策略研究 / 开发者 |
| [07-账户与执行器.md](07-账户与执行器.md) | Account 记账、Simulated/Broker Executor | 开发者 |
| [08-行情源Feed.md](08-行情源Feed.md) | Feed 抽象、分段查询、预热窗口、ChainedFeed | 开发者 |
| [09-引擎Engine与主流程.md](09-引擎Engine与主流程.md) | Engine 装配、on_bars 回调链、盈亏汇总 | 开发者 |
| [10-配置参数与运行指南.md](10-配置参数与运行指南.md) | CLI 参数全表、典型命令、输出解读、FAQ | 使用者 |
| [11-扩展指南.md](11-扩展指南.md) | 新增行情源 / 执行器 / 策略 / 周期的做法 | 二次开发者 |
| [12-重构与性能内核.md](12-重构与性能内核.md) | evtrade 包结构、numba 流式内核、差分测试、并发扫描、基准实测、GPU 路线 | 所有人（先读 01 再读这个） |
| [13-绩效评估与鲁棒选参框架.md](13-绩效评估与鲁棒选参框架.md) | 超额曲线口径、滚动 WFO、邻域衰减 S、复合 score、帕累托、蒙特卡洛置换检验 | 选参/实盘前必读 |
| [14-策略DSL与三端转译.md](14-策略DSL与三端转译.md) | 策略 DSL 写法、ctx 字段契约、Python/numba/CUDA 三端转译、kernel_dsl 特化内核、通用 CUDA kernel、新策略三步法 | 新策略开发必读 |
| [使用说明.md](使用说明.md) | **所有参数意思 + 完整命令行 + 网格扫参 + 资金模式 (阶段 2)** + 输出解读 | 操作手册, 跑前/看结果前查这个 |

## 建议阅读路径

- **新人上手**：01 → 02 → 10（跑起来）→ 06（理解策略）
- **修改/排查合并逻辑**：03 → 04 → 09
- **改策略或调参**：05 → 06 → 10 → 14（让新策略三端可用）
- **接实盘**：02 → 07 → 08 → 11

## 源码地图

现行代码已拆为 `evtrade/` 包，组件与文档的对应关系：

| 模块 | 内容 | 相关文档 |
|---|---|---|
| `evtrade/primitives.py` | `Bar`、`fmt` | 03 |
| `evtrade/core/config.py` | DB_URL / PERIODS / TF1 / 默认资金 | 01、10 |
| `evtrade/core/timeutils.py` | `compute_bucket`、`daterange` | 04 |
| `evtrade/core/aggregator.py` | `BarAggregator` | 04 |
| `evtrade/core/incremental_indicators.py` | `IncrementalEMA` / `EMAChannel`（热路径增量版） | 05 |
| `evtrade/indicators/ema.py` | EMA 纯函数（批量/复盘用）；含 `atr`/`boll`/`rsi` 同级模块 | 05 |
| `evtrade/strategies/base.py` | `StrategyBase` + 注册表 + `state_spec`/`params_spec`/展示 hook | 06、11、14 |
| `evtrade/strategies/channel_deviation.py` | `ChannelDeviationStrategy`（DSL 策略） | 06 |
| `evtrade/strategies/dsl.py` | 策略 DSL 转译器（三端）+ **CUDA 投影函数（从 `core/gpu.py` 迁入）** | 14 |
| `evtrade/execution/account.py`、`evtrade/execution/base.py` | `Account`、`Executor` 两实现 | 07 |
| `evtrade/feeds/` | `Feed` 抽象 + `mysql_history` / `chained` / `_registry` | 08 |
| `evtrade/core/engine.py` | `Engine`（参考引擎，verbose） | 09 |
| `evtrade/core/kernel.py` | ★ numba 流式决策内核（按 `state_spec` 工厂化） | 12 |
| `evtrade/core/kernel_dsl.py` | DSL 按策略特化内核（ctx→st 渲染 + splice） | 14 |
| `evtrade/core/data.py` | MySQL 拉取 + npz 缓存 + 合成数据 | 12 |
| `evtrade/core/sweep.py` / `evtrade/cli.py` / `evtrade/core/gpu.py` | 扫描 / CLI / GPU 档 | 12、14 |
| `evtrade/core/replay.py` | 行情回放 + 对账（per-bar dict 契约） | 10、13 |
| `tests/`、`scripts/benchmark.py` | 差分测试套件、基准脚本 | 12 |

> 现行代码已无 `mysql_analyze_demo.py` 兼容入口，统一用 `python -m evtrade`（子命令 `backtest`/`sweep`/`replay`/`params`）。
> 原始 685 行单文件实现见 git 历史（commit 5585371），下表行号以该版本为准（仅供追溯 01~11 号文档的历史行号引用）：

| 代码单元 | 行号 | 说明 |
|---|---|---|
| `DB_URL` / `TABLE` / `INTERVAL` | 28-30 | 数据库连接、表名、逐根 sleep 间隔 |
| `PERIODS` | 34-42 | 支持的 K 线周期配置表 |
| `TF1 = 21` | 44 | 通道轨 EMA 周期 |
| `Bar` (dataclass) | 49-59 | 统一行情 bar 结构 |
| `compute_bucket` | 70-82 | 合并桶时间戳计算（前开后闭，右端点标注） |
| `BarAggregator` | 91-146 | 增量周期合并器 |
| `IncrementalEMA` | 165-205 | 增量 EMA 状态机（O(1)/桶） |
| `EMAChannel` | 208-229 | 通达信蓝色通道轨（增量版） |
| `ChannelDeviationStrategy` | 249-327 | 通道偏离回撤策略（锁存 + 单桶单操作） |
| `Account` | 332-361 | 资金/持仓记账 |
| `Executor` 层次 | 366-418 | 下单抽象：模拟 / 实盘占位 |
| `daterange` / `MySQLBacktestFeed` / `ChainedFeed` | 430-511 | 行情源 |
| `Engine` | 516-625 | 统一引擎 |
| `main` | 637-680 | 命令行入口 |

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
| `state_spec` | DSL 策略在类上声明的持久状态字段 schema（`{"name": {"type": bool/int/float, "default": 标量}}`）；框架自动投影到 numba jitclass / CUDA device 函数 / Python ctx。空 dict = 无持久状态。DSL 策略必声明，缺则编译期抛 `CompileError`。详见 12、14 号文档 |
| `bundle_per_bar` | 内核/replay 把 per-bar 数组打包成 dict 契约（`{"up","dw","h","l","ts",...}`），framework 不命名指标字段；策略 hook 按需取值 |

## 快速运行

```bash
pip install pymysql sqlalchemy numpy numba
# 单次回测 (默认 numba 内核, 与原实现逐笔等价; 周期任意 m/h/d, 支持倍投 --scale)
python -m evtrade backtest --period 5m --start 20250101 --end 20260903 --no-sleep
# 参数扫描: 滚动 WFO 多窗 + 费率 + 邻域衰减评分 + 蒙特卡洛 (选参标准流程, 见 13)
python -m evtrade sweep --data-cache cache --grid "low1=1.0,1.5,2.0" ^
    --splits 20260101,20260401,20260701 --fee-bp 5 --mc 300
# 实盘一致性验收: 录制 bar 日志 -> 回放 -> 与参考引擎对账 (见 10/13)
python -m evtrade replay --log live_demo.log --warmup-until 20260701 --against-ref
```

前提：能访问 MySQL `192.168.10.2:33066` 的 `evtrade` 库（详见 08 文档，注意凭据为敏感信息，
可用环境变量 `EVTRADE_DB_URL` 覆盖）。更多用法见 10、12、13 号文档。
