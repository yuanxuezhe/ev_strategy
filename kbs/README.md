# KB 知识库 · evtrade 框架

本知识库是 `openspec/specs/evtrade-architecture/spec.md` 的中文投影,
与 spec 一一映射(改一处必改另一处)。

**2026-09-13 后核心约定**: framework 不再持有资金/持仓/撮合/PnL/收益概念;
业务概念下放到策略 `step` 内部 + `@dataclass state` 字段 + 策略内联撮合数学;
framework 只做 (1) 桶预计算 + (2) `step` 驱动循环 + (3) 可选 GPU-batched sweep hook。
CPU/GPU 双端统一由 PyTorch 后端提供(`backends.get_xp`)。

## 一句话简介

从 MySQL `minute_bars` 表读取 1 分钟行情(`core/data.py::load_bars`),
增量合并为多周期 K 线(任意 m/h/d 周期),
用 EMA 通道轨(通达信蓝轨)的"极端偏离 → 回撤确认"逻辑产生 BUY/SELL 信号,
回测与实盘共用同一套 Engine, 仅通过替换 bar 流(行情源)切换。
策略资金/持仓/撮合/账本由策略 `step` 内部自管。

## 文档索引

| 文档 | 内容 | 适合谁读 |
|---|---|---|
| [01-总览.md](01-总览.md) | 项目定位、分层架构、核心特性、技术栈、包结构、文档地图 | 所有人(先读这个) |
| [02-数据与桶.md](02-数据与桶.md) | `Bar` / 桶 dict / `stime` 14 位 / `compute_bucket_general` / `BarAggregator` / 任意 m/h/d | 开发者 |
| [03-指标-EMA.md](03-指标-EMA.md) | EMA 三形态(`ema_step` / `ema` / `torch_ema`) + 双 rail `ema_channel_step` + dataclass state | 开发者 |
| [04-策略.md](04-策略.md) | 三策略(channel_deviation / ma_crossover / filtered_mr) + 统一契约 + 策略自管执行 | 策略研究 / 开发者 |
| [05-引擎与CLI.md](05-引擎与CLI.md) | Engine / `run_vectorized` 装配 + 桶 CLOSE 语义 + CLI 全表 + 数据加载 + 输出解读 + FAQ | 使用者 |
| [06-扩展指南.md](06-扩展指南.md) | 新增 bar 流 / 策略 / 周期 / 指标 + `VectorizedStrategy` 模板 + 实盘接入 | 二次开发者 |
| [07-PyTorch与性能.md](07-PyTorch与性能.md) | PyTorch 统一后端 + CPU/GPU 透明路由 + 性能内核 + `batched_step` opt-in hook | 性能优化 / 写新策略 |
| [08-选参与鲁棒性.md](08-选参与鲁棒性.md) | WFO + 邻域衰减 S + 复合 score + 帕累托前沿 + 落盘默认参数 | 选参 / 实盘前必读 |
| [archive/DSL-HISTORY.md](archive/DSL-HISTORY.md) | 旧 DSL → 三端转译 (numba/CUDA/state_spec) 已下线, 历史存档 | 想了解历史 / 归档 |

## 建议阅读路径

- **新人上手**: 01 → 05(跑起来) → 04(理解策略) → 03(指标)
- **修改/排查合并逻辑**: 02
- **改策略 / 调参**: 03 → 04 → 05 → 07(CPU/GPU 双端)
- **接实盘**: 06(新 bar 流) → 04(策略自管执行)
- **选参**: 08
- **了解历史重构**: 07 §8 重构历史 + archive/DSL-HISTORY

## 源码地图

现行代码已拆为 `evtrade/` 包, 组件与文档的对应关系:

| 模块 | 内容 | 相关文档 |
|---|---|---|
| `evtrade/__init__.py` / `__main__.py` / `cli.py` | 顶层 re-export + `python -m evtrade` 入口 + argparse 子命令 | 05 |
| `evtrade/primitives.py` | `Bar` / `fmt` / `sig_to_side` | 02 |
| `evtrade/backends.py` | `get_xp` / `gpu_available` / `resolve_device` / `to_tensor` / `to_host`(PyTorch CPU/GPU 统一后端) | 07 |
| `evtrade/core/timeutils.py` | `compute_bucket_general` / `bucket_ts_encoded` / `resolve_period_seconds` / 历法 | 02 |
| `evtrade/core/aggregator.py` | `BarAggregator`(1m → 目标周期桶合并) | 02 |
| `evtrade/core/engine.py` | ★ `Engine`(逐 bar 路径, 桶 CLOSE 语义, 仅驱动 step) | 05 |
| `evtrade/core/vectorized_engine.py` | ★ `run_vectorized`(批量路径, numpy 桶聚合 + strategy-step-only 信号循环, 仅驱动 step) | 05 |
| `evtrade/core/batched_sweep.py` | ★ `run_batched`(opt-in GPU 批量 sweep; `ma_crossover` 已接入) | 07 |
| `evtrade/core/data.py` | `load_bars`(MySQL 拉取 + npz 缓存) + `synthetic_bars` + `DB_URL` / `TABLE` 常量 | 05 |
| `evtrade/core/sweep.py` | 并发参数扫描 + WFO + 邻域衰减 score(按 `state.primary_score`) | 08 |
| `evtrade/core/tsbucket.py` | 桶 ts/mark 预计算(`precompute_ts_mark`) + LRU 缓存 | 02、07 |
| `evtrade/core/_harness.py` | `ListBarFeed`(bar 流 adapter, 鸭子类型 `.stream() -> Iterator[Bar]`) | 02、05 |
| `evtrade/indicators/ema.py` | step 增量 `ema_step` / `ema_channel_step` + numpy 批量 `ema` / `ema_channel` + torch 批量 `torch_ema` | 03、07 |
| `evtrade/strategies/vectorized_base.py` | ★ **唯一策略基类** `VectorizedStrategy` + 注册表 + `params_spec` 校验 + `step` 抽象 + `batched_step` opt-in hook + `format_signal_line` hook | 04、06、07 |
| `evtrade/strategies/channel_deviation.py` | `ChannelDeviationStrategy`(`ema_channel_step` 增量 + Python FSM; mark=0 跳过; 不 batched) | 04 |
| `evtrade/strategies/ma_crossover.py` | `MACrossoverStrategy`(纯 EMA + batched_step opt-in) | 04、07 |
| `evtrade/strategies/filtered_mr.py` | `FilteredMRStrategy`(4 重过滤均值回归: 大周期顺势 + ADX + ATR + close FSM) | 04 |
| `evtrade/strategies/_defaults_loader.py` | 默认参数落盘 / 加载 / sweep 自动选最优并 git commit | 05、08 |
| `evtrade/strategies/_defaults/` | 落盘默认参数 JSON(按策略名) | 05、08 |

> 已下线: `evtrade/feeds/` 子包; `evtrade/execution/` 子包(Account / Executor / SimulatedExecutor /
> trade_decision); `evtrade/core/{config,metrics,replay,permutation}.py`(`drop-engine-finance`)。

## 术语表

| 术语 | 含义 |
|---|---|
| bar / K线 | 一根行情记录(OHLCV) |
| 1m bar | 数据库 `minute_bars` 中的原始 1 分钟行情 |
| 周期桶(bucket) | 若干根 1m bar 合并成的一根目标周期 K 线(如 5m) |
| 桶 ts | 桶的时间戳; **前开后闭区间, 标注右端点** |
| 闭合桶 | 已收完的桶(后续 1m bar 属于新桶了) |
| 当前桶(cur) | 尚未收完、正在被最新 1m bar 更新的桶 |
| UP / DW | 通道上轨 = `EMA(H, tf1)`, 下轨 = `EMA(L, tf1)`(通达信蓝色通道轨) |
| 偏离(dev) | 价格相对通道轨的百分比距离, 共 4 种定义, 见 [04-策略 §3.2](04-策略.md) |
| 锁存(latch / hysteresis) | "极端偏离"标记一旦置位保持不变, 直到"回撤确认"触发信号并解除 |
| mark | 桶的标志位: 0 = 预热(只累积指标), 1 = 策略期(驱动策略) |
| warmup | 策略起始日之前的预热数据, 用于让 EMA 有足够历史 |
| `step(state, bar, params) -> (state, sig)` | 策略唯一抽象入口; `bar` = 单桶 `{ts, o, h, l, c, v, mark}`; 返回 `(new_state, sig)`, `sig ∈ {-1, 0, 1}` |
| `init_state(params) -> state` | state 初值(无状态策略默认 `None`; stateful 策略覆写返 `@dataclass`); state 由 engine 持有跨调用 |
| `state` | 持久状态(推荐 `@dataclass`); 策略**无 instance attr**, 由引擎在 `step` 调用间持有 |
| `xp` | `backends.get_xp(device) -> torch.device`(`"cpu"→cpu`, `"gpu"→cuda`); 策略代码不直接 import 后端 |
| `VectorizedStrategy` | 唯一策略基类, 定义 `step` 抽象方法 + `params_spec` schema 校验 + `format_signal_line` 默认 hook |
| `format_signal_line` | 策略展示 hook(Engine.on_bars 在 verbose 时打印); 约定: sig != 0 时 MUST 含方向词 + 价格 + OHLCV |
| `batched_step` | opt-in GPU 批量 hook(仅产 sig); `ma_crossover` 已接入, `channel_deviation` / `filtered_mr` 不实现 |
| `state.primary_score` | 策略自选标量(默认 `cash + position * price`), sweep 据此算年化收益 |
| EVTRADE_DB_URL | 环境变量, 覆盖 `data.DB_URL` 默认值(临时切库 / 离线 / 测试) |
| EVTRADE_TABLE | 环境变量, 覆盖 `data.TABLE` 默认 `minute_bars` |

## 快速运行

```bash
# 安装 (torch 统一 CPU/GPU 后端; 无 numba)
pip install pymysql sqlalchemy pandas numpy torch
# GPU 机器一次性:
uv sync --extra gpu              # 锁住 torch==2.9.0+cu128

# 单次回测 (CPU)
python -m evtrade backtest --strategy channel_deviation --device cpu \
    --period 5m --start 20250101 --end 20260903

# 单次回测 (GPU; 无 CUDA 时 --device gpu 会抛 ValueError, 用 --device auto 自动降级)
python -m evtrade backtest --strategy channel_deviation --device auto \
    --period 5m --start 20250101 --end 20260903

# 网格扫描 + WFO + score 评分 (选参标准流程, 见 08)
python -m evtrade sweep --strategy channel_deviation --device auto \
    --grid "low1=1.0,1.5,2.0" --grid "high2=0.3,0.5,0.8" \
    --splits 20260101,20260401,20260701

# 无库体验 (合成数据)
python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 \
    --period 5m --start 20260101 --end 20260201
```

前提: 能访问 MySQL `192.168.10.2:33066` 的 `evtrade` 库(详见 05 §数据库连接,
凭据为敏感信息, 可用环境变量 `EVTRADE_DB_URL` 覆盖)。
更多用法见 05 §典型命令 + 06 §扩展指南 + 08 §选参工作流。
