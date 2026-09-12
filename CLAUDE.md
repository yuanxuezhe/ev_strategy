# CLAUDE.md

> 给 Claude Code / 协作者的工作约定。改本文件时请同步告知团队。
>
> 本项目为 2026-09-09 重构后版本：DSL 渲染层 / numba 流式内核 / NVRTC CUDA 编译
> **已下线**；策略唯一入口 = `VectorizedStrategy.step(state, bar, params) -> (state, sig)`
> + `init_state(params)`，state 由引擎持有；CPU (torch cpu) / GPU (torch cuda)
> 双端统一由 PyTorch 后端提供。

## 0. 核心工作流（最重要）

本项目走 **KB/spec 先行 + openspec 驱动** 的开发流程。任何代码优化或新功能，**严格按以下顺序**：

1. **更新 spec**：修改 `openspec/specs/evtrade-architecture/spec.md`（或在新增能力时新建 `openspec/specs/<new-name>/spec.md`），
   用 Requirement + Scenario 描述**目标可观测行为**。
2. **更新 KB**：同步更新 `kbs/` 中对应的中文文档。**spec 为权威，KB 为中文详述投影**；二者必须同步演进。
3. **openspec 走 change**：用项目内的 openspec 技能包（`.claude/skills/openspec-*` 或 `.claude/commands/opsx/*.md`）
   创建 change proposal，走 `proposal → specs → design → tasks → apply`。
4. **写代码**：按 tasks 实施，过程中保持代码与 spec/KB 一致。
5. **archive**：落地后 `openspec archive`；最后再确认 spec / KB / 代码三者已对齐。

> 跳过 1+2 直接动代码 = 违反约定。每次提交应在 commit message 里写明对应的 change 名 / spec 段。

## 1. 单一事实源

| 角色 | 路径 | 说明 |
|---|---|---|
| 行为权威 | `openspec/specs/evtrade-architecture/spec.md` | 13 条 Requirement + Scenario；通过 `openspec validate --specs` 校验 |
| 中文详述 | `kbs/*.md`（15 份 + README + 使用说明） | 与 spec 一一映射；改一处必改另一处 |
| 变更提案 | `openspec/changes/<name>/` | proposal.md + spec delta + design.md + tasks.md |

## 2. KB 索引（15 份 + README + 使用说明）

| 文档 | 内容 |
|---|---|
| `kbs/01-项目总览.md` | 项目定位、核心特性、技术栈、快速上手 |
| `kbs/02-系统架构.md` | 分层架构、数据流、回测/实盘切换（CPU/GPU 统一向量化） |
| `kbs/03-核心数据结构.md` | Bar、周期桶 dict、`resolve_period_seconds` 周期秒数、stime 格式 |
| `kbs/04-周期合并机制.md` | `compute_bucket_general` 桶算法、桶生命周期、warmup |
| `kbs/05-指标计算-EMA通道.md` | EMA / 通道轨增量版（`ema_step`/`ema_channel_step`）+ numpy 批量参考（`ema`/`ema_channel`） |
| `kbs/06-交易策略详解.md` | 通道偏离回撤策略、锁存、单桶单操作、策略 hook |
| `kbs/07-账户与执行器.md` | Account、`trade_decision` 单一成交决策、Simulated Executor + 自定义实盘 Executor |
| `kbs/08-行情源Feed.md` | 行情数据加载（`core/data.py` `load_bars`/`synthetic_bars`）、bar 流契约、预热窗口 |
| `kbs/09-引擎Engine与主流程.md` | Engine 装配、`on_bars` 桶 CLOSE 语义、`metrics.summarize` 盈亏汇总 |
| `kbs/10-配置参数与运行指南.md` | CLI 参数全表（含 `--device {cpu,gpu,auto}`）、典型命令、输出解读 |
| `kbs/11-扩展指南.md` | 新增 bar 流/执行器/策略/周期 + VectorizedStrategy 模板 |
| `kbs/12-重构与性能内核.md` | evtrade 包结构、vectorized 引擎、CPU/GPU 统一路径、重构历史 |
| `kbs/13-绩效评估与鲁棒选参框架.md` | WFO、邻域衰减 S、score、置换检验 |
| `kbs/14-策略DSL与三端转译.md` | 旧 DSL→三端转译（numba/CUDA/state_spec）**已下线**，归档说明 |
| `kbs/15-PyTorch统一策略.md` | PyTorch 统一后端（`backends.get_xp`）、CPU/GPU 双端统一 |

## 3. 源码地图（核心要点）

- 包结构：`evtrade/{cli,primitives,backends}.py` + `core/{config,timeutils,aggregator,engine,vectorized_engine,batched_sweep,data,metrics,sweep,replay,tsbucket,permutation,_harness}.py` + `indicators/{ema}.py` + `strategies/{vectorized_base,channel_deviation,ma_crossover,filtered_mr,_defaults_loader}.py`（`_defaults/` 子目录存落盘默认参数） + `execution/{account,base}.py`。
  （行情子包已删——数据加载内联 `core/data.py`；`atr/boll/rsi` 指标已删——仅留 EMA；GPU 探测并入 `backends`；旧 `core/gpu.py` 更名 `tsbucket.py`；`batched_sweep.py` 是 2026-09-11 GPU-batched sweep 的 opt-in 入口。）
- 策略唯一基类：`VectorizedStrategy` (`evtrade/strategies/vectorized_base.py`)。
  - 唯一抽象方法：`step(self, state, bar, params) -> (state, int)`；`state` 由引擎持有、跨调用持续，策略**无 instance 持久状态**。
  - state 初值：`init_state(self, params) -> state`（无状态策略默认返回 `None`；stateful 策略覆写返 `@dataclass`）。
  - 持久状态：`@dataclass` state（策略自定字段，如 EMA 增量 + FSM），**不再**用 `state_spec` AST 白名单或 instance attr。
- 框架契约：单桶 bar dict = `{"ts","o","h","l","c","v","mark"}`（标量，`step` 接收，`mark=0` 为预热段）；
  引擎侧另有桶级数组 dict `{"ts","o","h","l","c","v","mark","n_bars"}`（`vectorized_engine._aggregate_buckets` 产出，逐桶喂给 step）。
- CLI 入口：`python -m evtrade {backtest,sweep,replay,params}`。
  设备参数：`--device {cpu, gpu, auto}`（统一，替代旧 `--engine {kernel, ref, vectorized}`）。
  策略参数：`--params "k1:v1;k2:v2"`（按 `params_spec` 校验）。
- 指标：`evtrade/indicators/ema.py` 提供增量版 (`ema_step / ema_channel_step`，`@dataclass` state 由策略 `step` 自维护) + numpy 批量参考版 (`ema / ema_channel`，仅离线校验用)，无 numba / 无 CUDA；**只留 EMA**，`atr / boll / rsi` 已删。

## 4. OpenSpec 技能包

- 项目 `.claude/skills/` 下的 5 个 skill（explore / propose / apply-change / archive-change / sync-specs）。
- `.claude/commands/opsx/` 下对应 5 个 slash command（`/opsx:explore`、`/opsx:propose`、`/opsx:apply`、`/opsx:archive`、`/opsx:sync`）。
- 常用命令：
  - `openspec validate --specs` — 校验 spec
  - `openspec list --specs` / `openspec list` — 列出 spec / change
  - `openspec show <name>` — 查看详情（spec 或 change）

## 5. 关键约定（写代码时不要破坏）

- 框架层不假定任何指标字段名（up/dw/low_dev 等均不出现于 framework CLI / vectorized API 的关键字参数）。
- 策略唯一抽象入口是 `step(state, bar, params) -> (state, sig)`；持久状态用 `@dataclass` state（引擎持有跨调用），
  **不要**重新引入 `state_spec` AST 白名单 / `KernelState` jitclass / `self._xxx` instance 持久状态。
- 策略展示由一个 hook 完成：`format_signal_line(ts, sig, info) -> str`（`VectorizedStrategy` 默认仅打 `ts/sig/side`）；
  不再使用 `get_extra_bucket_columns` / `get_extra_signal_columns`（旧 bucket_table / per-bar bundle 已删除）。
- 指标计算在策略 `step` 内部通过 `evtrade.indicators.{ema_step, ema_channel_step}` 自维护（仅 EMA）；
  framework 不再持有任何 EMA 公式或指标字段。
- 信号行打印由策略 `format_signal_line` hook 完成，Engine 只 `print` 不假设 `info` 键集。
- Engine.on_bars 是**桶 CLOSE 语义**：桶切换时 (`cur.ts != last_cur.ts`) 用上一桶 finalized OHLCV 驱动策略一次；
  vectorized 路径在桶级 finalized OHLCV 上计算指标。两条路径通过 `reconcile` 对账，默认 `bucket_diff_cap=8` 容忍 EMA 累积漂移。
- `--device {cpu,gpu,auto}` 是唯一后端选择参数（torch CPU / torch CUDA 统一后端，`backends.get_xp` 路由；
  入口统一 `backends.resolve_device` resolve，`--device gpu` 无 CUDA 抛 ValueError）。旧 `--engine {kernel,ref,vectorized}` 已删除。

## 6. 验证命令

- KB ↔ 代码一致性（design §5 grep hygiene，三条应 0 命中）：
  - `grep -rn "feeds\|gpu_info\|capability\|BrokerExecutor\|xp_ema\|atr_step\|rsi_step\|boll_step" evtrade/ kbs/ CLAUDE.md openspec/specs/`
  - `grep -rn "compute_bucket\b\|daterange\|PERIODS\|trades_to_list\|build_engine\|print_summary" evtrade/ kbs/ CLAUDE.md openspec/specs/`（`\b` 词边界：`compute_bucket_general` 不算）
  - `grep -rn "--no-sleep\|--step-days\|--show-bars\|--bars-out\|ann_excess_pct" evtrade/ kbs/ CLAUDE.md`
  （spec / kbs 内"已删除/已下线"的显式删注与历史 change-note 除外。）
- spec 校验：`openspec validate --specs` 应通过。
- 测试：`uv run pytest -q`（应 60+ passed；含 CPU/GPU 容差、vectorized-vs-Engine 对账、metrics 字段集、batched_step、filtered_mr、pyproject 依赖审计）。
- 对账：`python -m evtrade replay --log <log.csv> --strategy channel_deviation --device cpu --against-ref` 应输出 PASS（`bucket_diff_cap=8` 默认 lenient；`EVT_RECONCILE_STRICT=1` 时 cap=0）。
