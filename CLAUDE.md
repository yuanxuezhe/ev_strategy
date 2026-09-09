# CLAUDE.md

> 给 Claude Code / 协作者的工作约定。改本文件时请同步告知团队。
>
> 本项目为 2026-09-09 重构后版本：DSL 渲染层 / numba 流式内核 / NVRTC CUDA 编译
> **已下线**；策略唯一入口 = `VectorizedStrategy.compute_signals(xp, bars, params)`；
> CPU (numpy) / GPU (cupy) 双端统一由 cupy 高阶封装提供。

## 0. 核心工作流（最重要）

本项目走 **KB/spec 先行 + openspec 驱动** 的开发流程。任何代码优化或新功能，**严格按以下顺序**：

1. **更新 spec**：修改 `openspec/specs/evtrade-architecture/spec.md`（或在新增能力时新建 `openspec/specs/<capability>/spec.md`），
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
| 行为权威 | `openspec/specs/evtrade-architecture/spec.md` | 8 条 Requirement + Scenario；通过 `openspec validate --specs` 校验 |
| 中文详述 | `kbs/*.md`（14 份） | 与 spec 一一映射；改一处必改另一处 |
| 变更提案 | `openspec/changes/<name>/` | proposal.md + spec delta + design.md + tasks.md |

## 2. KB 索引（14 份）

| 文档 | 内容 |
|---|---|
| `kbs/01-项目总览.md` | 项目定位、核心特性、技术栈、快速上手 |
| `kbs/02-系统架构.md` | 分层架构、数据流、回测/实盘切换（CPU/GPU 统一向量化） |
| `kbs/03-核心数据结构.md` | Bar、周期桶 dict、PERIODS、stime 格式 |
| `kbs/04-周期合并机制.md` | compute_bucket、桶生命周期、warmup |
| `kbs/05-指标计算-EMA通道.md` | EMA / 通道轨 xp 版 + 纯 Python 增量版 |
| `kbs/06-交易策略详解.md` | 通道偏离回撤策略、锁存、单桶单操作、策略 hook |
| `kbs/07-账户与执行器.md` | Account、Simulated/Broker Executor |
| `kbs/08-行情源Feed.md` | Feed 抽象、分段查询、预热窗口、ChainedFeed |
| `kbs/09-引擎Engine与主流程.md` | Engine 装配、`on_bars` 桶 CLOSE 语义、盈亏汇总 |
| `kbs/10-配置参数与运行指南.md` | CLI 参数全表（含 `--device {cpu,gpu,auto}`）、典型命令、输出解读 |
| `kbs/11-扩展指南.md` | 新增行情源/执行器/策略/周期 + VectorizedStrategy 模板 |
| `kbs/12-重构与性能内核.md` | evtrade 包结构、vectorized 引擎、xp 算子、CPU/GPU 统一路径 |
| `kbs/13-绩效评估与鲁棒选参框架.md` | WFO、邻域衰减 S、score、置换检验 |
| `kbs/14-统一策略契约.md` | VectorizedStrategy 契约、xp 算子、CPU/GPU 双端统一（旧 DSL→三端转译章节已废） |

## 3. 源码地图（核心要点）

- 包结构：`evtrade/{cli,primitives}.py` + `core/{config,timeutils,aggregator,engine,vectorized_engine,data,metrics,sweep,replay,gpu,permutation,capability}.py` + `indicators/{ema,atr,boll,rsi}.py` + `strategies/{vectorized_base,channel_deviation,ma_crossover}.py` + `execution/{account,base}.py` + `feeds/{base,mysql_history,chained,_registry}.py`。
- 策略唯一基类：`VectorizedStrategy` (`evtrade/strategies/vectorized_base.py`)。
  - 唯一抽象方法：`compute_signals(self, xp, bars, params) -> xp.ndarray[int8]`。
  - framework 单 bar 包装：`compute_signals_for_one_bar(self, xp, bar, params) -> int`（子类可覆写维护 instance 状态）。
  - 持久状态：Python 实例属性（`self._up_st / self._dw_st / self._fsm` 等），**不再**用 `state_spec` AST 白名单投影。
- 框架契约：桶级数组 = `{"ts","o","h","l","c","v","mark","n_bars"}` 1D 数组 dict（`compute_signals` 接收）；
  per-bar dict = `{"ts","o","h","l","c","v","mark"}`（`compute_signals_for_one_bar` 接收单 bar）。
- CLI 入口：`python -m evtrade {backtest,sweep,replay,params}`。
  设备参数：`--device {cpu, gpu, auto}`（统一，替代旧 `--engine {kernel, ref, vectorized}`）。
  策略参数：`--params "k1:v1;k2:v2"`（按 `params_spec` 校验）。
- 指标：`evtrade/indicators/*.py` 提供批量 xp 版 (`xp_ema / xp_ema_channel / xp_atr / xp_rsi / ...`) + 纯 Python 增量版 (`ema_push / ema_current / ema_channel_push / ema_channel_current / ...`)，无 numba / 无 CUDA。

## 4. OpenSpec 技能包

- 项目 `.claude/skills/` 下的 5 个 skill（explore / propose / apply-change / archive-change / sync-specs）。
- `.claude/commands/opsx/` 下对应 5 个 slash command（`/opsx:explore`、`/opsx:propose`、`/opsx:apply`、`/opsx:archive`、`/opsx:sync`）。
- 常用命令：
  - `openspec validate --specs` — 校验 spec
  - `openspec list --specs` / `openspec list` — 列出 spec / change
  - `openspec show <name>` — 查看详情（spec 或 change）

## 5. 关键约定（写代码时不要破坏）

- 框架层不假定任何指标字段名（up/dw/low_dev 等均不出现于 framework CLI / vectorized API 的关键字参数）。
- 策略唯一抽象入口是 `compute_signals(xp, bars, params)`；持久状态用 Python 实例属性，**不要**重新引入 `state_spec` AST 白名单或 `KernelState` jitclass。
- 策略展示由一个 hook 完成：`format_signal_line(ts, sig, info) -> str`（`VectorizedStrategy` 默认仅打 OHLCV + 信号）；
  不再使用 `get_extra_bucket_columns` / `get_extra_signal_columns`（旧 bucket_table / per-bar bundle 已删除）。
- 指标计算在策略 `compute_signals` / `compute_signals_for_one_bar` 内部通过 `evtrade.indicators.{xp_ema,ema_push,...}` 自维护；
  framework 不再持有任何 EMA 公式或指标字段。
- 信号行打印由策略 `format_signal_line` hook 完成，Engine 只 `print` 不假设 `info` 键集。
- Engine.on_bars 是**桶 CLOSE 语义**：桶切换时 (`cur.ts != last_cur.ts`) 用上一桶 finalized OHLCV 驱动策略一次；
  vectorized 路径在桶级 finalized OHLCV 上计算指标。两条路径通过 `reconcile` 对账，默认 `bucket_diff_cap=8` 容忍 EMA 累积漂移。
- `--device {cpu,gpu,auto}` 是唯一后端选择参数；旧 `--engine {kernel,ref,vectorized}` 已删除（仅 `--engine` 兼容层打印 DeprecationWarning 后自动映射）。

## 6. 验证命令

- KB ↔ 代码一致性：`grep -r compute_signals kbs/` 应在 02、05、06、09、11、13、14 出现；`grep -rn "numba\|@njit\|state_spec\|CUDA_DEVICE_\|kernel_dsl\|cuda_sweep_window_generic\|strategies\.dsl" kbs/ evtrade/` 应为 0 命中（tests/ 内可有注释）。
- spec 校验：`openspec validate --specs` 应通过。
- 测试：`uv run pytest -q`（应 110+ passed；含 CPU/GPU 容差、vectorized-vs-Engine 对账、metrics 字段集）。
- 对账：`python -m evtrade replay --log <log.csv> --strategy channel_deviation --device cpu --against-ref` 应输出 PASS（`bucket_diff_cap=8` 默认 lenient；`EVT_RECONCILE_STRICT=1` 时 cap=0）。
