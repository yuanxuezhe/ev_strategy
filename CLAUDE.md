# CLAUDE.md

> 给 Claude Code / 协作者的工作约定。改本文件时请同步告知团队。

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
| `kbs/02-系统架构.md` | 分层架构、数据流、回测/实盘切换 |
| `kbs/03-核心数据结构.md` | Bar、周期桶 dict、PERIODS、stime 格式 |
| `kbs/04-周期合并机制.md` | compute_bucket、桶生命周期、warmup |
| `kbs/05-指标计算-EMA通道.md` | EMA、IncrementalEMA、EMAChannel |
| `kbs/06-交易策略详解.md` | 通道偏离回撤策略、锁存、单桶单操作、策略 hook |
| `kbs/07-账户与执行器.md` | Account、Simulated/Broker Executor |
| `kbs/08-行情源Feed.md` | Feed 抽象、分段查询、预热窗口、ChainedFeed |
| `kbs/09-引擎Engine与主流程.md` | Engine 装配、on_bars 回调链、盈亏汇总 |
| `kbs/10-配置参数与运行指南.md` | CLI 参数全表、典型命令、输出解读 |
| `kbs/11-扩展指南.md` | 新增行情源/执行器/策略/周期 + 策略 hook 协议 |
| `kbs/12-重构与性能内核.md` | evtrade 包结构、numba 流式内核、bucket_table/bundle_per_bar、GPU 路线 |
| `kbs/13-绩效评估与鲁棒选参框架.md` | WFO、邻域衰减 S、score、置换检验 |
| `kbs/14-策略DSL与三端转译.md` | DSL 写法、ctx 字段契约、Python/numba/CUDA 三端转译 |

## 3. 源码地图（核心要点）

- 包结构：`evtrade/{cli,primitives}.py` + `core/{config,timeutils,aggregator,incremental_indicators,engine,data,kernel,kernel_dsl,sweep,replay,gpu,permutation,capability}.py` + `indicators/{ema,atr,boll,rsi}.py` + `strategies/{base,dsl,channel_deviation,example_*.py}` + `execution/{account,base}.py` + `feeds/{base,mysql_history,chained,_registry}.py`。
- 策略持久状态 = `StrategyBase.state_spec`（DSL 必填；空 dict = 无持久状态；缺声明编译期抛 `CompileError`）。
- 三端投影：numba（`kernel._build_kernel_state_source`）+ CUDA（`strategies/dsl.py::build_cuda_state_decls` / `build_cuda_strategy_check_call` / `render_cuda_device_function`）+ Python ctx。
- 框架契约：per-bar 数组 = `bundle_per_bar` dict（`{"sig", "per_bar": {"up","dw","ts","o","h","l","c","v"}}`）；`bucket_table` 只输出 OHLCV + 信号轨迹。
- CLI 入口：`python -m evtrade {backtest,sweep,replay,params}`。策略参数：`--params "k1:v1;k2:v2"`（按 `params_spec` 校验）。

## 4. OpenSpec 技能包

- 项目 `.claude/skills/` 下的 5 个 skill（explore / propose / apply-change / archive-change / sync-specs）。
- `.claude/commands/opsx/` 下对应 5 个 slash command（`/opsx:explore`、`/opsx:propose`、`/opsx:apply`、`/opsx:archive`、`/opsx:sync`）。
- 常用命令：
  - `openspec validate --specs` — 校验 spec
  - `openspec list --specs` / `openspec list` — 列出 spec / change
  - `openspec show <name>` — 查看详情（spec 或 change）

## 5. 关键约定（写代码时不要破坏）

- 框架层不假定任何指标字段名（up/dw/low_dev 等均不出现于 framework CLI / kernel API 的关键字参数）。
- 策略持久状态必须经 `state_spec` 声明，框架自动三端投影；**不要**手动改 `KernelState` jitclass 或 CUDA 模板加状态字段。
- 策略展示由三个 hook 完成：`format_signal_line` / `get_extra_bucket_columns` / `get_extra_signal_columns`；不要在 `bucket_table` 里硬塞指标列。
- DSL→CUDA 投影逻辑在 `strategies/dsl.py`，`core/gpu.py` 不持有。
- numba 内核 step + CUDA 模板中的 EMA 计算是硬编码的，换指标需同步改这两处（见 `kbs/11` §5 警示）。
- 信号行打印由策略 `format_signal_line` hook 完成，Engine 只 `print` 不假设 `info` 键集。

## 6. 验证命令

- KB ↔ 代码一致性：`grep -r state_spec kbs/` 应在 02、06、11、12、14 出现；`grep -rn "_CTX_TO_KERNEL\|_bucket_ts\|_low_acted\|mysql_analyze_demo" kbs/` 应为 0 命中。
- spec 校验：`openspec validate --specs` 应通过。
- 测试：`uv run pytest -q`（含差分、状态 spec 泛化、CUDA 差分、回放对账）。
