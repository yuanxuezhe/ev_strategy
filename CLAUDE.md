# CLAUDE.md

> 给 Claude Code / 协作者的工作约定。改本文件时请同步告知团队。
>
> 本项目为 2026-09-13 重构后版本：**framework 不持有资金/持仓/撮合/PnL/收益概念**；
> 这些业务概念下放到策略 `step` 内部（state 字段 + 策略内联撮合数学）。framework
> 只做 1) 桶预计算 + 2) `step` 驱动循环 + 3) 可选 GPU-batched sweep hook。
> CPU (torch cpu) / GPU (torch cuda) 双端统一由 PyTorch 后端提供。

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
| 行为权威 | `openspec/specs/evtrade-architecture/spec.md` | 20+ 条 Requirement + Scenario；通过 `openspec validate --specs` 校验 |
| 中文详述 | `kbs/*.md`（8 份 + README + archive/DSL-HISTORY） | 与 spec 一一映射；改一处必改另一处 |
| 变更提案 | `openspec/changes/<name>/` | proposal.md + spec delta + design.md + tasks.md |

## 2. KB 索引（8 份 + README + archive）

| 文档 | 内容 |
|---|---|
| `kbs/01-总览.md` | 项目定位、分层架构、核心特性、技术栈、包结构、文档地图 |
| `kbs/02-数据与桶.md` | `Bar` / 桶 dict / `stime` 14 位 / `compute_bucket_general` / `BarAggregator` / 任意 m/h/d |
| `kbs/03-指标-EMA.md` | EMA 三形态（`ema_step` / `ema` / `torch_ema`）+ 双 rail `ema_channel_step` + dataclass state |
| `kbs/04-策略.md` | 三策略（`channel_deviation` / `ma_crossover` / `filtered_mr`）+ 统一契约 + 策略自管执行 |
| `kbs/05-引擎与CLI.md` | Engine / `run_vectorized` 装配 + 桶 CLOSE 语义 + CLI 全表 + 数据加载 + 输出解读 + FAQ |
| `kbs/06-扩展指南.md` | 新增 bar 流 / 策略 / 周期 / 指标 + `VectorizedStrategy` 模板 + 实盘接入 |
| `kbs/07-PyTorch与性能.md` | PyTorch 统一后端 + CPU/GPU 透明路由 + 性能内核 + `batched_step` opt-in hook |
| `kbs/08-选参与鲁棒性.md` | WFO + 邻域衰减 S + 复合 score + 帕累托前沿 + 落盘默认参数 |
| `kbs/archive/DSL-HISTORY.md` | 旧 DSL → 三端转译（numba/CUDA/`state_spec`）**已下线** 归档 |

## 3. 源码地图（核心要点）

- 包结构：`evtrade/{cli,primitives,backends}.py` + `core/{timeutils,aggregator,engine,vectorized_engine,batched_sweep,data,sweep,tsbucket,_harness}.py` + `indicators/{ema}.py` + `strategies/{vectorized_base,channel_deviation,ma_crossover,filtered_mr,_defaults_loader}.py`（`_defaults/` 子目录存落盘默认参数）。
  （2026-09-13 重构：`execution/` 子包、`core/{metrics,replay,permutation,config}.py` 全部下线——framework 不持有资金/持仓/撮合/PnL/收益概念；行情子包已删——数据加载内联 `core/data.py`；`atr/boll/rsi` 指标已删——仅留 EMA；GPU 探测并入 `backends`；旧 `core/gpu.py` 更名 `tsbucket.py`；`batched_sweep.py` 是 GPU-batched sweep 的 opt-in 入口，仅产 sig 不计 cash/position。）
- 策略唯一基类：`VectorizedStrategy` (`evtrade/strategies/vectorized_base.py`)。
  - 唯一抽象方法：`step(self, state, bar, params) -> (state, int)`；`state` 由引擎持有、跨调用持续，策略**无 instance 持久状态**。
  - state 初值：`init_state(self, params) -> state`（无状态策略默认返回 `None`；stateful 策略覆写返 `@dataclass`）。
  - 持久状态：`@dataclass` state（策略自定字段，含 EMA 增量 + FSM + 资金/持仓/账本），**不再**用 `state_spec` AST 白名单或 instance attr。
  - 可选 GPU 加速：`batched_step` classmethod hook（仅产 sig，不计 cash/position）。
- 框架契约：单桶 bar dict = `{"ts","o","h","l","c","v","mark"}`（标量，`step` 接收，`mark=0` 为预热段）；引擎侧另有桶级数组 dict `{"ts","o","h","l","c","v","mark","n_bars"}`（`vectorized_engine._aggregate_buckets` 产出，逐桶喂给 step）。
- CLI 入口：`python -m evtrade {backtest,sweep,params}`（2026-09-13 删 `replay` 子命令）。
  设备参数：`--device {cpu, gpu, auto}`（统一，替代旧 `--engine {kernel, ref, vectorized}`）。
  策略参数：`--params "k1:v1;k2:v2"`（按 `params_spec` 校验；策略资金/持仓/撮合参数如 `init_cash`/`trade_qty`/`buy_pct`/`sell_pct` 都走 `--params`）。
- 指标：`evtrade/indicators/ema.py` 提供增量版 (`ema_step / ema_channel_step`，`@dataclass` state 由策略 `step` 自维护) + numpy 批量参考版 (`ema / ema_channel`，仅离线校验用) + torch 批量版 (`torch_ema`，batched_step 用)；**只留 EMA**，`atr / boll / rsi` 已删。

## 4. OpenSpec 技能包

- 项目 `.claude/skills/` 下的 5 个 skill（explore / propose / apply-change / archive-change / sync-specs）。
- `.claude/commands/opsx/` 下对应 5 个 slash command（`/opsx:explore`、`/opsx:propose`、`/opsx:apply`、`/opsx:archive`、`/opsx:sync`）。
- 常用命令：
  - `openspec validate --specs` — 校验 spec
  - `openspec list --specs` / `openspec list` — 列出 spec / change
  - `openspec show <name>` — 查看详情（spec 或 change）

## 5. 关键约定（写代码时不要破坏）

- **framework 不持有资金/持仓/撮合/PnL/收益概念**（2026-09-13）：
  - `evtrade/execution/` 子包、`core/metrics.py`、`core/replay.py`、`core/permutation.py`、`core/config.py` 已下线；
  - 现金/持仓/账本/撮合/PnL/收益 等业务概念全部在策略 `step` 内部 + 策略 `@dataclass state` 字段 + 策略内联撮合数学（与旧 `trade_decision` 等价）；
  - `run_vectorized` 返回 = `{sig, buckets, final_state}`（`final_state` 是策略 step 末尾 state 透传，framework 不读具体字段）。
- 框架层不假定任何指标字段名（up/dw/low_dev 等均不出现于 framework CLI / vectorized API 的关键字参数）。
- 策略唯一抽象入口是 `step(state, bar, params) -> (state, sig)`；持久状态用 `@dataclass` state（引擎持有跨调用），
  **不要**重新引入 `state_spec` AST 白名单 / `KernelState` jitclass / `self._xxx` instance 持久状态。
- 策略展示由一个 hook 完成：`format_signal_line(ts, sig, info) -> str`（`VectorizedStrategy` 默认仅打 `ts/sig/side`）；
  不再使用 `get_extra_bucket_columns` / `get_extra_signal_columns`（旧 bucket_table / per-bar bundle 已删除）。
- 指标计算在策略 `step` 内部通过 `evtrade.indicators.{ema_step, ema_channel_step}` 自维护（仅 EMA）；
  framework 不再持有任何 EMA 公式或指标字段。
- 信号行打印由策略 `format_signal_line` hook 完成，Engine 只 `print` 不` 不假设 `info` 键集。
- Engine.on_bars 是**桶 CLOSE 语义**：桶切换时 (`cur.ts != last_cur.ts`) 用上一桶 finalized OHLCV 驱动策略一次；
  vectorized 路径在桶级 finalized OHLCV 上计算指标。
- `--device {cpu,gpu,auto}` 是唯一后端选择参数（torch CPU / torch CUDA 统一后端，`backends.get_xp` 路由；
  入口统一 `backends.resolve_device` resolve，`--device gpu` 无 CUDA 抛 ValueError，提示改 `auto/cpu` + GPU 机器请 `uv sync` 后跑 `bash scripts/sync-torch-cu.sh`）。
  旧 `--engine {kernel,ref,vectorized}` 已删除。
- **GPU wheel 安装约定**（2026-09-13 `drop-gpu-extra-cpu-default`，见 spec R `PyTorch 统一后端` + kbs/07 §3.1）：
  默认 `uv sync` 装 pypi **CPU wheel**；`pyproject.toml` **不**声明 `gpu` optional extra，
  **不**为 torch 配 cu128 source/index（否则 `uv lock` 会把 base 与 extra 的 torch 统一塌缩成
  cu128，害无 GPU 的 CPU 机器也被迫下载 GPU wheel）。GPU 协作者 `uv sync` 之后跑
  `bash scripts/sync-torch-cu.sh` 把 venv 内 torch 覆盖到 cu128（与 lockfile 无关，
  接受 `EVT_TORCH_CU_TAG` 覆写 cu tag）。`uv.lock` 不入库（`.gitignore` 忽略），各机器本地生成。

## 6. 验证命令

- KB ↔ 代码一致性（design §5 grep hygiene，三条应 0 命中）：
  - `grep -rn "feeds\|gpu_info\|capability\|BrokerExecutor\|xp_ema\|atr_step\|rsi_step\|boll_step" evtrade/ kbs/ CLAUDE.md openspec/specs/`
  - `grep -rn "compute_bucket\b\|daterange\|PERIODS\|trades_to_list\|build_engine\|print_summary" evtrade/ kbs/ CLAUDE.md openspec/specs/`（`\b` 词边界：`compute_bucket_general` 不算）
  - `grep -rn "--no-sleep\|--step-days\|--show-bars\|--bars-out\|ann_excess_pct" evtrade/ kbs/ CLAUDE.md`
- spec 校验：`openspec validate --specs` 应通过。
- 测试：`uv run pytest -q`（应 60+ passed；含 CPU/GPU 容差、vectorized-vs-Engine 对账、strategy-owned 字段、batched_step、filtered_mr、pyproject 依赖审计）。
- 烟测：`python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device cpu` 应输出信号轨迹 + 策略 final_state 透传（无 PnL/收益/回撤/胜率字段）。