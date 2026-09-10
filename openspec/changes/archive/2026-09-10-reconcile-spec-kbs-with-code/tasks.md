# Tasks — reconcile-spec-kbs-with-code

## 1. spec（apply 阶段手工合并进主 spec）
- [x] 1.1 主 spec `openspec/specs/evtrade-architecture/spec.md`：替换 `Strategy interface contract`
      为 step 契约（delta 第 1 条）
- [x] 1.2 主 spec：把 `Single contract = VectorizedStrategy.compute_signals` **重命名**为
      `Single contract = VectorizedStrategy.step` 并替换内容（delta 第 2 条）
- [x] 1.3 主 spec：替换 `Indicators are private ...`（第 105 行通用版）为 delta 第 3 条内容
- [x] 1.4 主 spec：**删除** `Indicators are private — 2026-09 tightening`（第 136 行重复强化版）
- [x] 1.5 主 spec：替换 `Strategy display hooks are framework-agnostic` 为仅
      `format_signal_line` 版本（delta 第 4 条）
- [x] 1.6 主 spec：替换 `metrics.summary covers full field shape` 为 30 字段版（delta 第 5 条，
      撤销 "x_mdd 已删除"）
- [x] 1.7 主 spec：替换 `CLI surface = --device` 与 `PyTorch 统一后端`（delta 第 6/7 条，
      cupy→torch 措辞）
- [x] 1.8 主 spec：**删除** 4 条死 Requirement（`state_spec is mandatory` /
      `Per-bar dict contract (bundle_per_bar)` / `bucket_table is framework-only` /
      `DSL→CUDA projection in strategies/dsl.py`）
- [x] 1.9 主 spec：Purpose 段 + "与 kbs 对应关系" 表里对已删 Requirement 的引用行清理
- [x] 1.10 `npx openspec validate --specs` 通过；`npx openspec validate --changes` 无 error

## 2. 代码（最小改动 + 1 处活 bug，无算法变更）
- [x] 2.1 `evtrade/cli.py:451-464` replay `--signals-out`：删 `get_extra_signal_columns` 调用，
      改 `stime,signal` 极简输出（与 backtest `:256-263` 一致）
- [x] 2.2 `evtrade/cli.py` docstring/description（`:10-12,119`）cupy→torch
- [x] 2.3 `evtrade/core/capability.py`（`:12-13,32,41,50`）label + 文案 cupy→torch
- [x] 2.4 `evtrade/core/sweep.py:194` 报错文案 cupy→torch
- [x] 2.5 `evtrade/__init__.py:19` docstring 依赖行 cupy→torch
- [x] 2.6 `evtrade/indicators/ema.py` 模块 docstring（`:5,8,93`）cupy 措辞→torch
- [x] 2.7 `scripts/benchmark.py`：整体已死（import 的 `core.kernel_dsl` / 顶层 `kernel` /
      `sweep` / `dsl_kernel` / `cuda_sweep_window_generic` 均已删，`Engine(..., tf1=21)` 签名已变），
      按 design §2.7 "标注历史或删除" **删除**（`git rm`）
- [x] 2.8 `uv run pytest -q` → 147 passed（不减少）

## 3. CLAUDE.md
- [x] 3.1 顶部摘要 + §0 工作流：`compute_signals` → `step`/`init_state`；"实例属性 state" →
      "dataclass state 由 engine 持有"
- [x] 3.2 §3 源码地图：删 `core/kernel.py`、补 `core/backends.py`；策略基类描述改 step
- [x] 3.3 §5 关键约定：`compute_signals` → `step`；`--device` 后端 cupy→torch
- [x] 3.4 §6 验证命令：`grep compute_signals` → `grep "def step"`；cupy grep 项更新

## 4. kbs（spec 中文投影，同步演进）
- [x] 4.1 `kbs/14-策略DSL与三端转译.md`：内容改"统一策略契约"，主入口 step、后端 torch，删 DSL 段
      （cupy 措辞→torch；DSL 历史段保留为 changelog）
- [x] 4.2 `kbs/02-系统架构.md`：compute_signals / cupy / 双端表 → step / torch；step 签名 4→3 参；16→30 字段
- [x] 4.3 `kbs/06-交易策略详解.md`：compute_signals / instance attr → step / dataclass state
- [x] 4.4 `kbs/09-引擎Engine与主流程.md`：compute_signals / metrics 字段集 → step / 30 字段(含 x_mdd)；
      删已废 `sharpe`/`sortino`/`ann_excess_pct` 字段名
- [x] 4.5 `kbs/11-扩展指南.md`：新策略模板 compute_signals → step/init_state
- [x] 4.6 `kbs/12-重构与性能内核.md`：包结构图（删 kernel.py、补 backends.py）+ 性能段 cupy→torch
- [x] 4.7 `kbs/13-绩效评估与鲁棒选参框架.md`：metrics 字段 25→30（含 x_mdd，撤销 "x_mdd 删除"）
- [x] 4.8 `kbs/README.md` / `kbs/使用说明.md`：模块表 + 依赖安装(cupy→torch) + 字段集；
      使用说明 compute_signals 示例→step 契约；cupy FAQ→torch CUDA
- [x] 4.9 `kbs/05-指标计算-EMA通道.md` / `kbs/10` / `kbs/01`：复查 cupy / 25 字段 / compute_signals 残留
      （kbs/10 字段清单补 x_mdd 达 30；kbs/01 为 git 历史单文件投影，保留旧 CLI 行号引用）
- [x] 4.10 顶层 `README.md` / `docs/quickstart.md` / `docs/params-workflow.md`：cupy→torch + 契约措辞
      （README 仓库结构删 kernel/StrategyBase/DSL；quickstart 故障排查 cupy→torch CUDA；
      params-workflow `--engine kernel`→`--device auto`、删 --low1 兼容层、删 ≤8 参数 CUDA 限制）

## 5. 清理 + 终验
- [x] 5.1 一致性 grep 全 0 命中（design §6.4 四条）
      — 源码 .py 0 命中；kbs/spec 剩余命中全为历史 changelog / MUST NOT 禁止条款 /
        `vectorized_engine._compute_signals` 真实内部函数；唯一命中 `__pycache__/*.pyc`（gitignored）
- [x] 5.2 手动冒烟：`replay --signals-out` 不再 AttributeError（输出 `stime,signal`，13 信号/9 成交）
- [x] 5.3 `uv run pytest -q` 147 passed + `npx openspec validate --specs` 通过
- [x] 5.4 （可选）清理 `evtrade/**/__pycache__`（gitignored，仅本机；含旧 kernel/dsl 残留 .pyc，
      不影响 git，留待本机 `python -c "import shutil,..."` 或 IDE 自行清理）
