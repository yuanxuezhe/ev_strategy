# Design — reconcile-spec-kbs-with-code

## 1. 真源判定（为什么是"对齐到代码"而不是"改代码对齐 spec"）

本 change 的前提是**代码 = 行为真源**，文档 / spec 单向对齐到代码。依据：

1. `pytest -q` 全绿（147 passed），`tests/test_strategy_unified.py` /
   `tests/test_torch_backend.py` 显式锁定了当前契约：`get_xp` 返回 `torch.device`、
   策略实现 `step` / `init_state`、CPU/GPU 走同一份策略代码。
2. 引擎两条实路径都调用 `step`：
   - `evtrade/core/vectorized_engine.py:174`  `state, s = strategy.step(state, bar, params)`
   - `evtrade/core/engine.py:55`             `self._state, sig_int = self.strategy.step(...)`
   `evtrade/` 全库 grep `compute_signals` = **0 命中**（代码里已无此方法）。
3. 两次重构的 commit（`890c10e` pytorch、以及 `strategy-step-only` change 的代码部分）
   都只动了代码，没走完 spec/kbs 同步 —— 即 spec 是**滞后**方，不是意图方。
4. `strategy-step-only` 的 proposal 原话记录了用户意图："策略作者只写
   `step(state, bar, params)` 一个方法，state 由 engine 持有"。

结论：不存在"代码错了要改回 spec"的情形，全部是"spec/文档没跟上代码"。
因此本 change 不做任何指标算法 / 执行逻辑改动，只改文档 + 1 处崩溃 bug。

## 2. 现状 vs 目标 对照

| 维度 | 现状（代码） | spec/kbs 现描述 | 目标 |
|---|---|---|---|
| 策略唯一入口 | `step(state, bar, params) -> (state, sig)` + `init_state(params)` | `compute_signals(xp, bars, params)` + `compute_signals_for_one_bar` + instance attr | `step`/`init_state` |
| 持久状态 | `@dataclass state`，engine 持有跨调用 | "Python 实例属性 `self._fsm/_up_st`" | dataclass state 由 engine 持有 |
| 后端 | torch（`get_xp -> torch.device`） | cupy / numpy 模块 | torch |
| `xp` 语义 | `torch.device` | "numpy/cupy 模块" | `torch.device` |
| 展示 hook | 仅 `format_signal_line(ts, sig, info)` | + `get_extra_bucket_columns` / `get_extra_signal_columns` | 仅 `format_signal_line` |
| metrics 字段 | 30（含 `x_mdd`） | 25（"x_mdd 已删除"） | 30（含 `x_mdd`） |
| DSL/state_spec | 不存在 | spec 有 `state_spec` / DSL→CUDA Requirement | 删除 |

## 3. spec delta 结构（specs/evtrade-architecture/spec.md）

按 openspec delta 语法，本 delta 只写**变动**的 Requirement，archive 时合并进主 spec：

### MODIFIED（改行为描述，全量粘贴新内容）
1. `Strategy interface contract` — 重写为 `step(state, bar, params) -> (state, sig)` +
   `init_state`；bar dict 契约 `{ts,o,h,l,c,v,mark}`；`mark==0` 段 step 返 `(state,0)`；
   策略 MUST NOT 持 instance state；引擎持有 state 循环调 step。
2. `Single contract = VectorizedStrategy.compute_signals`（主 spec 里就叫这个名）—
   改为 `Single contract = VectorizedStrategy.step`；删除 "MUST NOT 拥有 ... step()" 反向
   条款（step 现在是唯一入口）；bitwise 场景改为 "同一份策略代码 cpu/gpu 下 sig 一致"。
3. `Indicators are private to strategies (framework MUST NOT compute any indicator)` —
   指标三类算子：`*_step`（策略 step 用）/ `xp_*`（engine fast-path，兼容 numpy/cupy/torch 模块）/
   纯 ndarray 版（jupyter 复盘）；`pyproject` MUST 声明 `torch>=2.0`、MUST NOT 声明 `cupy`/`numba`。
4. `Strategy display hooks are framework-agnostic` — 收敛为唯一 hook
   `format_signal_line(ts, sig, info) -> str`；删 `get_extra_bucket_columns` /
   `get_extra_signal_columns`。
5. `metrics.summary covers full field shape` — 字段集 25 → **30**（把 `x_mdd` 归入
   "回撤字段" 类，撤销 "x_mdd 已删除" 表述）；单位表补 `x_mdd` 行。
6. `CLI surface = --device {cpu, gpu, auto}` — 术语 cupy→torch；`get_xp("gpu")` CUDA 不可用
   fallback cpu + RuntimeWarning 保留。
7. `PyTorch 统一后端 (NEW 2026-09-10)` — 已存在且基本正确，微调：删 "cupy 时代遗留" 措辞里
   易误导处，确认 `gpu_info()` 用 torch（若无该函数则删该句）。

### REMOVED（引用已物理删除模块的死条款）
8. `state_spec is mandatory for DSL strategies` — Reason: DSL 三端转译已整体下线
   （`unify-strategy-contract`），`state_spec` AST 白名单无引用方；Migration: 策略持久状态
   用 `step` 的 state 参数（推荐 `@dataclass`）。
9. `Per-bar dict contract (bundle_per_bar)` — Reason: `core/kernel.py` 与 `bundle_per_bar`
   已删除；bar 契约现由 `step(state, bar, params)` 的 `bar` dict 承载。Migration: 见新
   `Strategy interface contract`。
10. `bucket_table is framework-only` — Reason: `kernel.bucket_table` 已删除；信号轨迹输出
    由 `cli --signals-out` 直接写 `stime,signal`（策略已无指标列 hook）。
11. `DSL→CUDA projection lives in strategies/dsl.py` — Reason: `strategies/dsl.py` +
    `_CALL_WHITELIST` 已删除；GPU 加速由 torch 算子透明提供，无 NVRTC 编译。
12. `Requirement: Indicators are private ... 2026-09 tightening`（主 spec 第 136 行的第二条
    同名强化版）— Reason: 与改写后的 `Indicators are private` 合并；其 "state_spec 声明 EMA
    增量字段" 场景已被 step 契约取代。

> 注意：主 spec 里 "Indicators are private" 出现了**两次**（第 105 行通用版 + 第 136 行
> "2026-09 tightening" 强化版）。本 change 保留通用版并改写，删除强化版，避免重复条款。

## 4. 代码改动（最小、无算法变更）

### 4.1 活 bug：`evtrade/cli.py:451-464`（replay `--signals-out`）
现状调 `strategy.get_extra_signal_columns(sig=...)`（方法已删，运行即 AttributeError）。
改为与 backtest `cli.py:256-263` 一致的极简输出：
```python
if args.signals_out:
    with open(args.signals_out, "w", encoding="utf-8-sig") as f:
        f.write("stime,signal\n")
        for i, b in enumerate(bars):
            f.write(f"{b.stime},{int(k['sig'][i])}\n")
    print(f"信号轨迹已保存: {args.signals_out} ({len(bars)} 行)")
```
（`bars` 是 Bar 对象列表，`k["sig"]` 是桶级信号数组，长度对齐。删除多余的
`strategy = get_strategy(...)` 与 `extra_cols` / `extras` 变量。）

### 4.2 措辞 / docstring（cupy→torch，纯注释，无逻辑）
- `evtrade/cli.py:10-12,119` docstring / description：`xp = numpy / cupy` → `torch CPU / torch CUDA`。
- `evtrade/core/capability.py:12-13` `TARGET_CAPS` label、`:32,41,50` docstring + 报错文案
  `cupy/CUDA` → `torch/CUDA`。
- `evtrade/core/sweep.py:194` 报错文案 `cupy/CUDA` → `torch/CUDA`。
- `evtrade/backends.py:51,53` `numpy/list` 措辞保留（描述输入类型，正确）。
- `evtrade/__init__.py:19` docstring `cupy 可选 GPU` → `torch (CPU 或 CUDA wheel)`。
- `evtrade/indicators/ema.py:5,8,93` docstring `cupy` 措辞 → `torch`。
- `scripts/benchmark.py` 顶部 docstring + `cupy` 措辞（若引用已删内核路径，标注为历史/删除）。

## 5. 文档改动（spec 投影，同步演进）

- `CLAUDE.md`：§0/§3 源码地图（删 `kernel.py`、补 `backends.py`）；策略入口 `compute_signals`
  → `step`/`init_state`；"实例属性 state" → "dataclass state 由 engine 持有"；`--device` 术语
  cupy→torch；§6 验证命令里 `grep compute_signals` 改为 `grep "def step"`。
- `kbs/14-策略DSL与三端转译.md`：改名为内容 "统一策略契约"，主入口 `step`，后端 torch，删 DSL 段。
- `kbs/{02,06,09,11}.md`：所有 `compute_signals` / `cupy` / `state_spec` 段落对齐 step + torch。
- `kbs/12-重构与性能内核.md`：包结构图（删 `kernel.py`、补 `backends.py`）；性能段 cupy→torch。
- `kbs/13-绩效评估与鲁棒选参框架.md` + `kbs/09`：metrics 字段 25→30（含 x_mdd）。
- `kbs/README.md` / `kbs/使用说明.md` / 顶层 `README.md` / `docs/quickstart.md` /
  `docs/params-workflow.md`：后端术语 + 依赖安装（cupy wheel → torch）+ 字段集。
- `kbs/10` / `kbs/01` 已在 --tf1 轮次更新，本 change 复查 torch 术语残留。

## 6. 验证（apply 后必须全绿）

1. `uv run pytest -q` → 147 passed（不得减少；signals-out 修复不改任何已测路径）。
2. `npx openspec validate --specs` → 通过（archive 合并后主 spec 无残留死条款）。
3. `npx openspec validate --changes`（或 `openspec list`）→ 本 change 无 error。
4. 一致性 grep（CLAUDE.md §6 更新后）：
   - `grep -rn "compute_signals" kbs/ CLAUDE.md openspec/specs/` → 0 命中（或仅历史 changelog）。
   - `grep -rn "import cupy\|cupy-cuda\|xp = cupy\|xp=cupy" evtrade/ kbs/ CLAUDE.md` → 0 命中。
   - `grep -rn "state_spec\|bundle_per_bar\|bucket_table\|get_extra_bucket_columns\|get_extra_signal_columns" evtrade/ kbs/ CLAUDE.md openspec/specs/` → 0 命中。
   - `grep -rn "def step" evtrade/strategies/` → 每个已注册策略 ≥1 命中。
5. 手动冒烟：`python -m evtrade replay --log bars.csv --signals-out /tmp/s.csv`（若 bars.csv
   可用）不再 AttributeError，输出 `stime,signal`。

## 7. 明确不做（Out of Scope）

- 不改任何指标算法 / EMA 公式 / 执行器成交语义 / metrics 计算逻辑。
- 不改 `indicators/*` 的 `*_step` / `xp_*` / ndarray 函数签名。
- 不重新引入 `compute_signals` / DSL / numba / NVRTC。
- 不重写 `strategy-step-only` 已归档/在途 change 的正文（本 change 独立承载"补完同步"）。
- 不动 `examples/`、`bars.csv`（未纳入 git 的临时产物）。
- 不删 `openspec/changes/archive/**` 里的历史 change（历史留痕，允许保留旧术语）。
