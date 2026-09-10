# reconcile-spec-kbs-with-code

## Why

两次重构（`strategy-step-only` 把策略入口从 `compute_signals(xp, bars, params)` 收敛到
`step(state, bar, params)` + `init_state(params)`；`pytorch-unified-strategy` 把后端从
cupy/numpy 换成 torch）**代码已落地并通过 147 个测试**，但 spec / CLAUDE.md / kbs / 代码
docstring 都停留在旧契约：

- spec 仍声明策略唯一入口是 `compute_signals`、`state_spec`、DSL→CUDA 投影、
  `kernel.bundle_per_bar` / `bucket_table` / `get_extra_bucket_columns` /
  `get_extra_signal_columns` 等**已物理删除**的模块与 hook；
- spec / CLAUDE.md / 多份 kbs 仍写 `cupy` / `numpy` 后端；
- spec / KB13 仍写 metrics "25 字段、x_mdd 已删除"，与代码实际 30 字段（含 x_mdd）矛盾；
- `cli.py:453` 在 replay `--signals-out` 路径调用已删除的
  `strategy.get_extra_signal_columns(...)`，运行即 AttributeError（活 bug）。

spec 是行为权威，当前它与代码脱节，导致 `openspec validate --specs` 描述的行为与真实
行为不符、新开发者按文档写策略会踩坑。本 change 让 spec / kbs / CLAUDE.md / 代码 docstring
**单向对齐到代码现状**（代码为真源），并修掉那个活 bug。

## What Changes

- **BREAKING（文档层，代码 API 不变）**: spec `evtrade-architecture` 的策略契约 Requirement
  从 `compute_signals(xp, bars, params)` 改写为 `step(state, bar, params) -> (state, sig)`
  + `init_state(params)`，state 由引擎持有、策略无 instance state。
- **删除 spec 中引用已移除模块的死 Requirement**: `state_spec is mandatory`、
  `Per-bar dict contract (bundle_per_bar)`、`bucket_table is framework-only`、
  `DSL→CUDA projection in strategies/dsl.py`，以及 `Strategy display hooks` 里的
  `get_extra_bucket_columns` / `get_extra_signal_columns`（现仅存 `format_signal_line`）。
- **术语对齐 torch**: spec / CLAUDE.md / kbs / `cli.py` / `capability.py` / `backends.py` /
  `evtrade/__init__.py` / `indicators/*` 模块 docstring 里的 `cupy` / `numpy 后端` 统一改为
  `torch`；`xp` 语义从 "numpy/cupy 模块" 明确为 "`torch.device`"。
- **metrics 字段集校正**: spec 从 "25 字段、x_mdd 已删除" 改为 **30 字段（含 x_mdd）**，
  与 `tests/test_metrics_v3.py::test_field_set_exactly_26` 实际锁定的 30 键一致；
  同步 KB13 / KB09。
- **源码地图校正**: CLAUDE.md 源码地图删 `core/kernel.py`（已不存在）、补 `backends.py`；
  kbs/README 模块表同步。
- **活 bug 修复**: `cli.py:453` replay `--signals-out` 不再调用已删除的
  `get_extra_signal_columns`，改为与 backtest 路径一致的 `stime,signal` 输出
  （策略无 per-bar 指标列 hook 了）。
- **清理**: 修正 `sweep.py` / `capability.py` 报错文案里的 "cupy/CUDA" 措辞为 "torch/CUDA"；
  清理 `evtrade/**/__pycache__` 残留的 numba `.nbc/.nbi` 缓存（gitignored，仅本机）。

## Capabilities

### New Capabilities
<!-- 无新增能力 -->

### Modified Capabilities
- `evtrade-architecture`: 策略入口契约改为 `step`/`init_state`；删除引用已移除模块的死
  Requirement；后端术语 cupy→torch；metrics 字段集 25→30（含 x_mdd）。

## Impact

- 受影响 spec：`openspec/specs/evtrade-architecture/spec.md`（apply 阶段由本 change delta 全量
  重写受影响的 8 条 Requirement）
- 受影响文档：`CLAUDE.md`、`kbs/{README,02,05,06,09,10,11,12,13,14,使用说明}.md`、
  `README.md`、`docs/quickstart.md`、`docs/params-workflow.md`
- 受影响代码（docstring / 措辞 / 1 处 bug，无算法改动）：
  `evtrade/cli.py`（docstring + `:453` bug）、`evtrade/core/capability.py`、
  `evtrade/core/sweep.py`（措辞）、`evtrade/backends.py`、`evtrade/__init__.py`、
  `evtrade/indicators/ema.py`（docstring）
- 行为：回测 / 扫参 / 对账信号与成交 **bitwise 不变**（本 change 不改任何指标算法或执行
  逻辑，只改文档与 1 处崩溃路径）；`pytest` 仍应 147 passed。
- 不受影响：`indicators/*` 的 `*_step` / `xp_*` / 纯 ndarray 函数签名与实现；`replay` /
  `sweep` / `permutation` 对外签名；`metrics.summarize` 实现。
