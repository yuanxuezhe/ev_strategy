# 2026-09-09: 框架不假定任何指标 — EMA 解耦到 `evtrade/indicators/`

## Why

CLAUDE.md §5 明示"框架层不假定任何指标字段名（up/dw/low_dev 等均不出现于 framework CLI / kernel API 的关键字参数）"，但当前实现在 3 层硬编码了 EMA 通道：

- `core/engine.py::Engine` 显式调用 `self.ema_ch.channel(...)` 算 up/dw 后灌给策略
- `core/kernel.py::KernelState` 6 个 EMA 字段 + `_ema_push` / `_ema_current` helper + `step()` 内联调用
- `core/gpu.py::_CUDA_SOURCE_GENERIC_TEMPLATE` 内联 EMA push/current 与 `tf1s` 数组

CLAUDE.md 同段又承认"换指标需同步改这两处"——项目自己知道这是历史包袱。

## What

1. **新增 `evtrade/indicators/` 增量 API**（`@njit` 兼容、DSL 白名单可调用）：
   - `ema_push` / `ema_current` / `ema_channel_push` / `ema_channel_current`
   - `atr_push` / `atr_current` / `rsi_push` / `rsi_current`
   - `boll_push` / `boll_current` / `sma_push` / `sma_current`
2. **DSL 白名单扩展**（`strategies/dsl.py::_CALL_WHITELIST`）加入上述增量 API
3. **删除 `evtrade/core/incremental_indicators.py`**（无引用方）
4. **Framework 解耦**：`Engine` / `KernelState` / CUDA 模板 / `replay` / `sweep` / `config` / `__init__.py` 全部删除 EMA 耦合
5. **策略 DSL 改造**：`channel_deviation.py` 在 `state_spec` 加 EMA 状态字段，DSL body 调 `ema_channel_push` / `ema_channel_current`
6. **Spec / KB / Tests 同步**

## Impact

- 受影响 capability：`evtrade-architecture`（已有 8 条 Requirement）+ `indicators` 子包（升级为 DSL 可调用库）
- 受影响 files：framework 7 文件 + indicators 5 文件 + strategies 2 文件 + spec/KB/tests 12 文件
- 受影响 public API：删除 `EMAChannel` / `IncrementalEMA` / `ema` / `ema_channel` 顶层导出；新增 `ema_push` / `ema_current` 等
- 不破坏向后兼容（仅删除冻结层 API，调用方需迁移到 `indicators.*_push` / `*_current`）

## Out of Scope

- 不重构 `example_breakout.py` / `example_dev_trigger.py`（用户表态不动）
- 不重写 `incremental_indicators.py` 到 `indicators/ema.py`（直接删除整个文件）
- 不改 `compute_indicators()` 钩子机制（DSL body 直接调 `_push` / `_current` 即够）

## Spec delta

新增 Requirement "Indicators are private to strategies"；改 Requirement "Per-bar dict contract"（删 up/dw 框架键）；改 Requirement "DSL→CUDA projection"（白名单包含 indicators 函数）；改 Requirement "Strategy interface contract"（framework 不预计算指标）。