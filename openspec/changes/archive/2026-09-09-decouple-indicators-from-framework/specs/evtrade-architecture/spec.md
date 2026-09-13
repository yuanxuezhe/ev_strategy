# evtrade-architecture (delta for change "2026-09-09-decouple-indicators-from-framework")

> 本文件为 `openspec/changes/2026-09-09-decouple-indicators-from-framework` 的 spec delta；
> 落地后由 `openspec archive` 同步到主 spec。

## ADDED Requirements

### Requirement: Indicators are private to strategies — 2026-09 tightening
**ADDED 2026-09-09:** 框架 MUST NOT 预计算任何指标（EMA / ATR / RSI / Bollinger 等）任何符号或字段。
所有指标计算由 DSL 策略在 body 内通过 `evtrade/indicators/` 子包导出的 API 完成。

> 注: 主 spec 已有 "Indicators are private to strategies" (上一 change 落地), 本 Requirement
> 是其 2026-09 强化版, 把锁定范围从 EMA 拓到所有指标, 并加 Scenario 锁定 framework 实际扫描条款。

#### Scenario: 框架不假定任何指标名
- **WHEN** 静态扫描 `evtrade/core/**/*.py`（排除 `__pycache__` 与 `indicators/`）
- **THEN** MUST NOT 出现 `IncrementalEMA` / `EMAChannel` / `tf1` / `k_ema` 任何符号（除 `evtrade/core/kernel.py` 内部 `_ema_push` 为 base state 桶切换用之外；该函数不外露，DSL / 策略层不可见）

#### Scenario: 策略 DSL body 自维护指标
- **WHEN** 一个 DSL 策略需要 EMA 通道（如 `ChannelDeviationStrategy`）
- **THEN** 策略 MUST 在 `state_spec` 中声明 EMA 通道增量状态字段，并在 DSL body 第一段调 `ema_channel_push(...)` / `ema_current(...)` 维护；框架不参与

#### Scenario: 策略 hook 暴露 per-bar 指标
- **WHEN** 策略希望在 `--signals-out` 输出中包含 per-bar 指标（如 EMA 上下轨）
- **THEN** 策略 MUST 在 `get_extra_signal_columns(sig, per_bar)` hook 内调 `evtrade/indicators.ema_channel(per_bar["h"], per_bar["l"], params["tf1"])` 批量算得 `up_arr` / `dw_arr` 并返回 `{"up": up_arr, "dw": dw_arr}`

## MODIFIED Requirements

### Requirement: Strategy interface contract
- framework MUST NOT 在调用 `Strategy.check(cur, indicators)` 前预计算任何指标
- `indicators` 参数 MUST 仅为通用扩展参数（保持 `{}` 兼容），不再传入 `up` / `dw` 等具体指标值
- DSL 策略 `check` 签名 MUST 为 `(cur, indicators=None) -> (signal|None, info)`；旧 `(cur, up, dw)` 兼容形式已删除
- 策略自维护指标所需的 state 字段 MUST 通过 `state_spec` 声明并由三端自动投影

#### Scenario: framework 不预计算指标
- **WHEN** `Engine.on_bars` 在 `mark=1` 时调用 `strategy.check(cur, indicators)`
- **THEN** MUST 传入 `indicators=None`；`up` / `dw` 等具体指标 MUST NOT 出现在 positional 或 keyword 参数中

#### Scenario: 策略 check 新签名
- **WHEN** 一个 DSL 策略定义 `check(self, cur, indicators=None)`
- **THEN** framework 调用方 MUST 按新签名调用，策略 MUST 在 body 内自行从 `cur` + `state_spec` 维护指标

### Requirement: Per-bar dict contract (`bundle_per_bar`)
- `bundle_per_bar` 返回的 `per_bar` 子字典 MUST 仅包含 OHLCV + ts 字段：`{"ts", "o", "h", "l", "c", "v"}`
- 框架层 MUST NOT 在 `per_bar` 中包含 `up` / `dw` 等指标键
- 策略如需 per-bar 指标（如 EMA 上下轨），MUST 通过 `get_extra_signal_columns(sig, per_bar)` hook 在策略内部计算并返回

#### Scenario: per_bar 不含指标键
- **WHEN** 调用 `core.kernel.bundle_per_bar(...)` 生成 per-bar 字典
- **THEN** 返回的 `per_bar` MUST 仅含 `ts/o/h/l/c/v`，不含 `up` / `dw` 等指标键

#### Scenario: 策略 hook 暴露 per-bar 指标
- **WHEN** 策略希望 per-bar 信号轨迹携带 EMA 上下轨
- **THEN** 策略 MUST 在 `get_extra_signal_columns` hook 内调 `evtrade.indicators.ema_channel(...)` 算 per-bar up/dw 并返回 dict

### Requirement: DSL→CUDA projection lives in `strategies/dsl.py`
- DSL 函数调用白名单 `_CALL_WHITELIST` MUST 包含 `min` / `max` / `abs` 以及 `evtrade/indicators/` 子包导出的所有 `*_push` / `*_current` 增量 API
- 三端（Python / numba / CUDA）渲染 MUST 共享同一份白名单
- `core/gpu.py` MUST 仅保留 CUDA kernel 源码模板，不假定任何指标名

#### Scenario: 新增指标仅需 indicators 子包 + 白名单一行
- **WHEN** 在 `evtrade/indicators/` 新增 `momentum.py` 实现 `@njit momentum_push / momentum_current`
- **THEN** `strategies/dsl.py::_CALL_WHITELIST` 加入这两个名字后，三端 (Python / numba / CUDA) MUST 自动支持；framework 不需要任何改动