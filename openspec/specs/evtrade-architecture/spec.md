# evtrade-architecture

> **本文档为 evtrade 项目架构的行为权威 spec。**
> **详细中文说明见 `kbs/` 目录（特别是 02、06、11、12、14 号文档）；kbs/ 为本 spec 的中文投影，二者必须同步演进。改一处必改另一处。**
>
> 本文用 OpenSpec `spec-driven` 模式：每条 Requirement 描述可观测行为，Scenario 给定 WHEN/THEN。新增/修改需求走 `openspec/changes/` 下的 change proposal（proposal → specs → design → tasks），落地后 `archive`。

## Purpose

evtrade 是一个多周期 K 线 + 策略回测/扫参框架。本 spec 定义其行为契约：分层、数据流、`state_spec` 声明机制、三端投影、per-bar dict 契约、策略 hook 协议、CLI 参数协议。

---
## Requirements
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

### Requirement: state_spec is mandatory for DSL strategies
DSL 策略 MUST 在类上声明 `state_spec`；缺声明编译期抛 `CompileError`。schema 为
`{"<字段名>": {"type": bool | int | float, "default": <标量初值>}}`；
空 dict 合法（无持久状态，DSL body 只能写局部变量）。框架据此自动投影到三端（numba jitclass / CUDA device 函数 / Python ctx），新增持久字段**无需改内核**。

#### Scenario: 缺 state_spec 编译失败
- **WHEN** 一个 DSL 策略类没有 `state_spec` 类属性
- **THEN** 编译/构造阶段抛 `CompileError`（`dsl._require_state_spec`，`strategies/dsl.py`）

#### Scenario: state_spec 字段自动三端投影
- **WHEN** 策略类 `state_spec` 新增字段 X（如 `"counter": {"type": int, "default": 0}`）
- **THEN** `ctx.X`（DSL body）、`KernelState.X`（numba）、device 函数引用 X（CUDA）、Python ctx X 均自动可用，无需修改 kernel.py / gpu.py

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

### Requirement: bucket_table is framework-only (no indicator columns)
`kernel.bucket_table(...)` MUST 仅输出行情 + 信号轨迹列
（`{"ts", "open", "high", "low", "close", "volume", "count", "sig", "n_sig"}`），
**不包含**任何策略专属指标列（EMA 上下轨、ATR、偏离百分比等）。
策略专属指标列由策略 hook `get_extra_bucket_columns` 按需拼接。

#### Scenario: bucket_table 输出不含指标
- **WHEN** 调用 `kernel.bucket_table(stime, sig, ts_out, o_out, h_out, l_out, c_out, v_out)`
- **THEN** 返回 dict 的键集严格等于 `{ts, open, high, low, close, volume, count, sig, n_sig}`，无 up/dw/low_dev/high_dev 等

### Requirement: Strategy display hooks are framework-agnostic
`StrategyBase` MUST 暴露三个可覆写的展示 hook，框架 SHALL NOT 假定任何指标字段名：
- `format_signal_line(cur, signal, info) -> str`（hook 签名无 positional 指标；所需字段从 `info` dict 自取）
- `get_extra_bucket_columns(*, tab, per_bar) -> dict`
- `get_extra_signal_columns(*, sig, per_bar) -> dict`

#### Scenario: Engine 仅 print 策略返回值
- **WHEN** verbose 模式下 Engine 调用信号行打印
- **THEN** Engine 直接 `print(strategy.format_signal_line(cur, signal, info))`，不修改、不假设 `info` 键集；hook 默认实现仅打 OHLCV + 信号

### Requirement: DSL→CUDA projection lives in `strategies/dsl.py`
- DSL 函数调用白名单 `_CALL_WHITELIST` MUST 包含 `min` / `max` / `abs` 以及 `evtrade/indicators/` 子包导出的所有 `*_push` / `*_current` 增量 API
- 三端（Python / numba / CUDA）渲染 MUST 共享同一份白名单
- `core/gpu.py` MUST 仅保留 CUDA kernel 源码模板，不假定任何指标名

#### Scenario: 新增指标仅需 indicators 子包 + 白名单一行
- **WHEN** 在 `evtrade/indicators/` 新增 `momentum.py` 实现 `@njit momentum_push / momentum_current`
- **THEN** `strategies/dsl.py::_CALL_WHITELIST` 加入这两个名字后，三端 (Python / numba / CUDA) MUST 自动支持；framework 不需要任何改动

### Requirement: CLI params via `--params` validated by `params_spec`
CLI MUST 接受 `--params "k1:v1;k2:v2"` 一次传入策略参数；`_resolve_strategy_params` 按策略类
`params_spec` 填默认 + 类型/范围校验；未在 `params_spec` 中声明的参数名会报错（防拼写错误）。
`params_spec` 声明顺序即 DSL 的 `p0..pN` 顺序（numba 端最多 16 个，CUDA 端最多 8 个）。

#### Scenario: 未声明参数报错
- **WHEN** CLI 传入 `--params "low1:1.5;unknown_x:0.3"`
- **THEN** 启动时报 `ValueError: ChannelDeviationStrategy 收到未声明的参数 ['unknown_x']`（`base.py:87-120`）

### Requirement: Engine is the sole assembly point
Engine MUST 是唯一装配点：构造时把 `aggregator.on_bars` 覆写为自己的 `on_bars`。
切换回测/实盘只换 Feed + Executor（设计意图），其余组件不变。

#### Scenario: 切换回测/实盘仅换 Feed+Executor
- **WHEN** 把 `MySQLBacktestFeed` 换成 `ChainedFeed(MySQLBacktestFeed(预热段), LiveFeed(实时))`，把 `SimulatedExecutor` 换成 `BrokerExecutor`
- **THEN** Engine / Aggregator / 策略 / Account 无需改动；`ChainedFeed` 保证实盘启动时指标已预热

### Requirement: Indicators are private to strategies (framework MUST NOT compute any indicator)
框架 MUST NOT 预计算任何指标（EMA / ATR / RSI / Bollinger 等）任何符号或字段。
所有指标计算由 DSL 策略在 body 内通过 `evtrade/indicators/` 子包导出的 API 完成：
- **增量版**（DSL 可调用、numba `@njit` 兼容）：`ema_push` / `ema_current` / `ema_channel_push` / `ema_channel_current` / `atr_push` / `atr_current` / `rsi_push` / `rsi_current` / `boll_push` / `boll_current` / `sma_push` / `sma_current` —— 供 DSL body 在 numba/CUDA 路径调用
- **批量版**（纯函数、复盘 / jupyter 用）：`ema` / `ema_channel` / `atr` / `rsi` / `bollinger` / `sma` / `true_range` —— 供策略 hook `get_extra_signal_columns` 一次性算 per-bar 数组

DSL 函数调用白名单（`strategies/dsl.py::_CALL_WHITELIST`）MUST 包含上述增量版所有名字；三端（Python / numba / CUDA）共享同一份白名单。

`evtrade/core/incremental_indicators.py` 整文件 MUST 被删除（无引用方：`grep` 仅命中 `__init__.py:81` 与 `engine.py:23` 两条导入；删除后顶层 API `EMAChannel` / `IncrementalEMA` / `ema` / `ema_channel` 同时下线）。

#### Scenario: 框架不假定 EMA
- **WHEN** 静态扫描 `evtrade/core/**/*.py`（排除 `__pycache__` 与 `indicators/`）
- **THEN** MUST NOT 出现 `IncrementalEMA` / `EMAChannel` / `_ema_push` / `_ema_current` / `up_sum` / `up_count` / `up_ema` / `dw_sum` / `dw_count` / `dw_ema` / `tf1` / `k_ema` 任何符号

#### Scenario: 策略 DSL body 自维护指标
- **WHEN** 一个 DSL 策略需要 EMA 通道（如 `ChannelDeviationStrategy`）
- **THEN** 策略 MUST 在 `state_spec` 中声明 `up_st` / `dw_st` 增量状态字段，并在 DSL body 第一段调 `ema_channel_push(ctx.up_st, ctx.dw_st, ctx.cur_high, ctx.cur_low, ctx.p_N)` 维护；框架不参与

#### Scenario: 策略 hook 暴露 per-bar 指标
- **WHEN** 策略希望在 `--signals-out` 输出中包含 per-bar 指标（如 EMA 上下轨）
- **THEN** 策略 MUST 在 `get_extra_signal_columns(sig, per_bar)` hook 内调 `evtrade/indicators.ema_channel(per_bar["h"], per_bar["l"], params["tf1"])` 批量算得 `up_arr` / `dw_arr` 并返回 `{"up": up_arr, "dw": dw_arr}`

### Requirement: Code hygiene — unused imports and non-underscore internal helpers MUST NOT accumulate
`evtrade/` 包内的源码文件 MUST NOT 留下未使用的 import 与未加下划线前缀的纯内部辅助函数。"未使用"指该项目内部（含 `tests/`、`scripts/`）无 import / 直接调用 / 字符串反射调用；"纯内部辅助函数"指仅在本文件内部被调用、无项目内外部 caller 的公开名。dev reload 工具（`*_cache` 类）、公共扩展 API（注册表 getter 等）、向后兼容 shim (`evtrade/__init__.py:95-114` 的 `sys.modules.setdefault` 层)、抽象基类的 `NotImplementedError` 占位等 MUST 保留。

#### Scenario: 死 import 已清零
- **WHEN** 静态扫描 `evtrade/**/*.py`（排除 `__pycache__`）中的 import 语句
- **THEN** 所有非 `from __future__ import` / typing / 显式 re-export shim 的 import 必须在文件内或项目内有引用方（`pytest` / `scripts/` 算项目内）

#### Scenario: 内部辅助函数以下划线标注
- **WHEN** 一个公开名（非下划线前缀）的函数 / 类仅在定义文件内部被调用，且项目内无任何外部 caller
- **THEN** 该符号 SHOULD 重命名为下划线前缀（如 `helper` → `_helper`），除非属于 dev reload 工具 / 公共扩展 API / 向后兼容 shim / 抽象基类占位等豁免类别

---

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

## 与 `kbs/` 的对应关系

| 本 spec 节 | `kbs/` 详述 |
|---|---|
| Strategy interface contract | kbs/14 §1, kbs/11 §3 |
| state_spec is mandatory | kbs/12 §2.5, kbs/14 §3, kbs/14 §9, kbs/06 §4 |
| Per-bar dict contract | kbs/12 §2.6, kbs/02 §3 |
| bucket_table is framework-only | kbs/12 §2.6, kbs/06 §9, kbs/11 §7 |
| Strategy display hooks | kbs/06 §9, kbs/11 §7, kbs/02 §1 |
| DSL→CUDA projection in `strategies/dsl.py` | kbs/12 §6, kbs/14 §5 |
| CLI params via `--params` | kbs/06 §3, kbs/10, kbs/14 §1 |
| Engine is the sole assembly point | kbs/02 §1-4 |
| Indicators are private to strategies | kbs/05, kbs/11 §5, kbs/12 §2.5, kbs/14 §2.1+§3 |
| Code hygiene (no unused imports, internal helpers underscored) | kbs/01 (源码地图: 本次清理 + 改名) |

修改本 spec 时**必须**同步更新对应 `kbs/` 文档（反之亦然），并在 commit message 中标注。
