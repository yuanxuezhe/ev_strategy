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
所有策略 MUST 实现 `StrategyBase.check(cur: dict, indicators: dict) -> (signal | None, info: dict)`；
DSL 策略（`compute_signal` 含 DSL docstring）三端（Python ref / numba kernel / CUDA）自动可用。
`ChannelDeviationStrategy` 兼容旧 `(cur, up, dw)` 调用以保证向后兼容。

#### Scenario: DSL 策略三端可用
- **WHEN** 一个 DSL 策略类（`@register_strategy` + `compute_signal` docstring + 合法 `state_spec`）被注册
- **THEN** 它在 Python 参考引擎（`Engine`）、`--engine kernel`（numba 特化内核）、`--device gpu`（CUDA 通用 kernel）三条路径下都可用，且信号轨迹与 Python ref 引擎**逐位（bitwise）一致**（锁定见 `tests/test_kernel_dsl.py`、`tests/test_dsl_cuda.py`）

#### Scenario: 非 DSL 策略仅 Python ref
- **WHEN** 一个策略类不含 DSL docstring（纯 Python 实现）
- **THEN** 它只能在 Python 参考引擎下工作；kernel / CUDA 路径会拒绝并提示"无 DSL docstring"

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
框架/内核/replay 在策略消费 per-bar 数组时 MUST 使用统一 dict 契约：
`{"sig": per-bar 信号轨迹, "per_bar": {"up", "dw", "ts", "o", "h", "l", "c", "v"}}`。
CLI / kernel / replay 的公共接口**不出现** up/dw 等指标键作为函数关键字参数；策略 hook 按需从 `per_bar` 取值。

#### Scenario: kernel 输出与策略消费
- **WHEN** `kernel.run_backtest_trace(...)` 返回 per-bar 数组
- **THEN** 调用方通过 `bundle_per_bar(sig, up, dw, ts_arr, o, h, l, c, v)` 拿到上述 dict；策略 hook `get_extra_bucket_columns(tab, per_bar)` / `get_extra_signal_columns(sig, per_bar)` 按需从 `per_bar["up"]` 等键取值

### Requirement: bucket_table is framework-only (no indicator columns)
`kernel.bucket_table(...)` MUST 仅输出行情 + 信号轨迹列
（`{"ts", "open", "high", "low", "close", "volume", "count", "sig", "n_sig"}`），
**不包含**任何策略专属指标列（EMA 上下轨、偏离百分比等）。
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
DSL→CUDA 投影函数 MUST 位于 `evtrade/strategies/dsl.py`，由以下函数承担：
`render_cuda_device_function`、`build_cuda_state_decls`、
`build_cuda_strategy_check_call`、`build_cuda_device_header`。`evtrade/core/gpu.py` 仅保留 CUDA kernel 源码模板、
编译与调度基础设施，**不持有策略投影逻辑**。
通用模板 `_CUDA_SOURCE_GENERIC_TEMPLATE` 含三个占位符：`{STRATEGY_BODY}`、
`{STATE_DECLS}`（按 `state_spec` 注入寄存器声明）、`{STRATEGY_STATE_ARGS}`
（`strategy_check` 调用处的 state arg 列表）。

#### Scenario: gpu.py 不依赖策略字段名
- **WHEN** 引入新 DSL 策略 P（P 的 `state_spec` 含新字段 X）
- **THEN** 无需修改 `core/gpu.py`；`{STATE_DECLS}` 与 `{STRATEGY_STATE_ARGS}` 在编译期由 `dsl.build_cuda_state_decls(P)` / `build_cuda_strategy_check_call(P)` 自动填充

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

### Requirement: Code hygiene — unused imports and non-underscore internal helpers MUST NOT accumulate
`evtrade/` 包内的源码文件 MUST NOT 留下未使用的 import 与未加下划线前缀的纯内部辅助函数。"未使用"指该项目内部（含 `tests/`、`scripts/`）无 import / 直接调用 / 字符串反射调用；"纯内部辅助函数"指仅在本文件内部被调用、无项目内外部 caller 的公开名。dev reload 工具（`*_cache` 类）、公共扩展 API（注册表 getter 等）、向后兼容 shim (`evtrade/__init__.py:95-114` 的 `sys.modules.setdefault` 层)、抽象基类的 `NotImplementedError` 占位等 MUST 保留。

#### Scenario: 死 import 已清零
- **WHEN** 静态扫描 `evtrade/**/*.py`（排除 `__pycache__`）中的 import 语句
- **THEN** 所有非 `from __future__ import` / typing / 显式 re-export shim 的 import 必须在文件内或项目内有引用方（`pytest` / `scripts/` 算项目内）

#### Scenario: 内部辅助函数以下划线标注
- **WHEN** 一个公开名（非下划线前缀）的函数 / 类仅在定义文件内部被调用，且项目内无任何外部 caller
- **THEN** 该符号 SHOULD 重命名为下划线前缀（如 `helper` → `_helper`），除非属于 dev reload 工具 / 公共扩展 API / 向后兼容 shim / 抽象基类占位等豁免类别

---

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
| Code hygiene (no unused imports, internal helpers underscored) | kbs/01 (源码地图: 本次清理 + 改名) |

修改本 spec 时**必须**同步更新对应 `kbs/` 文档（反之亦然），并在 commit message 中标注。
