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

策略 MUST 实现 `VectorizedStrategy.compute_signals(self, xp, bars: dict, params: dict) -> xp.ndarray[int8]`，**这是策略唯一入口**。`xp` MUST 为 `numpy` 或 `cupy` 模块（framework 通过 `evtrade.backends.get_xp(device)` 传入）；策略代码 MUST 仅使用 `xp` 提供的数组算子，禁止直接 `import numpy` / `import cupy` 后用具体名称。`bars` 契约 MUST 为 dict `{"ts", "o", "h", "l", "c", "v", "mark", "n_bars"}`（桶聚合后的 OHLCV + 策略期标记）。返回数组 MUST 为 `xp.int8`，长度 MUST 等于 `len(bars["ts"])`；元素 MUST ∈ {-1, 0, 1}（SELL / 无 / BUY）。`mark == 0` 的桶（预热段）策略 SHOULD 输出 0；framework 会再做一次 `sig *= mark` 兜底。framework MUST NOT 在调用 `compute_signals` 之前预计算任何指标（无 `up` / `dw` 等键出现于 bars dict）。策略如需在逐 bar 路径（`Engine.on_bars`）维护 instance 状态（FSM / 计数器等），SHOULD 覆写 `VectorizedStrategy.compute_signals_for_one_bar(self, xp, bar: dict, params: dict) -> int`；基类默认实现把单 bar 包成单元素 xp 数组后调 `compute_signals` 取 `[0]`（**仅对无状态的纯数组算子策略适用**）。

#### Scenario: framework 不预计算指标
- **WHEN** `run_vectorized` 调 `strategy.compute_signals(xp, buckets, params)`
- **THEN** MUST 传入 `buckets` dict 仅含 `ts/o/h/l/c/v/mark/n_bars`；MUST NOT 包含 `up` / `dw` / `low_dev` / `high_dev` 等任何具体指标键

#### Scenario: 策略 compute_signals 新签名
- **WHEN** 一个策略类继承 `VectorizedStrategy` 并实现 `compute_signals(self, xp, bars, params)`
- **THEN** framework MUST 按新签名调用，策略 MUST 在 body 内用 `xp` 数组算子完成所有计算（不再有 DSL docstring、不再有 `state_spec`、不再有 numba `@njit` / CUDA `__device__` 调用）

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

框架 MUST NOT 预计算任何指标（EMA / ATR / RSI / Bollinger 等）任何符号或字段。所有指标计算由策略在 `compute_signals(xp, bars, params)` body 内通过 `evtrade/indicators/` 子包导出的 API 完成。`evtrade/indicators/` 子包 MUST 提供两类函数：批量版（`xp` 兼容，`numpy.ndarray` 或 `cupy.ndarray`，策略主体使用）`xp_ema(xp, close, p)` / `xp_ema_channel(xp, h, l, p)` / `xp_atr` / `xp_true_range` / `xp_rsi` / `xp_sma` / `xp_bollinger` —— 内部走 numpy/cupy 高阶算子；增量版（纯 Python 函数，标量 in/out；供 `compute_signals_for_one_bar` 覆写里逐 bar 增量维护用）`ema_push / ema_current / ema_channel_push / ema_channel_current / atr_push / atr_current / rsi_push / rsi_current / sma_push / sma_current / boll_push / boll_current`。`evtrade/indicators/` 子包 MUST NOT import `numba`、MUST NOT 使用 `@njit` 装饰器、MUST NOT 包含 `CUDA_DEVICE_*` C99 字符串常量。`pyproject.toml` MUST NOT 声明 `numba` 为 required 依赖。

#### Scenario: 框架不假定任何指标名
- **WHEN** 静态扫描 `evtrade/core/**/*.py`（排除 `__pycache__` 与 `indicators/`）
- **THEN** MUST NOT 出现 `IncrementalEMA` / `EMAChannel` / `tf1` / `k_ema` 任何符号

#### Scenario: 策略 compute_signals 用 xp 算子
- **WHEN** 一个策略（如 `ChannelDeviationStrategy`）需要 EMA 通道
- **THEN** 策略 MUST 在 `compute_signals(xp, bars, params)` body 内调 `xp_ema_channel(xp, bars["h"], bars["l"], p)` 或自写 xp 兼容 EMA；CUDA 路径下 `xp.asarray` / `xp.where` / `xp.isfinite` 等高阶算子在 device 上跑

#### Scenario: 策略覆写 compute_signals_for_one_bar 维护 instance FSM
- **WHEN** 一个策略（如 `ChannelDeviationStrategy`）有跨桶持久状态（锁存、计数器等），且要支持 `Engine.on_bars` 逐 bar 路径
- **THEN** 策略 MUST 覆写 `compute_signals_for_one_bar`，在内部维护 instance-level FSM（Python dict / list），调 `ema_push(...)` 增量更新；与 `compute_signals` 批量路径在信号轨迹上 MUST bitwise 一致（同公式同初值）

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

### Requirement: Single contract = `VectorizedStrategy.compute_signals`

策略 MUST 是 `evtrade.strategies.VectorizedStrategy` 的子类（通过 `@register_strategy("name")` 注册）。策略 MUST 实现 `compute_signals(xp, bars, params) -> xp.ndarray[int8]` 一个方法。策略 MUST NOT 拥有除 `compute_signals / format_signal_line / compute_signals_for_one_bar` 之外的 framework 调用入口；不允许有 `check(cur)` / `step()` / `_strategy_check` 等旧 DSL 方法。`params_spec` 类属性 MUST 声明所有策略参数（含 default / type / min / max），framework 在 `__init__` / `_resolve_params` 阶段做校验（fill default + type cast + range check）。策略代码 MUST NOT `import numba` / `import cupy`（除通过 `xp` 参数间接使用）。同一份策略代码 MUST 在 `xp = numpy` 与 `xp = cupy` 下产生 bitwise 一致的信号数组（cupy 高阶算子与 numpy 在数值上 bitwise 一致；如有浮点舍入差异则在 `test_strategy_unified.py::test_cpu_vs_gpu_*` 锁定）。

#### Scenario: 策略代码一份同时跑 CPU / GPU / ref
- **WHEN** 同一份 `ChannelDeviationStrategy` 策略代码分别在 `device="cpu"` 与 `device="gpu"` 下跑同一组数据
- **THEN** 返回的 `sig` ndarray MUST bitwise 一致（`np.array_equal(sig_cpu, sig_gpu)` 为 True）

#### Scenario: vectorized vs ref 引擎 bitwise 一致
- **WHEN** 同一份策略代码分别走 `run_vectorized` 引擎与 `Engine.on_bars` 逐 bar 路径
- **THEN** 生成的 `trades` list 与终态 `cash / position` MUST 逐笔一致（同 ts / side / qty / price）

#### Scenario: 策略只写一处
- **WHEN** 静态扫描 `evtrade/strategies/*.py`
- **THEN** MUST NOT 出现 `def check(self, cur, indicators=None)`、`def _strategy_check(`、`state_spec`、`dsl_check`、`@njit`、`DSL docstring` 等已废 DSL 形态

### Requirement: metrics.summary covers full field shape

`evtrade.core.metrics.summarize` MUST 返回以下字段（缺一即视为 broken）：`final_price / n_trades / n_buy / n_sell / final_cash / final_position / final_equity / baseline / excess / excess_pct / years / ann_excess_pct / cagr / sharpe_excess / sortino_excess / calmar / max_dd_days / max_dd_recovered / x_mdd / max_drawdown / turnover`。`run_vectorized` MUST 累积 `equity_curve`（cash + position * close_to_now）与 `baseline_curve`（期初 init_position * close_to_now + init_cash），末尾调 `metrics.summarize` 输出。`sweep` 评分函数 MUST 不再依赖占位 0.0（所有 Sharpe / Sortino / Calmar / CAGR / max_dd_days 字段由 `metrics.summarize` 实算）。

字段单位 MUST 统一为下表约定（unify-metrics-units, 2026-09-09）：

| 字段 | 单位 | 说明 |
|---|---|---|
| `cagr` | 百分数 (%) | `(eq[-1]/eq[0])^(1/years) - 1` 后 ×100 |
| `max_drawdown` | **占当时 peak 的小数** (0.0~1.0+) | `max(0, peak - trough) / peak`；业界惯例 (Tradestation / PT) |
| `x_mdd` | **累计超额回撤小数** | 累计超额曲线同样的归一化 |
| `sharpe_excess` / `sortino_excess` | 年化（无量纲） | `mean(excess_rets) / std(excess_rets) * sqrt(bars_per_year)` |
| `calmar` | **无量纲**（比率） | `cagr(小数) / max_drawdown(小数)` = `(cagr/100) / max_drawdown` |
| `max_dd_days` | 自然日 | `n_dd_bars / bars_per_day` |
| `max_dd_recovered` | bar 数 | 触底到末尾 bar 数 |
| `final_equity` / `baseline` / `excess` | 金额（元） | `cash + position * last_price` |
| `final_price` | 价格（元） | 最后一根 close |
| `final_cash` / `turnover` | 金额（元） | |
| `final_position` | 股数 | |
| `excess_pct` / `ann_excess_pct` | 百分数 (%) | `(equity - baseline) / baseline * 100` |
| `years` | 年 | `(last_ts - first_ts) / (365.25 * 86400)` |

CLI 打印 MUST 与字段单位一致：`max_drawdown` / `cagr` / `excess_pct` / `ann_excess_pct` MUST 用 `:.2%`；`final_equity` / `baseline` / `excess` / `final_cash` / `turnover` MUST 用 `:.2f`；`max_dd_days` MUST 用 `:.1f` + " 天" 后缀；`calmar` MUST 用 `:.3f`（无量纲）。CLI 打印行 MUST NOT 出现"金额元"字段被 `:.2%` 格式化的输出。

#### Scenario: vectorized summary 字段齐
- **WHEN** `run_vectorized(...)` 返回 `result["summary"]`
- **THEN** dict keys MUST 包含上述字段；数值 MUST 非 NaN（无数据时填 0.0）

#### Scenario: sweep 评分不再依赖占位
- **WHEN** `sweep.run_one_vectorized(...)` 返回 metrics dict
- **THEN** `sharpe_excess / sortino_excess / calmar / cagr / max_dd_days` MUST 由 equity 序列实算（非默认 0.0）

#### Scenario: max_drawdown 单位为当时 peak 的小数
- **WHEN** `run_vectorized(...)` 返回 `result["summary"]["max_drawdown"]`
- **THEN** MUST 满足 `max_drawdown ∈ [0.0, 1.5]`（正常策略 ≤ 1.0；杠杆/极端可超 1.0 但应有界）；MUST NOT 是资金元（量级 ~1e5）
- **AND** MUST 等价于 `max(0, peak_equity - trough_equity) / peak_equity`，其中 `peak_equity = max(equity_curve[:i+1])`

#### Scenario: calmar 跨单位除法已修齐
- **WHEN** 合成曲线 `cagr=10%`、`max_drawdown=0.20`
- **THEN** `summary["calmar"]` MUST ≈ `0.10 / 0.20 = 0.5`（无量纲），MUST NOT 是 `10 / 0.20 = 50.0` 或 `10 / 200000 = 5e-5`

#### Scenario: CLI 打印格式与字段单位一致
- **WHEN** `python -m evtrade backtest ...` 打印盈亏汇总
- **THEN** `最大回撤` 行 MUST 输出合理量级的百分比（如 `-12.34%` 量级），MUST NOT 输出 `1.5e7%`
- **AND** `Calmar` 行 MUST 输出无量纲数字（如 `+0.500`），MUST NOT 输出带 `%` 后缀或与 mdd 量级挂钩

### Requirement: metrics field units are normalized

`metrics.summarize` / `vectorized_engine._execute_trades` / `cli.py` / `sweep.py` MUST 在**单位约定**上保持一致：所有下游消费者（CLI 打印、sweep 过滤、回归测试） MUST 假设上述单位表。任何"金额元"与"百分比"混用 MUST 视为 broken，由 `tests/test_metrics_units.py` 锁定。`sweep --max-mdd` 默认 1.0 MUST 含义为"100% 回撤 = 不限"；实盘建议 `0.15` 现在能真正生效（默认 `1.0` 永远过；`0.15` 拒回撤 > 15% 的策略）。

#### Scenario: sweep filter_pass 跨单位对齐
- **WHEN** `sweep.run(...)` 跑出某组参数 `max_drawdown=0.18` 且 `--max-mdd=0.15`
- **THEN** 该参数 MUST NOT 出现在 `filter_pass=True` 行（MUST 被 18% > 15% 拒掉）

#### Scenario: vectorized summary 字段齐
- **WHEN** `run_vectorized(...)` 返回 `result["summary"]`
- **THEN** dict keys MUST 包含上述字段；数值 MUST 非 NaN（无数据时填 0.0）

#### Scenario: sweep 评分不再依赖占位
- **WHEN** `sweep.run_one_vectorized(...)` 返回 metrics dict
- **THEN** `sharpe_excess / sortino_excess / calmar / cagr / max_dd_days` MUST 由 equity 序列实算（非默认 0.0）

### Requirement: CLI surface = `--device {cpu, gpu, auto}`

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto}`（默认 `auto`）；`auto` 模式下 framework 优先 GPU（cupy 可用时）否则降级 CPU。CLI MUST NOT 暴露 `--engine kernel|ref|vectorized` 三选一选项（已统一为单 `--device` 控制 xp 后端）。兼容期内检测到旧 `--engine` 时 MUST 打 `DeprecationWarning` 并自动转换（`kernel→auto / ref→cpu / vectorized→cpu`），但不永久保留。

#### Scenario: 旧 --engine 自动转换
- **WHEN** `python -m evtrade backtest --engine ref --strategy X ...`
- **THEN** 打 `DeprecationWarning: --engine 已废弃，请用 --device {cpu, gpu, auto}`，参数 `--engine ref` 自动映射为 `--device cpu`，继续执行

#### Scenario: 端到端 CLI 跑通
- **WHEN** `python -m evtrade backtest --strategy channel_deviation --device auto --synthetic-days 30`
- **THEN** MUST 跑通并打印完整盈亏汇总（含 cagr / sharpe_excess / sortino_excess / calmar / max_dd_days）

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
