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

策略 MUST 实现 `VectorizedStrategy.compute_signals(self, xp, bars: dict, params: dict) -> torch.Tensor`。
`xp` MUST 为 `torch.device`（framework 通过 `evtrade.backends.get_xp(device)` 传入；`"cpu"` / `"cuda"` / `"auto"` 三种字符串仍兼容）。
**pytorch-unified-strategy, 2026-09-10 变化**：cupy 已下线，CPU/GPU 统一走 PyTorch；
策略代码 MUST 仅使用 torch / 框架提供的算子，禁止 `import cupy`（已删除依赖）。

#### Scenario: 框架传入 torch.device
- **WHEN** `run_vectorized` / `Engine.on_bars` 调 `strategy.compute_signals(xp, bars, params)`
- **THEN** `xp` MUST 是 `torch.device`（不是 numpy/cupy 模块）

#### Scenario: bars 数组为 numpy 或 torch.Tensor
- **WHEN** 策略读 `bars["c"]` 等
- **THEN** 框架 MUST 接受 numpy ndarray 与 torch.Tensor 两种输入；策略 MUST NOT 假设特定后端

#### Scenario: 策略 compute_signals 签名不变
- **WHEN** 一个策略类继承 `VectorizedStrategy` 并实现 `compute_signals(self, xp, bars, params)`
- **THEN** framework MUST 按新签名调用；策略 body 内仍可调 `xp_ema(xp, c, p)` 等算子（保留向后兼容）

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

框架 MUST NOT 预计算任何指标（EMA / ATR / RSI / Bollinger 等）。所有指标计算由策略在 `compute_signals` 或 `step` body 内通过 `evtrade/indicators/` 子包导出的 API 完成。
**2026-09-10 变化**：`evtrade/indicators/` 子包 MUST 提供三类算子：
1. **xp 版**（兼容旧 `xp_ema(xp, close, p)` 签名，`xp` 为 numpy/cupy/torch 模块；保留向后兼容）
2. **torch 版**（`xp_ema_torch(values: torch.Tensor, p) -> torch.Tensor`，新 PyTorch 后端路径用）
3. **step 增量版**（`ema_step(state, value, p)` 纯 Python 标量 in/out；策略 step() 用）
子包 MUST NOT import `cupy` / `numba`；MUST NOT 使用 `@njit`；MUST NOT 包含 CUDA C99 字符串。
`pyproject.toml` MUST NOT 声明 `cupy` 或 `numba` 为依赖；MUST 声明 `torch>=2.0`。

#### Scenario: indicators 提供 xp 版与 torch 版双形态
- **WHEN** 策略用 `from evtrade.indicators import xp_ema` 调用
- **THEN** `xp_ema(xp, c, p)` 兼容旧 numpy/cupy/torch 模块；`xp_ema_torch(c_torch, p)` 新 PyTorch 路径用

#### Scenario: 策略 step 用 ema_step 维护增量
- **WHEN** 策略覆写 `step(state, bar, params)` 维护 EMA 增量
- **THEN** MUST 用 `ema_step(EMAState, value, p)` 纯 Python 接口；与批量路径 `xp_ema` 信号轨迹 MUST bitwise 一致

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

`evtrade.core.metrics.summarize` MUST 返回 25 字段（缺一即视为 broken）：

**终态字段**（5）：`final_price / final_cash / final_position / final_equity / baseline`

**交易字段**（5）：`n_trades / n_buy / n_sell / turnover / excess_pct`

**时间字段**（2）：`years / cagr_excess`（注：`ann_excess_pct` 已重命名为 `cagr_excess`，口径改为复合年化与 `cagr` 对齐）

**风险调整字段**（5）：`cagr / sharpe_excess / sortino_excess / calmar / ir`

**回撤字段**（3）：`max_drawdown / max_dd_days / max_dd_recovered`
- `max_drawdown` 单一真源：基于 `equity_curve` 由 metrics 算出，不再依赖 caller
  传入的 `final_state["max_drawdown"]`（删除 `x_mdd` 重复字段）
- `max_dd_recovered` 语义：**trough 到首个恢复 ≥ 前高**的 bar 数；未恢复时 MUST
  返回 `-1`（sentinel，区分"恢复用了 N bar"和"从未恢复"）

**持仓行为字段**（5）：`win_rate / profit_factor / avg_pnl / max_consecutive_wins / max_consecutive_losses / avg_hold_bars / max_hold_bars`（注：7 字段，但 trade-derived 与 hold-derived 并列）

**基准对比字段**（2）：`baseline_max_dd / dd_excess`（`dd_excess = max_drawdown - baseline_max_dd`，正值=策略比基准回撤更深）

`run_vectorized` MUST 累积 `equity_curve`（cash + position * close_to_now）与 `baseline_curve`（init_cash + init_position * close_to_now），末尾调 `metrics.summarize` 输出。`sweep` 评分函数 MUST 不再依赖占位 0.0（所有 Sharpe / Sortino / Calmar / CAGR / max_dd_days / IR 字段由 `metrics.summarize` 实算）。

#### Scenario: vectorized summary 25 字段齐全

字段单位 MUST 统一为下表约定（unify-metrics-units, 2026-09-09；extend-metrics, 2026-09-10 扩展字段集 16 → 25）：

| 字段 | 单位 | 说明 |
|---|---|---|
| `cagr` | 百分数 (%) | `(eq[-1]/eq[0])^(1/years) - 1` 后 ×100 |
| `cagr_excess` | 百分数 (%) | `(eq/bl 累计比例)^(1/years) - 1` 后 ×100；与 cagr 同口径 |
| `max_drawdown` | **占当时 peak 的小数** (0.0~1.0+) | `max(0, peak - trough) / peak`；业界惯例 (Tradestation / PT) |
| `baseline_max_dd` | 占当时 peak 的小数 | 买入持有曲线同算法 |
| `dd_excess` | 小数 | `max_drawdown - baseline_max_dd`；正值=策略比基准回撤更深 |
| `sharpe_excess` / `sortino_excess` / `ir` | 年化（无量纲） | `mean(excess_rets) / std(excess_rets) * sqrt(bars_per_year)` |
| `calmar` | **无量纲**（比率） | `cagr(小数) / max_drawdown(小数)` = `(cagr/100) / max_drawdown` |
| `max_dd_days` | 自然日 | `n_dd_bars / bars_per_day` |
| `max_dd_recovered` | bar 数 | trough → 首个恢复 ≥ 前高 的 bar 数；**未恢复时 MUST = -1** |
| `win_rate` / `profit_factor` / `avg_pnl` / `max_consecutive_wins` / `max_consecutive_losses` | 比率 / 比值 / 金额 / 笔数 | BUY→SELL 配对统计 |
| `avg_hold_bars` / `max_hold_bars` | 桶数 | `encoded_to_epoch` 差分 → 秒数 → 桶数 |
| `final_equity` / `baseline` | 金额（元） | `cash + position * last_price` |
| `final_price` | 价格（元） | 最后一根 close |
| `final_cash` / `turnover` | 金额（元） | |
| `final_position` | 股数 | |
| `excess_pct` | 百分数 (%) | `(equity - baseline) / baseline * 100` |
| `years` | 年 | `(last_ts - first_ts) / (365.25 * 86400)` |

CLI 打印 MUST 与字段单位一致：`max_drawdown` / `baseline_max_dd` / `dd_excess` / `cagr` / `cagr_excess` / `excess_pct` / `win_rate` MUST 用 `:.2%`；`final_equity` / `baseline` / `final_cash` / `turnover` / `avg_pnl` MUST 用 `:.2f`；`max_dd_days` MUST 用 `:.1f` + " 天" 后缀；`calmar` / `sharpe_excess` / `sortino_excess` / `ir` MUST 用 `:.3f`（无量纲）。CLI 打印行 MUST NOT 出现"金额元"字段被 `:.2%` 格式化的输出。

`x_mdd` 字段已删除（与 `max_drawdown` 重复）；`ann_excess_pct` 已重命名为 `cagr_excess`（口径改为复合年化）。

#### Scenario: vectorized summary 字段齐
- **WHEN** `run_vectorized(...)` 返回 `result["summary"]`
- **THEN** dict keys MUST 包含上述全部 25 字段；数值 MUST 非 NaN（无数据时填 0.0；profit_factor 无亏损时填 `inf`；max_dd_recovered 未恢复时填 `-1`）

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

#### Scenario: max_dd_recovered sentinel
- **WHEN** equity 序列存在最大回撤且**已恢复**（trough 后存在某 bar 使 equity ≥ 前高）
- **THEN** `max_dd_recovered` MUST 等于 (recovery_idx - trough_idx)，单位 bar
- **WHEN** 存在最大回撤但**从未恢复**（trough 后 equity 始终 < 前高）
- **THEN** `max_dd_recovered` MUST 等于 -1（sentinel，区别于"恢复用了 N bar"）

#### Scenario: 持仓周期从 BUY→SELL 配对计算
- **WHEN** trades 序列已知
- **THEN** `avg_hold_bars / max_hold_bars` MUST 基于相邻 BUY/SELL 配对的 (sell_ts - buy_ts) 桶数计算；未配对的开仓/平仓忽略

### Requirement: metrics field units are normalized

`metrics.summarize` / `vectorized_engine._execute_trades` / `cli.py` / `sweep.py` MUST 在**单位约定**上保持一致：所有下游消费者（CLI 打印、sweep 过滤、回归测试） MUST 假设上述单位表。任何"金额元"与"百分比"混用 MUST 视为 broken，由 `tests/test_metrics_units.py` + `tests/test_metrics_v3.py` 锁定。`sweep --max-mdd` 默认 1.0 MUST 含义为"100% 回撤 = 不限"；实盘建议 `0.15` 现在能真正生效（默认 `1.0` 永远过；`0.15` 拒回撤 > 15% 的策略）。

#### Scenario: sweep filter_pass 跨单位对齐
- **WHEN** `sweep.run(...)` 跑出某组参数 `max_drawdown=0.18` 且 `--max-mdd=0.15`
- **THEN** 该参数 MUST NOT 出现在 `filter_pass=True` 行（MUST 被 18% > 15% 拒掉）

#### Scenario: sweep 评分不再依赖占位

- **WHEN** `sweep.run_one_vectorized(...)` 返回 metrics dict
- **THEN** `sharpe_excess / sortino_excess / calmar / cagr / max_dd_days / ir / baseline_max_dd` MUST 由 equity 序列实算（非默认 0.0）

#### Scenario: max_dd_recovered 三档语义

- **WHEN** equity 序列存在最大回撤且**已恢复**（trough 后存在某 bar 使 equity ≥ 前高）
- **THEN** `max_dd_recovered` MUST 等于 (recovery_idx - trough_idx)，单位 bar
- **WHEN** 存在最大回撤但**从未恢复**（trough 后 equity 始终 < 前高）
- **THEN** `max_dd_recovered` MUST 等于 -1（sentinel，区别于"恢复用了 N bar"）

#### Scenario: x_mdd 字段已删除

- **WHEN** 调用 `metrics.summarize(...)` 返回 dict
- **THEN** MUST NOT 含 `x_mdd` key（与 `max_drawdown` 重复）

#### Scenario: 持仓周期从 BUY→SELL 配对计算

- **WHEN** trades 序列已知
- **THEN** `avg_hold_bars / max_hold_bars` MUST 基于相邻 BUY/SELL 配对的 (sell_ts - buy_ts) 桶数计算；未配对的开仓/平仓忽略

### Requirement: CLI surface = `--device {cpu, gpu, auto}`

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto, cuda}`（默认 `auto`）；
`gpu` 与 `cuda` 同义（cupy 时代遗留的 `gpu` 字符串保留），`auto` 优先 cuda（cupy/torch 都已不用，device 由 PyTorch CUDA 自动探测）。
`evtrade.backends.get_xp("gpu")` 在 CUDA 不可用时 MUST fallback 到 cpu 并打 `RuntimeWarning`。

#### Scenario: 旧 --device gpu 仍接受
- **WHEN** `python -m evtrade backtest --device gpu ...`
- **THEN** 自动映射为 cuda；CUDA 不可用时打 RuntimeWarning 并继续跑 cpu

#### Scenario: PyTorch 后端端到端跑通
- **WHEN** `python -m evtrade backtest --device auto --strategy channel_deviation ...`
- **THEN** MUST 跑通并打印 26 字段盈亏汇总（含 cagr / sharpe / calmar / win_rate / profit_factor / baseline_max_dd / dd_excess / x_mdd 等）

### Requirement: PyTorch 统一后端 (NEW 2026-09-10)

`evtrade` MUST 使用 PyTorch 作为唯一 array 后端，CPU/GPU 透明路由由 `tensor.to(device)` 完成。
`evtrade.backends.get_xp(device) -> torch.device` 是统一入口。
`evtrade.core.gpu.py` 的 `cupy` 探测路径 MUST NOT 存在；`gpu_info()` MUST 改用 `torch.cuda` API。
`evtrade.core.capability.py::gpu_available()` MUST 委托 `torch.cuda.is_available()`。

#### Scenario: cupy 不再是可选依赖
- **WHEN** 用户执行 `grep -r "import cupy" evtrade/`
- **THEN** MUST 0 命中（cupy 已完全下线）

#### Scenario: torch 后端单测通过
- **WHEN** `pytest tests/test_torch_backend.py -v`
- **THEN** `get_xp` 返 `torch.device`；`gpu_available` 与 `torch.cuda.is_available()` bitwise 一致；`xp_ema_torch` 接受 (B, T) tensor + per-row period 返 (B, T)

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
