# evtrade-architecture

> **本文档为 evtrade 项目架构的行为权威 spec。**
> **详细中文说明见 `kbs/` 目录（特别是 02、06、11、12、14 号文档）；kbs/ 为本 spec 的中文投影，二者必须同步演进。改一处必改另一处。**
>
> 本文用 OpenSpec `spec-driven` 模式：每条 Requirement 描述可观测行为，Scenario 给定 WHEN/THEN。新增/修改需求走 `openspec/changes/` 下的 change proposal（proposal → specs → design → tasks），落地后 `archive`。

## Purpose

evtrade 是一个多周期 K 线 + 策略回测/扫参框架。本 spec 定义其行为契约：分层、数据流、`step`/`init_state` 策略契约、per-bar dict 契约、策略 hook 协议、CLI 参数协议。

---
## Requirements
### Requirement: Strategy interface contract

策略 MUST 实现 `VectorizedStrategy.step(self, state, bar: dict, params: dict) -> tuple[state, int]`
作为**唯一** framework 调用入口。`bar` 契约 MUST 为 dict
`{"ts", "o", "h", "l", "c", "v", "mark"}`（单桶 OHLCV + 策略期标记，全部标量）；
`mark == 0` 表示预热段，策略 `step` 在该段 MUST 返回 `(state, 0)`。
返回元组 MUST 长度 2：`(new_state, signal)`，`signal` ∈ {-1, 0, 1}（SELL / 无 / BUY）。

策略 MUST NOT 持有 instance-level 持久状态（不允许 `self._fsm / self._up_st / self._dw_st`
等 instance attr）——所有跨桶持久化 MUST 由 `state` 参数承载（推荐 `@dataclass`，dict 亦允许）。
策略 SHOULD 覆写 `VectorizedStrategy.init_state(self, params) -> state` 提供 state 初值
（基类默认返回 `None`，表示无状态策略）。

引擎层（`run_vectorized` 的 `_compute_signals` 与 `Engine.on_bars`）MUST 在循环外持有 state，
每桶一次调用 `strategy.step(state, bar, params)`，state 跨调用持续。策略 `step` MUST 仅做
"算法"（拿 `{ts,h,l,...}` + 上一步 state → 指标增量 → FSM → 返回 `(state, sig)`），MUST NOT 在
step 内出现批量循环（`for i in range(n)`）或后端数据迁移（`hasattr(x, "get").get()` 等）。

`params_spec` 类属性 MUST 声明所有策略参数（含 `default / type / min / max`），framework 在
`__init__` / `_resolve_params` 阶段做校验（填默认 + 类型转换 + 范围检查）；未在 `params_spec`
声明的参数名 MUST 报错（防拼写错误）。策略代码 MUST NOT `import numba` / `import cupy`。

#### Scenario: framework 不预计算指标
- **WHEN** 引擎（任一路径）调 `strategy.step(state, bar, params)`
- **THEN** 传入的 `bar` dict MUST 仅含 `ts/o/h/l/c/v/mark`；MUST NOT 含 `up` / `dw` /
  `low_dev` / `high_dev` 等任何具体指标键

#### Scenario: 策略 step 签名
- **WHEN** 一个策略类继承 `VectorizedStrategy` 并实现 `step(self, state, bar, params)`
- **THEN** framework MUST 按该签名调用；策略 MUST 返回 `(new_state, signal)` 且
  `signal ∈ {-1, 0, 1}`

#### Scenario: 策略无 instance state
- **WHEN** 静态扫描 `evtrade/strategies/*.py`（排除 `vectorized_base.py`）
- **THEN** MUST NOT 出现 `self._fsm` / `self._up_st` / `self._dw_st` / `self._has_prev` 等
  instance-level 持久状态字段；所有持久状态 MUST 由 `step` 的 state 参数承载

### Requirement: Strategy display hooks are framework-agnostic
`VectorizedStrategy` MUST 仅暴露一个可覆写展示 hook：
`format_signal_line(self, ts, sig, info=None) -> str`（hook 签名无 positional 指标；所需字段从
`info` dict 自取，默认实现仅打 `ts / sig / side`）。framework（`Engine` / `run_vectorized`
verbose 路径）SHALL NOT 假定任何指标字段名，仅 `print(strategy.format_signal_line(...))`。
`get_extra_bucket_columns` / `get_extra_signal_columns` 已删除（DSL/bundle 时代产物）。

#### Scenario: Engine 仅 print 策略返回值
- **WHEN** verbose 模式下引擎触发信号行打印
- **THEN** 引擎直接 `print(strategy.format_signal_line(ts, sig, info))`，不修改、不假设
  `info` 键集；默认实现仅显示 `ts / sig / side`

### Requirement: CLI params via `--params` validated by `params_spec`
CLI MUST 接受 `--params "k1:v1;k2:v2"` 一次传入策略参数；`_resolve_strategy_params` 按策略类
`params_spec` 填默认 + 类型/范围校验；未在 `params_spec` 中声明的参数名会报错（防拼写错误）。
策略参数（如 `tf1` / `low1` / `high2`）不再是独立 CLI flag，一律经 `--params` 传入；
`--device {cpu,gpu,auto}` 是唯一的后端选择参数。

#### Scenario: 未声明参数报错
- **WHEN** CLI 传入 `--params "low1:1.5;unknown_x:0.3"`
- **THEN** 启动时报 `ValueError: ChannelDeviationStrategy 收到未声明的参数 ['unknown_x']`
  （`vectorized_base.py::_resolve_params`）

### Requirement: Engine is the sole assembly point
Engine MUST 是唯一装配点：构造时把 `aggregator.on_bars` 覆写为自己的 `on_bars`。
切换回测/实盘只换 Feed + Executor（设计意图），其余组件不变。

#### Scenario: 切换回测/实盘仅换 Feed+Executor
- **WHEN** 把 `MySQLBacktestFeed` 换成 `ChainedFeed(MySQLBacktestFeed(预热段), LiveFeed(实时))`，把 `SimulatedExecutor` 换成 `BrokerExecutor`
- **THEN** Engine / Aggregator / 策略 / Account 无需改动；`ChainedFeed` 保证实盘启动时指标已预热

### Requirement: Indicators are private to strategies (framework MUST NOT compute any indicator)

框架 MUST NOT 预计算任何指标（EMA / ATR / RSI / Bollinger 等）任何符号或字段。所有指标计算由
策略在 `step(state, bar, params)` body 内通过 `evtrade/indicators/` 子包导出的 API 完成。
`evtrade/indicators/` MUST 提供三类算子：
1. **step 增量版**（`@dataclass state` in/out，策略 `step` 逐桶调用）：
   `ema_step` / `ema_channel_step` / `atr_step` / `rsi_step` / `sma_step` / `boll_step`
2. **xp 批量版**（`xp_ema` / `xp_ema_channel` / `xp_atr` / `xp_rsi` / `xp_sma` / `xp_bollinger`，
   兼容旧 numpy/torch 模块签名，engine fast-path 用，策略不直接调）
3. **torch 批量版**（`xp_ema_torch` / `xp_ema_channel_torch`，`(B,T)` tensor 输入输出）+
   **纯 ndarray 版**（`ema` / `atr` / `rsi` / `sma` / `bollinger`，jupyter / 复盘用）

`evtrade/indicators/` MUST NOT `import cupy` / `import numba`，MUST NOT 使用 `@njit`，
MUST NOT 包含 CUDA C99 字符串。`pyproject.toml` MUST NOT 声明 `cupy` / `numba` 为依赖，
MUST 声明 `torch>=2.0`。

#### Scenario: indicators 提供 step 版与 torch 版
- **WHEN** 策略用 `from evtrade.indicators import ema_step` 调用
- **THEN** `ema_step(state, value, p) -> (state, ema)` 为策略 step 用增量接口；
  `xp_ema_torch(c_torch, p)` 为 torch 批量路径用

#### Scenario: 策略 step 用 ema_step 维护增量
- **WHEN** 策略在 `step(state, bar, params)` 内维护 EMA 增量
- **THEN** MUST 用 `ema_step` / `ema_channel_step` 等纯 Python 接口；与批量路径
  `xp_ema` 信号轨迹 MUST bitwise 一致

### Requirement: Code hygiene — unused imports and non-underscore internal helpers MUST NOT accumulate
`evtrade/` 包内的源码文件 MUST NOT 留下未使用的 import 与未加下划线前缀的纯内部辅助函数。"未使用"指该项目内部（含 `tests/`、`scripts/`）无 import / 直接调用 / 字符串反射调用；"纯内部辅助函数"指仅在本文件内部被调用、无项目内外部 caller 的公开名。dev reload 工具（`*_cache` 类）、公共扩展 API（注册表 getter 等）、向后兼容 shim (`evtrade/__init__.py:95-114` 的 `sys.modules.setdefault` 层)、抽象基类的 `NotImplementedError` 占位等 MUST 保留。

#### Scenario: 死 import 已清零
- **WHEN** 静态扫描 `evtrade/**/*.py`（排除 `__pycache__`）中的 import 语句
- **THEN** 所有非 `from __future__ import` / typing / 显式 re-export shim 的 import 必须在文件内或项目内有引用方（`pytest` / `scripts/` 算项目内）

#### Scenario: 内部辅助函数以下划线标注
- **WHEN** 一个公开名（非下划线前缀）的函数 / 类仅在定义文件内部被调用，且项目内无任何外部 caller
- **THEN** 该符号 SHOULD 重命名为下划线前缀（如 `helper` → `_helper`），除非属于 dev reload 工具 / 公共扩展 API / 向后兼容 shim / 抽象基类占位等豁免类别

---

### Requirement: Single contract = VectorizedStrategy.step

策略 MUST 是 `evtrade.strategies.VectorizedStrategy` 的子类（通过 `@register_strategy("name")` 注册），
MUST 实现 `step(self, state, bar, params)` 一个 framework 调用入口。策略 MUST NOT 拥有除
`step` / `init_state` / `format_signal_line` 之外的 framework 调用入口；MUST NOT 出现
`check(cur)` / `_strategy_check` / `state_spec` / `compute_signals` / `compute_signals_for_one_bar`
等已废形态。`params_spec` 类属性 MUST 声明所有策略参数（含 default / type / min / max），
framework 在 `__init__` / `_resolve_params` 阶段做校验（fill default + type cast + range check）。
策略代码 MUST NOT `import numba` / `import cupy`。同一份策略代码 MUST 在 `device="cpu"` 与
`device="gpu"` 下产生 bitwise 一致的信号序列（如有浮点舍入差异则在
`tests/test_strategy_unified.py::test_*_cpu_vs_gpu_*` 锁定）。

#### Scenario: 策略代码一份同时跑 CPU / GPU
- **WHEN** 同一策略分别在 `device="cpu"` 与 `device="gpu"` 下跑同一组数据
- **THEN** 返回的 `sig` 序列 MUST bitwise 一致（`np.array_equal(sig_cpu, sig_gpu)` 为 True）

#### Scenario: vectorized vs Engine 引擎 bitwise 一致
- **WHEN** 同一策略分别走 `run_vectorized`（桶级批量循环 step）与 `Engine.on_bars`（逐 bar
  循环 step）两条路径
- **THEN** 生成的 `trades` list 与终态 `cash / position` MUST 逐笔一致（同 ts / side / qty / price）

#### Scenario: 策略只写一处
- **WHEN** 静态扫描 `evtrade/strategies/*.py`
- **THEN** MUST NOT 出现 `def compute_signals(`、`def check(self,`、`state_spec`、
  `dsl_check`、`@njit`、`DSL docstring` 等已废形态

### Requirement: metrics.summary covers full field shape

`evtrade.core.metrics.summarize` MUST 返回 30 字段（缺一即视为 broken）：

**终态字段**（5）：`final_price / final_cash / final_position / final_equity / baseline`

**交易字段**（5）：`n_trades / n_buy / n_sell / turnover / excess_pct`

**时间字段**（2）：`years / cagr_excess`（注：`ann_excess_pct` 已重命名为 `cagr_excess`，口径改为复合年化与 `cagr` 对齐）

**风险调整字段**（5）：`cagr / sharpe_excess / sortino_excess / calmar / ir`

**回撤字段**（4）：`max_drawdown / max_dd_days / max_dd_recovered / x_mdd`
- `max_drawdown` 单一真源：基于 `equity_curve` 由 metrics 算出，不再依赖 caller
  传入的 `final_state["max_drawdown"]`
- `x_mdd` = 累计超额曲线（equity - baseline）的回撤，单位**小数**；与 `max_drawdown`
  （equity 曲线回撤）不同，**保留**
- `max_dd_recovered` 语义：**trough 到首个恢复 ≥ 前高**的 bar 数；未恢复时 MUST
  返回 `-1`（sentinel，区分"恢复用了 N bar"和"从未恢复"）

**持仓行为字段**（7）：`win_rate / profit_factor / avg_pnl / max_consecutive_wins / max_consecutive_losses / avg_hold_bars / max_hold_bars`

**基准对比字段**（2）：`baseline_max_dd / dd_excess`（`dd_excess = max_drawdown - baseline_max_dd`，正值=策略比基准回撤更深）

`run_vectorized` MUST 累积 `equity_curve`（cash + position * close_to_now）与 `baseline_curve`（init_cash + init_position * close_to_now），末尾调 `metrics.summarize` 输出。`sweep` 评分函数 MUST 不再依赖占位 0.0（所有 Sharpe / Sortino / Calmar / CAGR / max_dd_days / IR 字段由 `metrics.summarize` 实算）。

#### Scenario: vectorized summary 30 字段齐全

字段单位 MUST 统一为下表约定（unify-metrics-units, 2026-09-09；extend-metrics, 2026-09-10 扩展字段集 16 → 25 → 30，含 `x_mdd`）：

| 字段 | 单位 | 说明 |
|---|---|---|
| `cagr` | 百分数 (%) | `(eq[-1]/eq[0])^(1/years) - 1` 后 ×100 |
| `cagr_excess` | 百分数 (%) | `(eq/bl 累计比例)^(1/years) - 1` 后 ×100；与 cagr 同口径 |
| `max_drawdown` | **占当时 peak 的小数** (0.0~1.0+) | `max(0, peak - trough) / peak`；业界惯例 (Tradestation / PT) |
| `x_mdd` | 占初始 baseline 的小数 | 累计超额曲线 (equity - baseline) 的回撤 |
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

CLI 打印 MUST 与字段单位一致：`max_drawdown` / `baseline_max_dd` / `dd_excess` / `x_mdd` / `cagr` / `cagr_excess` / `excess_pct` / `win_rate` MUST 用 `:.2%`；`final_equity` / `baseline` / `final_cash` / `turnover` / `avg_pnl` MUST 用 `:.2f`；`max_dd_days` MUST 用 `:.1f` + " 天" 后缀；`calmar` / `sharpe_excess` / `sortino_excess` / `ir` MUST 用 `:.3f`（无量纲）。CLI 打印行 MUST NOT 出现"金额元"字段被 `:.2%` 格式化的输出。

`ann_excess_pct` 已重命名为 `cagr_excess`（口径改为复合年化）；`x_mdd` **保留**（超额曲线回撤，与 equity 曲线回撤 `max_drawdown` 不同）。

#### Scenario: vectorized summary 字段齐
- **WHEN** `run_vectorized(...)` 返回 `result["summary"]`
- **THEN** `set(summary.keys())` MUST 恰好等于上述全部 30 字段（含 `x_mdd`）；数值 MUST 非 NaN（无数据时填 0.0；profit_factor 无亏损时填 `inf`；max_dd_recovered 未恢复时填 `-1`）

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

#### Scenario: x_mdd 为超额曲线回撤小数

- **WHEN** 调用 `metrics.summarize(...)` 返回 dict
- **THEN** MUST 含 `x_mdd` key，且为小数（`0.0 <= x_mdd <= 2.0`，量级非元；
  `tests/test_metrics_units.py::test_x_mdd_is_fraction` 锁定）

#### Scenario: 持仓周期从 BUY→SELL 配对计算

- **WHEN** trades 序列已知
- **THEN** `avg_hold_bars / max_hold_bars` MUST 基于相邻 BUY/SELL 配对的 (sell_ts - buy_ts) 桶数计算；未配对的开仓/平仓忽略

### Requirement: CLI surface = `--device {cpu, gpu, auto}`

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto}`（默认 `auto`）；
后端为 **torch**（`evtrade.backends.get_xp(device) -> torch.device`），`"gpu"/"cuda"` 同义
均映射到 `torch.device("cuda")`，`"auto"` 优先 cuda（不可用时静默降级 cpu）。
`evtrade.backends.get_xp("gpu")` 在 CUDA 不可用时 MUST fallback 到 cpu 并打 `RuntimeWarning`
（`"auto"` 不告警）。

#### Scenario: 旧 --device gpu 仍接受
- **WHEN** `python -m evtrade backtest --device gpu ...`
- **THEN** 映射为 `torch.device("cuda")`；CUDA 不可用时打 RuntimeWarning 并继续跑 cpu

#### Scenario: PyTorch 后端端到端跑通
- **WHEN** `python -m evtrade backtest --device auto --strategy channel_deviation ...`
- **THEN** MUST 跑通并打印 30 字段盈亏汇总（含 cagr / sharpe_excess / calmar / win_rate /
  profit_factor / baseline_max_dd / dd_excess / x_mdd 等）

### Requirement: PyTorch 统一后端 (NEW 2026-09-10)

`evtrade` MUST 使用 PyTorch 作为唯一 array 后端，CPU/GPU 透明路由由 `tensor.to(device)` 完成。
`evtrade.backends.get_xp(device) -> torch.device` 是统一入口。
`evtrade.core.gpu.py::gpu_info()` MUST 基于 `torch.cuda` API 探测（无 cupy 路径）。
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
| Strategy display hooks | kbs/06 §9, kbs/11 §7, kbs/02 §1 |
| Single contract = VectorizedStrategy.step | kbs/14 §1, kbs/06, kbs/09 |
| CLI params via `--params` | kbs/06 §3, kbs/10, kbs/14 §1 |
| Engine is the sole assembly point | kbs/02 §1-4 |
| Indicators are private to strategies | kbs/05, kbs/11 §5, kbs/12, kbs/14 §2 |
| metrics.summary covers full field shape | kbs/13, kbs/09 |
| Code hygiene (no unused imports, internal helpers underscored) | kbs/01 (源码地图: 本次清理 + 改名) |

修改本 spec 时**必须**同步更新对应 `kbs/` 文档（反之亦然），并在 commit message 中标注。
