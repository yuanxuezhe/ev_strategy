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
Engine 消费任意 `stream() -> Iterator[Bar]` 的 bar 流对象（回测用
`core/_harness.ListBarFeed`；实盘接入时提供自己的流实现），切换回测/实盘 = 换 bar 流 +
换 Executor，Engine / Aggregator / 策略 / Account 不变。`evtrade/feeds/` 子包已于 2026-09-10
删除（`MySQLBacktestFeed` 的 SQL 与 `core/data.py` 重复；`ChainedFeed` 无使用方）；
回测数据加载唯一入口 MUST 为 `core/data.load_bars`（MySQL 单次拉取 + npz 缓存）或
`core/data.synthetic_bars`（合成数据）。

#### Scenario: 切换回测/实盘仅换 bar 流 + Executor
- **WHEN** 把 `ListBarFeed` 换成自定义实盘 bar 流，把 `SimulatedExecutor` 换成自定义 Executor
- **THEN** Engine / Aggregator / 策略 / Account 无需改动

#### Scenario: feeds 子包已删除
- **WHEN** 静态检查仓库
- **THEN** `evtrade/feeds/` 目录 MUST 不存在；`evtrade/` 内 MUST NOT 有 `BrokerExecutor` /
  `ChainedFeed` / `get_feed` / `register_feed` 符号

### Requirement: Indicators are private to strategies (framework MUST NOT compute any indicator)

框架 MUST NOT 预计算任何指标。所有指标计算由策略在 `step(state, bar, params)` body 内通过
`evtrade/indicators/` 子包导出的 API 完成。`evtrade/indicators/` 目前 MUST 仅提供 **EMA**
一种指标、两种形态：

1. **step 增量版**（`@dataclass state` in/out，策略 `step` 逐桶调用）：
   `ema_step` / `ema_channel_step`（state：`EMAState` / `EMAChannelState`）
2. **numpy 批量参考版**（`ema` / `ema_channel`，reconcile 参考实现与测试用）

`evtrade/indicators/` MUST NOT 包含 xp/torch 变体或 EMA 之外的指标（`atr/rsi/boll` 已于
2026-09-10 删除；新增指标按上述两形态添加）。`evtrade/indicators/` MUST NOT `import cupy` /
`import numba`。`pyproject.toml` MUST NOT 声明 `cupy` / `numba` 为依赖，MUST 声明 `torch>=2.0`。

#### Scenario: indicators 提供 step 版与 numpy 参考版
- **WHEN** 策略用 `from evtrade.indicators import ema_step` 调用
- **THEN** `ema_step(state, value, p) -> (state, ema)` 为策略 step 用增量接口；
  `ema(values, p) -> ndarray` 为 numpy 批量参考接口

#### Scenario: 策略 step 用 ema_step 维护增量
- **WHEN** 策略在 `step(state, bar, params)` 内维护 EMA 增量
- **THEN** MUST 用 `ema_step` / `ema_channel_step` 纯 Python 接口；与 numpy 批量参考版
  `ema` / `ema_channel` 的信号轨迹 MUST 在 reconcile `bucket_diff_cap` 内一致

#### Scenario: atr/rsi/boll 已删除
- **WHEN** 静态扫描 `evtrade/` 与 `tests/`
- **THEN** MUST NOT 出现 `atr_step` / `rsi_step` / `boll_step` / `sma_step` /
  `xp_ema` / `xp_ema_torch` 等已删符号的 import 或调用

### Requirement: Code hygiene — unused imports and non-underscore internal helpers MUST NOT accumulate

`evtrade/` 包内的源码文件 MUST NOT 留下未使用的 import 与未加下划线前缀的纯内部辅助函数。
"未使用"指该项目内部（含 `tests/`、`examples/`）无 import / 直接调用 / 字符串反射调用。
豁免类别：dev reload 工具（`*_cache` 类）、公共扩展 API（注册表 getter）、抽象基类的
`NotImplementedError` 占位、以及 `evtrade/__init__.py` 的向后兼容 shim 层——该 shim
MUST 收敛为最小集（`sys.modules.setdefault` 仅覆盖仓库内仍使用的旧 import 路径：
`evtrade.data` / `evtrade.metrics` / `evtrade.account` / `evtrade.aggregator` /
`evtrade.replay` / `evtrade.execution` / `evtrade.primitives` 共 7 项）；
`evtrade.kernel` ModuleType stub MUST NOT 存在（2026-09-10 删除）。

#### Scenario: 死 import 已清零
- **WHEN** 静态扫描 `evtrade/**/*.py`（排除 `__pycache__`）中的 import 语句
- **THEN** 所有非 `from __future__ import` / typing / 显式 re-export shim 的 import
  必须在文件内或项目内有引用方

#### Scenario: 内部辅助函数以下划线标注
- **WHEN** 一个公开名（非下划线前缀）的函数 / 类仅在定义文件内部被调用，且项目内无任何外部 caller
- **THEN** 该符号 SHOULD 重命名为下划线前缀，除非属于豁免类别

#### Scenario: shim 最小集
- **WHEN** 静态扫描 `evtrade/__init__.py`
- **THEN** `sys.modules.setdefault` 项 MUST ≤ 7 个且全部有仓库内 import 方；
  MUST NOT 存在 `types.ModuleType("evtrade.kernel")` 或等价 stub

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

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto}`
（默认 `auto`）。设备解析统一由 `evtrade.backends.resolve_device(requested, gpu_available)`
在三个子命令入口执行：`"gpu"` 且 CUDA 不可用 MUST 抛 `ValueError`（带可操作提示）；
`"auto"` 优先 gpu，不可用时降级 cpu 并打 warning（不抛）。`"cpu"` 直接返回。
引擎层（`run_vectorized` 等）MUST NOT 携带 device 参数——后端为 torch 统一，实际计算
路径设备无关（numpy 桶预计算 + 标量 step 循环）。旧 `--engine {kernel,ref,vectorized}`
flag MUST NOT 存在（2026-09-10 删除，传入报 argparse unknown option）。
MUST NOT 存在"被接受但从不读取"的 CLI flag（`--no-sleep` / `--step-days` /
`--show-bars` / `--bars-out` 已删除）。

#### Scenario: --device gpu 无 CUDA 报错
- **WHEN** 无 CUDA 的机器执行 `python -m evtrade backtest --device gpu ...`
- **THEN** MUST 抛 `ValueError`（提示 CUDA 不可用，改用 `--device auto` 或 `--device cpu`）

#### Scenario: --device auto 静默降级
- **WHEN** 无 CUDA 的机器执行 `python -m evtrade sweep --device auto ...`
- **THEN** 降级 cpu 并打 warning，正常运行完成

#### Scenario: --engine 已删除
- **WHEN** 执行 `python -m evtrade backtest --engine kernel ...`
- **THEN** argparse 报 `unrecognized arguments: --engine` 退出

#### Scenario: PyTorch 后端端到端跑通
- **WHEN** `python -m evtrade backtest --device auto --strategy channel_deviation ...`
- **THEN** MUST 跑通并打印 30 字段盈亏汇总（含 cagr / sharpe_excess / calmar / win_rate /
  profit_factor / baseline_max_dd / dd_excess / x_mdd 等）

### Requirement: PyTorch 统一后端

`evtrade` MUST 使用 PyTorch 作为唯一 array 后端能力来源（`pyproject.toml` 声明
`torch>=2.0`）；MUST NOT 存在 `gpu` / `all` 等 `[project.optional-dependencies]`
独立安装路径（torch 在核心依赖，CPU/CUDA 是同一 torch 包的不同运行时 wheel）。
`evtrade.backends.gpu_available()` MUST 委托 `torch.cuda.is_available()`。当前热路径
（桶预计算 + 策略 step）为设备无关 numpy/标量实现，`backends.get_xp` 仅为需要
tensor 的扩展代码提供 `torch.device` 路由。`evtrade/core/capability.py` MUST NOT
存在（能力探测收编至 `backends`）；`evtrade/core/tsbucket.py` MUST 为纯 numpy 桶级
`ts/mark` 预计算（模块名/内容 MUST NOT 含 gpu/cuda/torch 语义依赖，LRU 缓存行为不变）；
历法运算（`encoded_to_epoch` / `epoch_to_encoded`）MUST 单一真源于 `core/timeutils.py`
（标量与向量两形态共享 Hinnant 整数日历）。`gpu_info` 已删除。

#### Scenario: cupy 不再是可选依赖
- **WHEN** 用户执行 `grep -r "import cupy" evtrade/`
- **THEN** MUST 0 命中（cupy 已完全下线）

#### Scenario: gpu 不再是独立安装路径
- **WHEN** 用户执行 `grep -n "optional-dependencies" pyproject.toml`
- **THEN** MUST 0 命中（`[project.optional-dependencies]` 已删除，CPU/GPU 统一走
  `uv sync` / `uv sync --group dev`；`--device` 是运行时参数而非安装路径）

#### Scenario: tsbucket 纯 numpy
- **WHEN** 静态扫描 `evtrade/core/tsbucket.py`
- **THEN** MUST 无 `torch` / `cuda` / `cupy` 引用；`precompute_ts_mark` 输出与重构前
  bitwise 一致（`tests/test_tsbucket_cache.py` 锁定，仅 import 路径变更）

#### Scenario: torch 后端单测通过
- **WHEN** `pytest tests/test_torch_backend.py -v`
- **THEN** `get_xp` 返 `torch.device`；`gpu_available` 与 `torch.cuda.is_available()` bitwise 一致

### Requirement: Single trade-execution implementation

成交决策（取量 + 资金/持仓约束 + 现金/持仓更新）MUST 有唯一实现
`evtrade.execution.base.trade_decision(side, price, cash, position, cur_qty, buy_pct,
sell_pct) -> (new_cash, new_position, qty, filled)`。`SimulatedExecutor.trade` 与
`run_vectorized` 的成交循环（`vectorized_engine._execute_trades`）MUST 均调用它，
MUST NOT 各自复制取量/约束公式。scale（加仓翻倍）与 last_side 状态逻辑属于调用方
状态，MAY 留在各自调用处。`replay --against-ref` MUST 仍逐笔（ts/side/qty/price）
对账两条引擎路径的成交结果。

#### Scenario: 成交公式仅一处
- **WHEN** 静态扫描 `evtrade/`（排除 tests/）中的取量表达式（`cash / price`、
  `buy_pct *`、`sell_pct *` 组合）
- **THEN** MUST 仅出现于 `execution/base.py::trade_decision` 一处

#### Scenario: 双路径逐笔一致
- **WHEN** `replay --log <log> --strategy channel_deviation --device cpu --against-ref`
- **THEN** 输出 `对账 [PASS]`，两条路径 trades 逐笔（ts/side/qty/price）一致且终态
  cash/position 一致

### Requirement: Account is a pure trade ledger

`Account` MUST 只做资金/持仓记账：字段限 `init_cash / init_position / cash /
position / trades`，方法限 `apply(side, qty, price, ts)`。MUST NOT 暴露权益估值
方法（`equity` / `baseline_equity` 等），MUST NOT 持有 `last_price` 类最新价状态
——权益 / 基线曲线与期末估值唯一真源是 `metrics.summarize`（输入 `final_state`
= `{cash, position, last_price, ...}`，由 `vectorized_engine` 自维护）与
`Engine` 腿的 `exec_state` 同口径字段。`Executor` 基类 MUST NOT 提供
`update_price` 钩子（无消费者）；Engine 桶 CLOSE 驱动序列为
`strategy.step → (sig != 0 时) executor.trade`，不含价格预更新步骤。

#### Scenario: 静态扫描无 equity 估值残留
- **WHEN** 静态扫描 `evtrade/`（排除 tests/）
- **THEN** `equity(` / `baseline_equity` / `last_price = price`（Account 侧赋值）/
  `update_price` MUST 均 0 命中；`Account` 实例化后无 `last_price` 属性

#### Scenario: 权益口径不受影响
- **WHEN** `python -m evtrade replay --log <log> --strategy channel_deviation
  --device cpu --against-ref`
- **THEN** 对账 [PASS]，两腿 trades 逐笔一致，`final_equity` / `baseline` /
  `excess_pct` 与删除前完全一致（真源在 `metrics.summarize`）

### Requirement: CLI options declared once

**共享选项 MUST 仅声明一次**：`backtest` / `sweep` / `replay` 三个子命令的共享选项（`--strategy --params --code
--start --end --period --trade-qty --scale --all-in --buy-pct --sell-pct --warmup-days
--device --data-cache --verbose` 等）MUST 声明于单一共享父 parser（argparse `parents=`），
MUST NOT 各子命令重复声明同义选项。CLI 汇总打印 MUST 使用实际 `args` 值
（期初资金/持仓打印 `args.init_cash` / `args.init_position`，MUST NOT 打印模块常量
`INIT_CASH` / `INIT_POSITION`）。

#### Scenario: 共享选项单点声明
- **WHEN** 静态扫描 `evtrade/cli.py` 的 `add_argument` 调用
- **THEN** 上述共享选项名 MUST 仅出现于共享父 parser 定义处一次

#### Scenario: 汇总打印实际值
- **WHEN** `python -m evtrade backtest --init-cash 99999 ...`
- **THEN** 汇总头部"期初资金"行 MUST 打印 99999（非默认常量 100000）

### Requirement: 用户文档单一入口

用户文档 MUST 收敛为两条线：根 `README.md`（唯一入口：定位 / 快速上手 / 工作流
四步 / 命令速查 / 结构 / 指向 `kbs/`）+ `kbs/`（中文详述投影 + `使用说明.md`
操作手册）。仓库根 MUST NOT 存在 `docs/` 目录（原 `docs/quickstart.md` /
`docs/params-workflow.md` 内容已并入 `kbs/使用说明.md` 与 `kbs/10-配置参数与运行指南.md`）。
README 的"详见"清单 MUST 只指向 `kbs/` 内文档（不得指向仓库内不存在的文件）。

#### Scenario: docs/ 目录已删除
- **WHEN** 执行 `ls docs/`
- **THEN** MUST 报 "No such file or directory"

#### Scenario: README quickstart 命令现行
- **WHEN** 按 README「快速上手」的示例命令逐条执行（`python -m evtrade sweep ...` 等）
- **THEN** MUST 无 "unrecognized arguments" / "No such file" 错误

### Requirement: Sweep grid accepts only effective axes

sweep 的 `--grid` 网格轴 MUST 仅接受两类键：引擎轴（`period / trade_qty / scale /
buy_pct / sell_pct`）与策略 `params_spec` 声明的参数名。MUST NOT 接受"下游从不读取"
的轴（2026-09-10 `all_in` 曾是此类——被解析为 bool 后无任何消费者，静默空转；已删）。
资金模式网格化 MUST 经 `--grid buy_pct=...` / `sell_pct=...` 表达。`--grid all_in=...`
MUST 报 `ValueError: 不支持的网格参数 'all_in'; 可用: [...]`。backtest 子命令的
`--all-in` flag 不受影响（等价 `--buy-pct 1.0 --sell-pct 1.0`）。

#### Scenario: all_in 网格轴被拒
- **WHEN** 执行 `python -m evtrade sweep --grid all_in=true,false ...`
- **THEN** 报 `ValueError`（提示不支持的网格参数及可用列表），MUST NOT 静默跑完

#### Scenario: 资金模式经 buy_pct/sell_pct 网格化
- **WHEN** 执行 `python -m evtrade sweep --grid buy_pct=0.5,1.0 --grid sell_pct=0.5,1.0 ...`
- **THEN** 正常展开笛卡尔积并逐组回测（`all_in` 语义 = buy_pct=sell_pct=1.0 的组合）

### Requirement: reconcile legs receive identical effective funding

`replay --against-ref`（`core.replay.reconcile`）MUST 让 vectorized 腿与 Engine 腿收到
**相同的有效成交参数**。`all_in` 的解析（→ `buy_pct = sell_pct = 1.0`）MUST 在
CLI/调用边界对**两腿统一**生效，MUST NOT 只作用于 Engine 腿（`SimulatedExecutor(all_in=)`
）而让 vectorized 腿停留在原始 `buy_pct / sell_pct`——后者会使两腿成交流不可比、
对账必然 FAIL（2026-09-10 前 bug）。

#### Scenario: all-in 对账可 PASS
- **WHEN** `python -m evtrade replay --log <log> --strategy channel_deviation --against-ref --all-in ...`
  且两腿在统一 buy_pct=sell_pct=1.0 下运行
- **THEN** 成交对账（ts/side/qty/price 逐笔）MUST 不因资金模式不对称而 FAIL

### Requirement: signal trajectory output carries aligned timestamps

`--signals-out`（backtest 与 replay 两路径）MUST 写出与信号**逐元素对齐**的桶级
`(ts, sig)` 行。`core.replay.replay_vectorized` MUST 返回与 `"sig"` 等长的桶级时间戳键
`"ts"`（与 `trades` 的 ts 同源，即 `_aggregate_buckets` 产出的桶 ts）；backtest 路径
MUST 用 `result["buckets"]["ts"][mark==1]` 与之 zip。MUST NOT 以 1m bar 数组数索引
**桶级** sig 数组（桶数 < bar 数）——2026-09-10 前 backtest / replay 两路径即以此
方式索引（latent bug：warmup 存在时越界崩溃，无 warmup 时写出大量 sig=0 错位行）。

#### Scenario: replay sig 与 ts 等长且对齐
- **WHEN** `replay_vectorized(...)` 返回 `out`
- **THEN** `len(out["ts"]) == len(out["sig"])`；每对 `(ts, sig)` 中 ts 与该笔
  `trades` 用的桶 ts 同源

#### Scenario: --signals-out 写出桶级轨迹
- **WHEN** `python -m evtrade replay --log <log> --signals-out <out.csv> --warmup-until <ymd> ...`
- **THEN** 正常完成，`<out.csv>` 行数 = 桶数（非 1m bar 数），无越界异常

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
| PyTorch 统一后端 (无独立 GPU 安装路径) | kbs/15 §7.1, kbs/10, 使用说明 §0 |
| 用户文档单一入口 (无 docs/ 目录) | kbs/README, kbs/使用说明 |
| Sweep grid accepts only effective axes (无空转轴) | kbs/10 (sweep 参数), kbs/13 |
| reconcile legs receive identical effective funding | kbs/09, kbs/10 (replay) |
| signal trajectory output carries aligned timestamps | kbs/10 (replay --signals-out) |
| Account is a pure trade ledger (无 equity 估值方法) | kbs/07, kbs/03, kbs/09 |

修改本 spec 时**必须**同步更新对应 `kbs/` 文档（反之亦然），并在 commit message 中标注。
