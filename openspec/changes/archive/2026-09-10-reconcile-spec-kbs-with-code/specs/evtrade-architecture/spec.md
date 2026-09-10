# evtrade-architecture (delta)

> 本 delta 由 `reconcile-spec-kbs-with-code` change 承载：把 spec 从滞后的
> `compute_signals` / cupy / DSL / 25 字段 描述，对齐到代码真源
> （`step(state,bar,params)` + `init_state` / torch / 无 DSL / 30 字段）。
> apply 阶段由 change 作者手工把下列 MODIFIED/REMOVED 合并进
> `openspec/specs/evtrade-architecture/spec.md` 并 `openspec validate --specs` 通过。

## MODIFIED Requirements

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

### Requirement: Single contract = VectorizedStrategy.step

策略 MUST 是 `evtrade.strategies.VectorizedStrategy` 的子类（通过
`@register_strategy("name")` 注册），MUST 实现 `step(self, state, bar, params)` 一个
framework 调用入口。策略 MUST NOT 拥有除 `step` / `init_state` / `format_signal_line`
之外的 framework 调用入口；MUST NOT 出现 `check(cur)` / `_strategy_check` / `state_spec` /
`compute_signals` / `compute_signals_for_one_bar` 等已废形态。
同一份策略代码 MUST 在 `device="cpu"` 与 `device="gpu"` 下产生 bitwise 一致的信号序列
（`tests/test_strategy_unified.py::test_*_cpu_vs_gpu_*` 锁定）。

#### Scenario: 策略代码一份同时跑 CPU / GPU
- **WHEN** 同一策略分别在 `device="cpu"` 与 `device="gpu"` 下跑同一组数据
- **THEN** 返回的 `sig` 序列 MUST bitwise 一致（`np.array_equal(sig_cpu, sig_gpu)`）

#### Scenario: vectorized vs Engine 引擎 bitwise 一致
- **WHEN** 同一策略分别走 `run_vectorized`（桶级批量循环 step）与 `Engine.on_bars`（逐 bar
  循环 step）两条路径
- **THEN** 生成的 `trades` list 与终态 `cash / position` MUST 逐笔一致（同 ts / side / qty / price）
  （`tests/test_strategy_unified.py::test_vectorized_vs_engine_on_bars_reconcile` 锁定）

#### Scenario: 策略只写一处
- **WHEN** 静态扫描 `evtrade/strategies/*.py`
- **THEN** MUST NOT 出现 `def compute_signals(` / `def compute_signals_for_one_bar(` /
  `def check(self,` / `state_spec` / `@njit` / `dsl_check` 等已废形态

### Requirement: Indicators are private to strategies (framework MUST NOT compute any indicator)

框架 MUST NOT 预计算任何指标（EMA / ATR / RSI / Bollinger 等）任何符号或字段。所有指标计算由
策略在 `step(state, bar, params)` body 内通过 `evtrade/indicators/` 子包导出的 API 完成。
`evtrade/indicators/` MUST 提供三类算子：
1. **step 增量版**（`@dataclass state` in/out，策略 `step` 逐桶调用）：
   `ema_step` / `ema_channel_step` / `atr_step` / `rsi_step` / `sma_step` / `boll_step`
2. **xp 批量版**（`xp_ema` / `xp_ema_channel` / `xp_atr` / `xp_rsi` / `xp_sma` / `xp_bollinger`，
   兼容旧 numpy/cupy/torch 模块签名，engine fast-path 用，策略不直接调）
3. **torch 批量版**（`xp_ema_torch` / `xp_ema_channel_torch`，`(B,T)` tensor 输入输出）+
   **纯 ndarray 版**（`ema` / `atr` / `rsi` / `sma` / `bollinger`，jupyter / 复盘用）

`evtrade/indicators/` MUST NOT `import cupy` / `import numba`，MUST NOT 使用 `@njit`，
MUST NOT 包含 CUDA C99 字符串。`pyproject.toml` MUST NOT 声明 `cupy` / `numba` 为依赖，
MUST 声明 `torch>=2.0`。

#### Scenario: 框架不假定任何指标名
- **WHEN** 静态扫描 `evtrade/core/**/*.py`（排除 `__pycache__` 与 `indicators/`）
- **THEN** MUST NOT 出现 `IncrementalEMA` / `EMAChannel` / `k_ema` 等指标符号

#### Scenario: 策略 step 自维护指标增量
- **WHEN** 一个策略（如 `ChannelDeviationStrategy`）需要 EMA 通道
- **THEN** 策略 MUST 在 `step` body 内调 `ema_channel_step(state, bar["h"], bar["l"], tf1)`
  拿 `(state, up, dw)`；step 增量版与批量版信号轨迹 MUST bitwise 一致

#### Scenario: pyproject 后端依赖
- **WHEN** 读取 `pyproject.toml`
- **THEN** MUST 含 `torch>=2.0`；MUST NOT 含 `cupy` / `numba` 为（可选或必选）依赖
  （`tests/test_pyproject.py` 锁定）

### Requirement: Strategy display hooks are framework-agnostic

`VectorizedStrategy` MUST 仅暴露一个可覆写展示 hook：
`format_signal_line(self, ts, sig, info=None) -> str`（hook 签名无 positional 指标；所需字段从
`info` dict 自取，默认实现仅打 `ts / sig / side`）。framework（`Engine` / `run_vectorized`
verbose 路径）SHALL NOT 假定任何指标字段名，仅 `print(strategy.format_signal_line(...))`。
`get_extra_bucket_columns` / `get_extra_signal_columns` 已删除（DSL/bundle 时代产物）。

#### Scenario: Engine 仅 print 策略返回值
- **WHEN** verbose 模式下引擎触发信号行打印
- **THEN** 引擎直接 `print(strategy.format_signal_line(ts, sig, info))`，不修改、不假设
  `info` 键集；默认实现仅显示 OHLCV 无关的 `ts / sig / side`

### Requirement: metrics.summary covers full field shape

`evtrade.core.metrics.summarize` MUST 返回 **30 字段**（缺一即视为 broken）：

- **终态（5）**：`final_price / final_cash / final_position / final_equity / baseline`
- **交易（5）**：`n_trades / n_buy / n_sell / turnover / excess_pct`
- **时间（2）**：`years / cagr_excess`
- **风险调整（5）**：`cagr / sharpe_excess / sortino_excess / calmar / ir`
- **回撤（4）**：`max_drawdown / max_dd_days / max_dd_recovered / x_mdd`
  - `x_mdd` = 累计超额曲线（equity - baseline）的回撤，单位**小数**；与 `max_drawdown`
    （equity 曲线回撤）不同，**保留**（`tests/test_metrics_v3.py::test_x_mdd_present` 锁定）
  - `max_dd_recovered` 未恢复时 MUST = -1（sentinel）
- **持仓行为（7）**：`win_rate / profit_factor / avg_pnl / max_consecutive_wins /
  max_consecutive_losses / avg_hold_bars / max_hold_bars`
- **基准对比（2）**：`baseline_max_dd / dd_excess`

字段单位约定沿用下表（`x_mdd` 单位 = 占初始 baseline 的小数，与 `max_drawdown` 同量级）：
`cagr / cagr_excess / excess_pct / win_rate` 为百分数或比率；`max_drawdown / baseline_max_dd /
dd_excess / x_mdd` 为小数；`calmar / sharpe_excess / sortino_excess / ir` 无量纲；
`max_dd_days` 自然日；`avg_hold_bars / max_hold_bars` 桶数；`*_equity / baseline / final_cash /
turnover` 金额元。CLI 打印 MUST 与单位一致（金额 `:.2f`、百分比 `:.2%`、无量纲 `:.3f`、
天数 `:.1f`）。

#### Scenario: summary 30 字段齐全
- **WHEN** `run_vectorized(...)` 返回 `result["summary"]`
- **THEN** `set(summary.keys())` MUST 恰好等于上述 30 字段（`tests/test_metrics_v3.py::
  test_field_set_exactly_26` 的 `REQUIRED_KEYS` 已锁定 30 键）；`x_mdd` MUST 存在且为小数

#### Scenario: x_mdd 为超额曲线回撤小数
- **WHEN** 读取 `summary["x_mdd"]`
- **THEN** MUST 满足 `0.0 <= x_mdd <= 2.0`（量级为小数，非元；`tests/test_metrics_units.py::
  test_x_mdd_is_fraction` 锁定）

#### Scenario: max_drawdown 单位为 peak 的小数
- **WHEN** 读取 `summary["max_drawdown"]`
- **THEN** MUST 等价于 `max(0, peak_equity - trough_equity) / peak_equity`，量级 ∈ [0.0, 1.5]

### Requirement: CLI surface = --device {cpu, gpu, auto}

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto}`（默认
`auto`）。后端为 **torch**：`evtrade.backends.get_xp(device) -> torch.device`，
`"cpu" -> cpu`、`"gpu"/"cuda" -> cuda`、`"auto"` 优先 cuda（不可用降级 cpu）。
`get_xp("gpu")` / `get_xp("cuda")` 在 CUDA 不可用时 MUST fallback 到 cpu 并打 `RuntimeWarning`
（`"auto"` 不告警静默降级）。CLI MUST NOT 暴露 `--engine kernel|ref|vectorized` 三选一；
兼容层检测到旧 `--engine` 时打 `DeprecationWarning` 并自动映射（`kernel→auto / ref→cpu /
vectorized→cpu`），不长期保留。

#### Scenario: 旧 --engine 自动转换
- **WHEN** `python -m evtrade backtest --engine ref ...`
- **THEN** 打 `DeprecationWarning`，`--engine ref` 自动映射为 `--device cpu`，继续执行

#### Scenario: gpu 不可用告警降级
- **WHEN** `--device gpu` 但 `torch.cuda.is_available()` 为 False
- **THEN** `get_xp` 返 `torch.device("cpu")` 并打 `RuntimeWarning`；回测仍跑通

#### Scenario: 端到端 CLI 跑通
- **WHEN** `python -m evtrade backtest --strategy channel_deviation --device auto --synthetic-days 30`
- **THEN** MUST 跑通并打印 30 字段盈亏汇总（含 cagr / sharpe_excess / sortino_excess /
  calmar / max_dd_days / x_mdd）

### Requirement: PyTorch 统一后端 (NEW 2026-09-10)

`evtrade` MUST 使用 PyTorch 作为唯一 array 后端，CPU/GPU 透明路由由 `tensor.to(device)` 完成。
`evtrade.backends.get_xp(device) -> torch.device` 是统一入口。`evtrade.core.capability.py::
gpu_available()` MUST 委托 `torch.cuda.is_available()`。`evtrade/` MUST NOT 存在 cupy 探测
路径（`grep -r "import cupy" evtrade/` = 0 命中）。

#### Scenario: cupy 不是依赖
- **WHEN** 执行 `grep -r "import cupy" evtrade/`
- **THEN** MUST 0 命中（cupy 已完全下线）

#### Scenario: torch 后端单测通过
- **WHEN** `pytest tests/test_torch_backend.py -v`
- **THEN** `get_xp` 返 `torch.device`；`gpu_available` 与 `torch.cuda.is_available()` 一致；
  `xp_ema_torch` 接受 `(B,T)` tensor 返 `(B,T)`

## REMOVED Requirements

### Requirement: state_spec is mandatory for DSL strategies
**Reason**: DSL 三端转译（numba jitclass / CUDA device 函数 / Python ctx）已在
`unify-strategy-contract` 整体下线；`state_spec` AST 白名单与三端投影无引用方，`core/kernel.py`
与 `strategies/dsl.py` 均已物理删除。
**Migration**: 策略持久状态改用 `step(state, bar, params) -> (state, sig)` 的 `state` 参数
承载（推荐 `@dataclass`，由引擎跨调用持有），`init_state(params)` 提供初值。见
`Strategy interface contract`。

### Requirement: Per-bar dict contract (bundle_per_bar)
**Reason**: `core/kernel.py` 与 `kernel.bundle_per_bar` 已删除；per-bar 契约现由
`step(state, bar, params)` 的 `bar` dict `{ts,o,h,l,c,v,mark}` 承载。
**Migration**: 见 `Strategy interface contract` 的 bar dict 契约。

### Requirement: bucket_table is framework-only (no indicator columns)
**Reason**: `kernel.bucket_table` 已删除；信号轨迹输出改由 `cli --signals-out` 直接写
`stime,signal`（策略已无指标列 hook）。
**Migration**: 信号轨迹用 `python -m evtrade {backtest,replay} --signals-out out.csv`，
输出列为 `stime,signal`。

### Requirement: DSL→CUDA projection lives in strategies/dsl.py
**Reason**: `strategies/dsl.py` 与 `_CALL_WHITELIST` 已删除；GPU 加速由 torch 算子
（`tensor.to("cuda")`）透明提供，无 NVRTC / 手写 CUDA C99。
**Migration**: 无需操作；策略写 torch 兼容算子，`--device gpu` 自动路由到 CUDA。

### Requirement: Indicators are private to strategies — 2026-09 tightening
**Reason**: 与改写后的 `Indicators are private to strategies (framework MUST NOT compute any
indicator)` 条款重叠；其 "state_spec 声明 EMA 增量字段" 场景已被 step 契约取代。
**Migration**: 指标私有性约束并入保留的 `Indicators are private ...` Requirement。
