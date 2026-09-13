# evtrade-architecture (delta)

> 本 spec 是 2026-09-10 `strategy-step-only` change 的 delta；落地后由 `openspec archive` 合并到 `openspec/specs/evtrade-architecture/spec.md`。

## Removed Requirements（DSL 时代死代码，spec 未及时删）

### ~~Requirement: state_spec is mandatory for DSL strategies~~
**REMOVED 2026-09-10 (change: strategy-step-only).** DSL 三端转译整体下线后
（见 `unify-strategy-contract`），`state_spec` AST 白名单 + 三端投影已无引用方。
策略持久状态改由 `VectorizedStrategy.step(state, bar, params) -> (state, sig)`
的 state 参数承载（state 类型由策略自定，推荐 `@dataclass`），不再走 DSL 白名单。

### ~~Requirement: DSL→CUDA projection lives in `strategies/dsl.py`~~
**REMOVED 2026-09-10 (change: strategy-step-only).** `strategies/dsl.py` 与
`_CALL_WHITELIST` 在 `unify-strategy-contract` 时已物理删除，本条 Requirement 是 spec
未同步删的死条款。CPU/GPU 性能由 CuPy 高阶封装承担（`cupy.where / cumsum / searchsorted` 等），
策略层仅通过 `xp` 抽象后端使用 numpy 兼容算子。

## Removed Scenarios

### ~~Scenario: 策略覆写 compute_signals_for_one_bar 维护 instance FSM~~
**REMOVED 2026-09-10 (change: strategy-step-only).** `compute_signals_for_one_bar`
已删除；策略持久状态改由 `step(state, ...)` 显式传入传出，无 instance state。

## Modified Requirements

### Requirement: Strategy interface contract

策略 MUST 实现 `VectorizedStrategy.step(self, state, bar: dict, params: dict) -> tuple[state, int]`
**唯一入口**。`bar` 契约 MUST 为 dict `{"ts", "o", "h", "l", "c", "v", "mark"}`
（单桶 OHLCV + 策略期标记；`mark == 0` 表示预热段，策略 step 内必须返 `(state, 0)`）。
返回元组 MUST 长度 2：`(new_state, signal)`，`signal` ∈ {-1, 0, 1}（SELL / 无 / BUY）。

策略 MUST NOT 持有 instance-level 持久状态（不允许 `self._fsm / self._up_st` 等 instance attr）——
所有跨调用持久化 MUST 由 `state` 参数承载（推荐 `@dataclass`，但 dict 也允许）。
策略如需自定义 state 初值，SHOULD 覆写 `VectorizedStrategy.init_state(self, params) -> state`
（基类默认返回 `None`，表示无状态）。

引擎层（`VectorizedEngine.run_vectorized` / `Engine.on_bars`） MUST 在循环外持有 state，
每桶一次调用 `strategy.step(state, bar, params)`，state 跨调用持续；策略 MUST 仅做"算法"
（拿到 `{ts, h, l}` + 上一步 state → 算指标 → FSM → 返回 `(state, sig)`），
MUST NOT 在 step 内出现：
- 批量循环 (`for i in range(n)`)
- `xp` 模块引用 / `hasattr(ts, "get").get()` 数据迁移
- `mark == 0` 之外的桶过滤逻辑

`params_spec` 类属性 MUST 声明所有策略参数（含 default / type / min / max），
framework 在 `__init__` / `_resolve_params` 阶段做校验。
策略代码 MUST NOT `import numba` / `import cupy`（除通过 `xp` 参数间接使用）。

#### Scenario: framework 不预计算指标
- **WHEN** 引擎（任一路径）调 `strategy.step(state, bar, params)`
- **THEN** MUST 传入 `bar` dict 仅含 `ts/o/h/l/c/v/mark`；MUST NOT 包含 `up` / `dw` / `low_dev` / `high_dev` 等任何具体指标键

#### Scenario: 策略 step 新签名
- **WHEN** 一个策略类继承 `VectorizedStrategy` 并实现 `step(self, state, bar, params)`
- **THEN** framework MUST 按新签名调用；策略 MUST 在 body 内返回 `(new_state, signal)`，
且 signal ∈ {-1, 0, 1}

### Requirement: Indicators are private to strategies (framework MUST NOT compute any indicator)

框架 MUST NOT 预计算任何指标（EMA / ATR / RSI / Bollinger 等）任何符号或字段。
所有指标计算由策略在 `step(state, bar, params)` body 内通过 `evtrade/indicators/` 子包导出的 API 完成。
`evtrade/indicators/` 子包 MUST 提供两类函数：
- **批量 xp 版**（`numpy.ndarray` 或 `cupy.ndarray`，engine 内部 fast-path 用，
  策略**不直接调用**）：`xp_ema(xp, values, p)` / `xp_ema_channel(xp, h, l, p)` /
  `xp_atr` / `xp_true_range` / `xp_rsi` / `xp_sma` / `xp_bollinger` —— 内部走 numpy/cupy 高阶算子
- **step 增量版**（`@dataclass state` in/out，供策略 `step` 调用）：
  `ema_step(state, value, p) -> (state, ema)` /
  `ema_channel_step(state, h, l, p) -> (state, up, dw)` /
  `atr_step(state, h, l, c, p) -> (state, atr)` /
  `rsi_step(state, close, p) -> (state, rsi)` /
  `sma_step(state, value, p) -> (state, sma)` /
  `boll_step(state, value, p, k) -> (state, mid, upper, lower)`

`evtrade/indicators/` 子包 MUST NOT import `numba`、MUST NOT 使用 `@njit` 装饰器、
MUST NOT 包含 `CUDA_DEVICE_*` C99 字符串常量。
`pyproject.toml` MUST NOT 声明 `numba` 为 required 依赖。

#### Scenario: 框架不假定任何指标名
- **WHEN** 静态扫描 `evtrade/core/**/*.py`（排除 `__pycache__` 与 `indicators/`）
- **THEN** MUST NOT 出现 `IncrementalEMA` / `EMAChannel` / `tf1` / `k_ema` 任何符号

#### Scenario: 策略 step 用 compute_ema_channel 拿 up/dw
- **WHEN** 一个策略（如 `ChannelDeviationStrategy`）需要 EMA 通道
- **THEN** 策略 MUST 在 `step` body 内调 `ema_channel_step(state, bar["h"], bar["l"], tf1)`
  拿 `(state, up, dw)`；CUDA 路径下 engine fast-path 用 `xp_ema_channel` 一次算全序列，
state 由 engine 持有并传入 step（state 在 GPU 上是 `cupy.ndarray`）

### Requirement: Single contract = `VectorizedStrategy.step`

策略 MUST 是 `evtrade.strategies.VectorizedStrategy` 的子类（通过 `@register_strategy("name")` 注册）。
策略 MUST 实现 `step(self, state, bar, params) -> tuple[state, int]` 一个方法。
策略 MUST NOT 拥有除 `step / format_signal_line / init_state` 之外的 framework 调用入口；
不允许有 `check(cur)` / `_strategy_check` / `state_spec` 等旧 DSL 方法。
`params_spec` 类属性 MUST 声明所有策略参数（含 default / type / min / max），
framework 在 `__init__` / `_resolve_params` 阶段做校验。
策略代码 MUST NOT `import numba` / `import cupy`（除通过 `xp` 参数间接使用）。
同一份策略代码 MUST 在 `xp = numpy` 与 `xp = cupy` 下产生 bitwise 一致的信号序列
（cupy 高阶算子与 numpy 在数值上 bitwise 一致；如有浮点舍入差异则在
`tests/test_strategy_unified.py::test_step_cpu_vs_gpu_*` 锁定）。

#### Scenario: 策略代码一份同时跑 CPU / GPU / ref
- **WHEN** 同一份 `ChannelDeviationStrategy` 策略代码分别在 `device="cpu"` 与 `device="gpu"` 下跑同一组数据
- **THEN** 返回的 sig 序列 MUST bitwise 一致（`np.array_equal(sig_cpu, sig_gpu)` 为 True）

#### Scenario: vectorized vs ref 引擎 bitwise 一致
- **WHEN** 同一份策略代码分别走 `VectorizedEngine.run_vectorized`（批量 fast-path 调 step）
与 `Engine.on_bars`（逐 bar 调 step）两条路径
- **THEN** 生成的 `trades` list 与终态 `cash / position` MUST 逐笔一致
（同 ts / side / qty / price）

#### Scenario: 策略无 instance state
- **WHEN** 静态扫描 `evtrade/strategies/*.py`（排除 `vectorized_base.py`）
- **THEN** MUST NOT 出现 `self._fsm` / `self._up_st` / `self._dw_st` /
`self._has_prev` 等 instance-level 状态字段；所有持久状态 MUST 由 `step` 的 state 参数承载

### Requirement: CLI surface = `--device {cpu, gpu, auto}`

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto}`（默认 `auto`）；
`auto` 模式下 framework 优先 GPU（cupy 可用时）否则降级 CPU。
CLI MUST NOT 暴露 `--engine kernel|ref|vectorized` 三选一选项
（已统一为单 `--device` 控制 xp 后端）。
兼容期内检测到旧 `--engine` 时 MUST 打 `DeprecationWarning` 并自动转换
（`kernel→auto / ref→cpu / vectorized→cpu`），但不永久保留。

#### Scenario: 旧 --engine 自动转换
- **WHEN** `python -m evtrade backtest --engine ref --strategy X ...`
- **THEN** 打 `DeprecationWarning: --engine 已废弃，请用 --device {cpu, gpu, auto}`，
参数 `--engine ref` 自动映射为 `--device cpu`，继续执行

#### Scenario: 端到端 CLI 跑通
- **WHEN** `python -m evtrade backtest --strategy channel_deviation --device auto --synthetic-days 30`
- **THEN** MUST 跑通并打印完整盈亏汇总（含 cagr / sharpe_excess / sortino_excess / calmar / max_dd_days）