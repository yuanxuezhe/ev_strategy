# evtrade-architecture (delta)

> 本 spec 是 2026-09-09 `unify-strategy-contract` change 的 delta；落地后由 `openspec archive` 合并到 `openspec/specs/evtrade-architecture/spec.md`。

## Removed Requirements

### ~~Requirement: state_spec is mandatory for DSL strategies~~
**REMOVED 2026-09-09 (change: unify-strategy-contract).** DSL 三端转译（Python exec + numba @njit + CUDA C99）整体下线；策略不再声明 `state_spec`，持久状态由策略子类自己以 instance 字段维护（Python FSM 走 `compute_signals_for_one_bar`；批量向量化路径用函数内局部变量）。

### ~~Requirement: DSL→CUDA projection lives in `strategies/dsl.py`~~
**REMOVED 2026-09-09 (change: unify-strategy-contract).** `strategies/dsl.py`（907 行 DSL AST 白名单 + 三端渲染器）整体删除；`evtrade/core/kernel.py`（889 行 numba 流式内核）和 `evtrade/core/kernel_dsl.py`（307 行 numba 特化内核）已删除；`evtrade/core/gpu.py::cuda_sweep_window_generic`（NVRTC 通用 kernel）已删除。性能由 CuPy 高阶封装承担（cupy.where / cupy.cumsum / cupy.searchsorted 等映射到 cuBLAS / cuDNN 预编译 kernel），无需手写 CUDA。

## Modified Requirements

### Requirement: Strategy interface contract

策略 MUST 实现 `VectorizedStrategy.compute_signals(self, xp, bars: dict, params: dict) -> xp.ndarray[int8]`，**这是策略唯一入口**。`xp` MUST 为 `numpy` 或 `cupy` 模块（framework 通过 `evtrade.backends.get_xp(device)` 传入）；策略代码 MUST 仅使用 `xp` 提供的数组算子，禁止直接 `import numpy` / `import cupy` 后用具体名称。`bars` 契约 MUST 为 dict `{"ts", "o", "h", "l", "c", "v", "mark", "n_bars"}`（桶聚合后的 OHLCV + 策略期标记）。返回数组 MUST 为 `xp.int8`，长度 MUST 等于 `len(bars["ts"])`；元素 MUST ∈ {-1, 0, 1}（SELL / 无 / BUY）。`mark == 0` 的桶（预热段）策略 SHOULD 输出 0；framework 会再做一次 `sig *= mark` 兜底。framework MUST NOT 在调用 `compute_signals` 之前预计算任何指标（无 `up` / `dw` 等键出现于 bars dict）。策略如需在逐 bar 路径（`Engine.on_bars`）维护 instance 状态（FSM / 计数器等），SHOULD 覆写 `VectorizedStrategy.compute_signals_for_one_bar(self, xp, bar: dict, params: dict) -> int`；基类默认实现把单 bar 包成单元素 xp 数组后调 `compute_signals` 取 `[0]`（**仅对无状态的纯数组算子策略适用**）。

#### Scenario: framework 不预计算指标
- **WHEN** `run_vectorized` 调 `strategy.compute_signals(xp, buckets, params)`
- **THEN** MUST 传入 `buckets` dict 仅含 `ts/o/h/l/c/v/mark/n_bars`；MUST NOT 包含 `up` / `dw` / `low_dev` / `high_dev` 等任何具体指标键

#### Scenario: 策略 compute_signals 新签名
- **WHEN** 一个策略类继承 `VectorizedStrategy` 并实现 `compute_signals(self, xp, bars, params)`
- **THEN** framework MUST 按新签名调用，策略 MUST 在 body 内用 `xp` 数组算子完成所有计算（不再有 DSL docstring、不再有 `state_spec`、不再有 numba `@njit` / CUDA `__device__` 调用）

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

## Added Requirements

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
