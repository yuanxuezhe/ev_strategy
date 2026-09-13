# 2026-09-09: 统一策略契约 — 去掉 DSL，保留 Python CPU + CuPy GPU 两条路径

## Why

`cupy-unified` 分支已经把 `channel_deviation` 重写成"compute_signals(xp) + check() + 共享 FSM"形态，且 `core/kernel.py` / `core/kernel_dsl.py` 已在磁盘删除；但当前 main 分支处于 broken 状态：

- `import evtrade` 第一步就抛 `ModuleNotFoundError: No module named 'evtrade.core.kernel_dsl'`（`strategies/channel_deviation.py:225 → core/sweep.py:25` 引用已死模块）。
- 15 个测试仍依赖 `kernel_dsl / dsl / kernel`；CLI 的 `kernel / vectorized / sweep / replay --against-ref` 路径全断。
- `pyproject.toml` description / keywords / numba 依赖仍按"DSL 三端同源" 描述；KB / spec 仍以 DSL 为核心约定。

DSL 三端转译（Python exec + numba @njit + CUDA C99）维护成本高、策略代码扩到三份拷贝、`dsl.py` 907 行 + `kernel.py` 889 行 + `kernel_dsl.py` 307 行堆叠形成认知包袱。

**目标**：彻底关掉 DSL 渲染层 + numba 流式内核 + NVRTC CUDA 编译；策略代码**只写一份**，由 `compute_signals(xp, bars, params)` 一个入口承担 CPU/GPU/ref 三条路径；CLI / sweep / replay 全部收口为 `--device {cpu, gpu, auto}`。

## What

### 1. 策略契约统一

- 唯一基类 `VectorizedStrategy`（`strategies/vectorized_base.py`）；唯一抽象方法 `compute_signals(xp, bars, params) -> xp.ndarray[int8]`。
- framework 在 `Engine.on_bars` 中把单 bar dict 包成单元素 xp 数组 → 调 compute_signals → 取 `[0]`；channel_deviation 子类覆写 `compute_signals_for_one_bar` 维护 instance FSM 避免 O(n²) 重算。
- 删 `strategies/base.py::StrategyBase` / `state_spec` / DSL docstring 协议。
- 删 `strategies/{dsl, example_dev_trigger, example_breakout}.py`。
- 保留 `strategies/ma_crossover.py` 作为向量化范式（几乎无改）。

### 2. indicators 子包 xp 化（彻底删 numba）

- `evtrade/indicators/{ema,atr,rsi,boll}.py`：删 `from numba import njit`、删 `@njit(cache=True)`、删 `CUDA_DEVICE_*` C99 字符串常量。
- 批量版（`ema / ema_channel / atr / true_range / rsi / sma / bollinger`）改返 `np.ndarray`，内部用 numpy 向量化算子。
- 增量版（`ema_push / ema_current / ema_channel_push / ema_channel_current / atr_push / atr_current / rsi_push / rsi_current / sma_push / sma_current / boll_push / boll_current`）保留 3-6 标量 in/out API 但改为纯 Python 函数。
- `pyproject.toml` 删 `"numba>=0.56"` 依赖。

### 3. CLI 收口 `--device`

- 所有子命令（backtest / sweep / replay）内部统一走 `run_vectorized`；`--device {cpu, gpu, auto}`（默认 `auto`）决定 `xp` 后端。
- 删 `--engine kernel|ref|vectorized`（保留 hidden arg 打 `DeprecationWarning` + 自动转换，限本次回归期）。

### 4. metrics 补齐

`vectorized_engine._summarize` 当前缺 `cagr / sharpe_excess / sortino_excess / calmar / max_dd_days / max_dd_recovered / x_mdd / max_drawdown`。在 `_execute_trades` 循环里累积 `equity_curve[i] = cash + position * close_to_now[i]`，新 `metrics.summarize` 扫描算全套 16 字段；sweep 评分不再依赖占位 0.0。

### 5. 清死 import / 删 kernel / kernel_dsl

- `evtrade/__init__.py` 清掉所有 `_sys.modules.setdefault("evtrade.kernel", ...)` 等已死模块 shim；删 `incremental_indicators` stub。
- `evtrade/core/replay.py` 删 `replay_kernel`；`replay_engine` 改走 `run_vectorized`；`reconcile` 简化为 vectorized vs Engine.on_bars 对账。
- `evtrade/core/sweep.py` 删 `run_one_dsl` / `strategy_has_dsl` 路径；统一走 `run_one_vectorized`。

### 6. 测试重写

- 删 15 个 DSL/kernel 测试。
- 新增 `tests/test_strategy_unified.py`（核心锁定：CPU/GPU 信号 bitwise、vectorized vs Engine.on_bars 逐笔一致、metrics 16 字段齐全）。
- 新增 `test_strategy_params_v2 / test_metrics_v2 / test_replay_v2 / test_sweep_v2` 替换旧版。
- 改 `test_vectorized / test_cli / test_cli_params / test_strategy_class_access / test_strategy_template / conftest`。

### 7. KB / spec / CLAUDE.md / pyproject 同步

- `kbs/14-策略DSL与三端转译.md` 整篇废止重写为"统一策略契约"。
- `kbs/{02,05,06,09,10,11,12,13}.md` 同步。
- `CLAUDE.md` 源码地图 / 关键约定 / 验证命令 grep 关键词全部更新。
- `pyproject.toml` description / keywords / dependencies 改写。

## Impact

- 受影响 capability：`evtrade-architecture`（删 2 条 DSL Requirement、改 2 条、新增 1 条）
- 受影响 files：framework 9 文件 + indicators 4 文件 + strategies 4 文件 + execution / feeds / primitives 不变 + spec 1 文件 + KB 9 文件 + CLAUDE.md + pyproject.toml + tests 15 删 + 5 新 + 6 改
- 受影响 public API：
  - 删：`KernelState / run_backtest / run_backtest_trace / step / bucket_table / summarize / trades_to_list / bars_to_arrays / bucket_ts_encoded / encoded_to_epoch / epoch_to_encoded / _days_from_civil / build_dsl_kernel / dsl_kernel / make_state_general / run_one_dsl / strategy_has_dsl / cuda_sweep_window_generic / compile_all / render_numba_body / render_cuda_body / render_numba_state_body / render_cuda_device_function / build_ctx_to_kernel_map / build_cuda_sig_fields / build_cuda_device_header / make_dsl_ctx / DSLCtx / dsl_check / StrategyBase / state_spec`
  - 保留：`Bar / fmt / compute_bucket / BarAggregator / Account / ChannelDeviationStrategy / MACrossoverStrategy / VectorizedStrategy / get_strategy / available_strategies / run_vectorized / sweep / replay / permutation_test / gpu_info / indicators.*_push / *_current / ema / atr / ...`
- 不破坏外部调用方：保留 `evtrade.{data,config,primitives,account,execution}` 模块 shim（旧路径仍可 import）。

## Out of Scope

- 不实装 CuPy RawKernel / cupy.cuda.compile 等手写 CUDA（沿用 CuPy 高阶封装，足够快）。
- 不优化 EMA 递推为 cumsum 近似（XP 数组算子 + Python FSM 已经够，保留语义清晰）。
- 不重构 `aggregator / execution / feeds / primitives` 子包（已与策略接口解耦）。
- 不引入新策略（保留 `channel_deviation` + `ma_crossover` 两份示例）。

## Spec delta

新增 `openspec/changes/2026-09-09-unify-strategy-contract/specs/evtrade-architecture/spec.md` 的 delta：

- 删 `Requirement: state_spec is mandatory for DSL strategies`（替换为"compute_signals 是策略唯一入口"）。
- 删 `Requirement: DSL→CUDA projection lives in strategies/dsl.py`。
- 改 `Requirement: Strategy interface contract`：`check(cur, indicators)` → `compute_signals(xp, bars, params)`。
- 改 `Requirement: Indicators are private to strategies`：去掉 DSL 三端投影相关 Scenario，加"指标为纯 xp / 纯 Python 函数"。
- 新增 `Requirement: Single contract = VectorizedStrategy.compute_signals`。
