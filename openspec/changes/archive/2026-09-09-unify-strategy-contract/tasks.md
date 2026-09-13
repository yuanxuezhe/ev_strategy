# Tasks: 统一策略契约 — 去掉 DSL

## 1. Spec / KB / pyproject 同步

- [x] 1.1 改 `openspec/specs/evtrade-architecture/spec.md`：删 `Requirement: state_spec is mandatory for DSL strategies`；删 `Requirement: DSL→CUDA projection lives in strategies/dsl.py`；改 `Strategy interface contract` 把 `check(cur, indicators)` 替换为 `compute_signals(xp, bars, params)`；新增 `Requirement: Single contract = VectorizedStrategy.compute_signals`；改 `Indicators are private to strategies` Scenario 把 DSL 三端投影相关条目改为"xp / 纯 Python 函数"。
- [x] 1.2 `kbs/14-策略DSL与三端转译.md`：整篇废止，重写为"统一策略契约"（~80 行）。
- [x] 1.3 `kbs/12-重构与性能内核.md`：§2/§5/§6 (numba/CUDA 段) 整段重写。
- [x] 1.4 `kbs/02-系统架构.md`：line 59 三端表改为"CPU ref 逐 bar + CuPy 向量化"。
- [x] 1.5 `kbs/05-指标计算-EMA通道.md`：删 `CUDA_DEVICE_*` 引用；批量版 + 增量版（纯 Python）。
- [x] 1.6 `kbs/06-交易策略详解.md`：§5 DSL body 整段改写为 compute_signals 写法。
- [x] 1.7 `kbs/09-引擎Engine与主流程.md`：line 17 / §2 on_bars 描述重写。
- [x] 1.8 `kbs/10-配置参数与运行指南.md`：line 31 `--engine` 行改为 `--device`。
- [x] 1.9 `kbs/11-扩展指南.md`：§3 DSL 三步法改为 `VectorizedStrategy` 模板；§5 DSL 增量 API 改为普通 Python 函数。
- [x] 1.10 `kbs/13-绩效评估与鲁棒选参框架.md`：line 18 改为 vectorized 引擎 + equity 序列累积。
- [x] 1.11 `CLAUDE.md`：源码地图 / 关键约定 / 验证命令 grep 关键词全部更新。
- [x] 1.12 `pyproject.toml`：`description` / `keywords` 改写；`dependencies` 删 `"numba>=0.56"`。
- [x] 1.13 跑 `openspec validate --specs` 通过。

## 2. indicators 子包 xp 化（删 numba）

- [x] 2.1 `evtrade/indicators/ema.py`：删 `from numba import njit`；删所有 `@njit(cache=True)`；删末尾 `CUDA_DEVICE_EMA_*` 字符串常量；批量 `ema(values, p)` 改返 `np.ndarray`；增量 `ema_push / ema_current / ema_channel_push / ema_channel_current` 改纯 Python（保留 3-6 标量 in/out API）。
- [x] 2.2 `evtrade/indicators/atr.py`：同样去 numba + CUDA_DEVICE_*；`true_range / atr` 用 numpy；`atr_push / atr_current` 改纯 Python。
- [x] 2.3 `evtrade/indicators/rsi.py`：同样形态。
- [x] 2.4 `evtrade/indicators/boll.py`：同样形态。
- [x] 2.5 `evtrade/indicators/__init__.py`：删 `CUDA_DEVICE_*` re-export；`__all__` 重写。
- [x] 2.6 验证：`uv run python -c "from evtrade.indicators import ema_push, ema_current; print(ema_current(0.0, 0, 0.0, 0.0, 21))"` 通过。

## 3. metrics 补齐 + vectorized 引擎改

- [x] 3.1 `evtrade/core/metrics.py::summarize` 重写：接收 `equity_curve / baseline_curve / years / n_trades / n_buy / n_sell / turnover / init_cash / init_position / final_price`，输出全套 16 字段（cagr / sharpe_excess / sortino_excess / calmar / max_dd_days / max_dd_recovered / x_mdd / max_drawdown）。
- [x] 3.2 `evtrade/core/vectorized_engine.py::_execute_trades`：累积 `equity_curve` + `baseline_curve` 返回。
- [x] 3.3 `evtrade/core/vectorized_engine.py::run_vectorized`：删 `from .kernel import encoded_to_epoch, resolve_period_seconds`；改 `from .timeutils import encoded_to_epoch, resolve_period_seconds`；末尾调新 `metrics.summarize`。
- [x] 3.4 `evtrade/core/vectorized_engine.py::_summarize` 删除（旧内置简化版被 metrics.summarize 替代）。

## 4. 策略基类统一

- [x] 4.1 `evtrade/strategies/vectorized_base.py`：搬入 `_STRATEGIES` / `register_strategy` / `get_strategy` / `get_strategy_class` / `get_strategy_param_spec` / `available_strategies` + `_resolve_params`；新增 `compute_signals_for_one_bar(xp, bar, params) -> int` 默认实现（包单元素数组调 compute_signals 取 [0]）。
- [x] 4.2 `evtrade/strategies/base.py`：物理删除。
- [x] 4.3 `evtrade/strategies/channel_deviation.py`：单继承 `VectorizedStrategy`；删 `check()`；保留 `_fsm_step` Python FSM + cupy `.get()` 拉回 host；覆写 `compute_signals_for_one_bar` 维护 instance FSM；改 import 路径。
- [x] 4.4 `evtrade/strategies/ma_crossover.py`：`from .base import register_strategy` → `from .vectorized_base import register_strategy`。
- [x] 4.5 `evtrade/strategies/_defaults_loader.py`：改 import 路径。
- [x] 4.6 `evtrade/strategies/__init__.py`：删 `base.py / dsl.py / example_*` import；只 re-export `VectorizedStrategy` + 注册副作用入口。
- [x] 4.7 验证：`from evtrade import VectorizedStrategy, get_strategy, available_strategies; 'channel_deviation' in available_strategies()` 通过。

## 5. 删 DSL 渲染层与示例

- [x] 5.1 物理删除 `evtrade/strategies/dsl.py`（907 行）。
- [x] 5.2 物理删除 `evtrade/strategies/example_dev_trigger.py`。
- [x] 5.3 物理删除 `evtrade/strategies/example_breakout.py`。

## 6. 清死 import + 删 kernel / kernel_dsl

- [x] 6.1 物理删除 `evtrade/core/kernel.py` / `evtrade/core/kernel_dsl.py`（磁盘已删，确认）。
- [x] 6.2 `evtrade/__init__.py`：删所有 `_sys.modules.setdefault("evtrade.kernel", ...)` 等已死模块 shim；删 `incremental_indicators` stub；删 `from .core.kernel / kernel_dsl / .strategies.dsl import ...`；`__all__` 重写。
- [x] 6.3 `evtrade/core/replay.py`：删 `from .kernel import ...`；删 `replay_kernel`；`replay_engine` 改走 `run_vectorized` + Engine.on_bars 双路径；`reconcile` 简化为 vectorized vs Engine.on_bars 对账。
- [x] 6.4 `evtrade/core/sweep.py`：删 `from .kernel_dsl import _EMPTY_F, _EMPTY_SIG, run_one_dsl, strategy_has_dsl`；`run_one_from_dict` 改调 `run_vectorized` 拿全 metrics；`sweep()` 内 `dsl_fast` 分支删除，统一 `run_one_vectorized`；`run_one_general` 仍走 Engine 但补 equity + metrics。
- [x] 6.5 `evtrade/core/engine.py`：`on_bars` 重写为 `compute_signals_for_one_bar`；删 `_sync_strategy_state`；`tf1` 参数废弃保留兼容；`build_engine` 工厂保留。
- [x] 6.6 `evtrade/core/permutation.py`：`from .sweep import run_one_from_dict` 不变。
- [x] 6.7 `evtrade/core/gpu.py`：删 `cuda_sweep_window_generic`（已无引用方）；保留 `_ensure_cupy` / `gpu_info` / `precompute_ts_mark`。
- [x] 6.8 `evtrade/core/capability.py`：删除 `TARGET_CAPS["cpu"]["max_params"]` 数字限制。
- [x] 6.9 `evtrade/core/__init__.py`：删 kernel / engine shim 描述。

## 7. CLI 收口 `--device`

- [x] 7.1 `evtrade/cli.py::build_backtest_parser`：删 `--engine` 选项；保留 `--device {cpu, gpu, auto}` 默认 `auto`。
- [x] 7.2 `evtrade/cli.py::backtest_main`：删 `_run_kernel / _run_ref` 分支；统一调 `_run_backtest`（原 `_run_vectorized` 重命名）。
- [x] 7.3 `evtrade/cli.py::build_sweep_parser`：`--device` choices 保持 `{auto, cpu, gpu}`；help 去 numba/CUDA 字样。
- [x] 7.4 `evtrade/cli.py::sweep_main`：删 `from .kernel import bars_to_arrays`；改 `from .metrics import bars_to_arrays`。
- [x] 7.5 `evtrade/cli.py::build_replay_parser`：保留 `--against-ref`；删 `--engine`。
- [x] 7.6 `evtrade/cli.py::replay_main`：删 `replay_kernel` 路径；统一调 `run_vectorized`。
- [x] 7.7 `evtrade/cli.py::main`：检测到 `--engine` 时打 `DeprecationWarning` 并自动转换（hidden arg）；不影响默认调用。

## 8. 测试重写

- [x] 8.1 物理删除 15 个旧 DSL/kernel 测试：`tests/test_dsl_cuda.py` / `tests/test_dsl_whitelist.py` / `tests/test_dsl_runner_cache.py` / `tests/test_kernel_dsl.py` / `tests/test_kernel_dsl_cache.py` / `tests/test_kernel_unit.py` / `tests/test_metrics.py` / `tests/test_funding.py` / `tests/test_differential.py` / `tests/test_state_spec_generalization.py` / `tests/test_strategy_params.py` / `tests/test_replay.py` / `tests/test_sweep.py` / `tests/test_sweep_warmup.py` / `tests/test_sweep_main_resolve.py`。
- [x] 8.2 新增 `tests/test_strategy_unified.py`：核心锁定 6 项（signature / cpu-vs-gpu-ma / cpu-vs-gpu-channel / vectorized-vs-ref-channel / metrics 字段集）。
- [x] 8.3 新增 `tests/test_strategy_params_v2.py`：字典 / kwargs / 默认 / 范围校验。
- [x] 8.4 新增 `tests/test_metrics_v2.py`：cagr / sharpe / sortino / calmar / max_dd_days 与手算一致。
- [x] 8.5 新增 `tests/test_replay_v2.py`：reconcile vectorized vs Engine.on_bars pass=True。
- [x] 8.6 新增 `tests/test_sweep_v2.py`：sweep 全 vectorized, metrics 字段非 0。
- [x] 8.7 改 `tests/test_vectorized.py`：import 路径改；加 cagr / sortino / calmar / max_dd_days 断言。
- [x] 8.8 改 `tests/test_cli.py`：加 `--device auto` 默认值检查；删 `--engine` 断言。
- [x] 8.9 改 `tests/test_cli_params.py`：验证落盘 JSON 不含 DSL 痕迹。
- [x] 8.10 改 `tests/test_strategy_class_access.py`：断言视角改 VectorizedStrategy。
- [x] 8.11 改 `tests/test_strategy_template.py`：模板策略继承 VectorizedStrategy；test_08_replay_engine 走新 reconcile。
- [x] 8.12 改 `tests/conftest.py`：删 `_RUNNER_BY_KEY.clear()` 那两行。
- [x] 8.13 跑 `uv run pytest -q` 全部 PASS（36 个 → 26 个 test 文件）。

## 9. 端到端 + 一致性校验

- [x] 9.1 `python -m evtrade backtest --strategy channel_deviation --device auto --start 20250101 --end 20260101 --synthetic-days 30` 跑通。
- [x] 9.2 `python -m evtrade backtest --strategy ma_crossover --device cpu ...` 跑通。
- [x] 9.3 `python -m evtrade sweep --strategy channel_deviation --device gpu --grid low1=0.5,1.0 --splits 20250601 --out /tmp/sweep.csv` 跑通，CSV 含非零 sortino / calmar / cagr。
- [x] 9.4 `python -m evtrade replay --log /tmp/demo.log --strategy channel_deviation --device cpu --against-ref` 跑通且对账 PASS。
- [x] 9.5 `openspec validate --specs` 通过。
- [x] 9.6 grep 校验：`grep -rn 'kernel_dsl\|strategy_has_dsl\|cuda_sweep_window_generic\|numba\|@njit\|state_spec\|_DSL_\|_CTX_TO_KERNEL\|dsl_check\|make_dsl_ctx\|compile_all\|render_numba' evtrade/ kbs/ openspec/specs/` 应 0 命中。
- [x] 9.7 `grep -rn 'CUDA_DEVICE_' evtrade/ kbs/` 应 0 命中。
- [x] 9.8 `grep -rn 'compute_signals' kbs/ openspec/specs/` 应在 kbs/02 / kbs/06 / kbs/11 / kbs/14(重写后) / spec.md 多处命中。

## 10. openspec archive

- [x] 10.1 `openspec archive --change 2026-09-09-unify-strategy-contract` 落地。
