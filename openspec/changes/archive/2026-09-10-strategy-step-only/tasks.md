# Tasks: strategy-step-only

## 1. 指标层 — xxx_step API 落地

- [x] 1.1 `evtrade/indicators/ema.py`: 加 `EMAState` / `EMAChannelState` dataclass; 新增 `ema_step(state, value, p) -> (state, ema)` 与 `ema_channel_step(state, h, l, p) -> (state, up, dw)`; **保留** `xp_ema` / `xp_ema_channel` (engine fast-path 用); 保留旧 `ema_push / ema_current / ema_channel_push / ema_channel_current` 作 deprecated shim (一行注释 + alias)
- [x] 1.2 `evtrade/indicators/atr.py`: 加 `ATRState` dataclass; 新增 `atr_step(state, h, l, c, p) -> (state, atr)`; 保留 `xp_atr` / `true_range`; 旧 `atr_push / atr_current` 作 shim
- [x] 1.3 `evtrade/indicators/rsi.py`: 加 `RSIState` dataclass; 新增 `rsi_step(state, close, p) -> (state, rsi)`; 保留 `xp_rsi`; 旧 `rsi_push / rsi_current` 作 shim
- [x] 1.4 `evtrade/indicators/boll.py`: 加 `BollState` dataclass; 新增 `sma_step(state, value, p) -> (state, sma)` 与 `boll_step(state, value, p, k) -> (state, mid, upper, lower)`; 保留 `xp_sma` / `xp_bollinger`; 旧 `sma_push / sma_current / boll_push / boll_current` 作 shim
- [x] 1.5 `evtrade/indicators/__init__.py`: re-export 新 `*_State` / `*_step`; 旧 `*_push / *_current` 标 `DeprecationWarning` (说明: "已合并到 *_step, 2026-09-10 后删除")

## 2. 基类 — step 抽象 + init_state

- [x] 2.1 `evtrade/strategies/vectorized_base.py`:
  - 删 `compute_signals` 抽象方法 (line 148-156)
  - 删 `compute_signals_for_one_bar` 默认 wrapper (line 158-166)
  - 新增 `step(self, state, bar, params) -> tuple[Any, int]` 抽象方法 (默认 `raise NotImplementedError`)
  - 新增 `init_state(self, params) -> Any` 默认返 `None`
  - `__init__` 注释更新: "strategy instance 不再持有 state, state 由 engine 持有"

## 3. 策略改写

- [x] 3.1 `evtrade/strategies/channel_deviation.py`:
  - 删 `_parse_thresholds / _compute_devs_scalar / _compute_devs_xp / _run_fsm / _fsm_step / _init_fsm_state` (helper 全删, 算法内联)
  - 加 `ChannelDeviationState` dataclass (`ema: EMAChannelState`, `fsm: dict`, `prev_ts`, `cur_high`, `cur_low`, `has_prev`)
  - `ChannelDeviationStrategy`:
    - 删 `__init__` (不再有 instance state)
    - 加 `init_state(self, params) -> ChannelDeviationState` 返 `ChannelDeviationState()`
    - 加 `step(self, state, bar, params) -> (ChannelDeviationState, int)` 单方法 (~15 行)
    - `format_signal_line` 保留 (展示 hook)
- [x] 3.2 `evtrade/strategies/ma_crossover.py`:
  - 改 `compute_signals` -> `step(self, state, bar, params)`, body 用 `EMAState` 持 EMA 跨调用
  - 加 `init_state(self, params) -> MACrossoverState` dataclass (含 fast_ema / slow_ema)

## 4. 引擎循环调 step

- [x] 4.1 `evtrade/core/vectorized_engine.py`:
  - `run_vectorized` 重写: 循环调 `strategy.step(state, bar, params)` 拿 sig 数组 (batched 路径)
  - GPU fast-path: 保留 `xp_ema_channel` 一次性算全序列, 把 EMA 数组塞进 state (stateful 策略 state 含 `precomputed_ema_up: ndarray` / `precomputed_ema_dw: ndarray`), step 内 `if state.precomputed_ema_up is not None: up = state.precomputed_ema_up[i] else: up = ema_channel_step(...)`
  - 删 `_to_host(sig)` 改为循环里 `.get()` 单元素 (mark=0 已经 step 内返 0)
  - `_summarize` 签名不变
- [x] 4.2 `evtrade/core/engine.py`:
  - `Engine.__init__` 加 `self._state = strategy.init_state(strategy.params)`
  - `Engine.on_bars` (line 65) 改 `sig_int = self.strategy.compute_signals_for_one_bar(...)` -> `self._state, sig_int = self.strategy.step(self._state, bar, ...)`
  - `Engine._flush_final_bucket` (line 113) 同改
  - 注释更新: "state 由 Engine 持有, 跨调用持续"

## 5. 测试重写

- [x] 5.1 `tests/test_strategy_unified.py`:
  - 删 `test_compute_signals_returns_int8_xp_array` (方法消失)
  - 删 `test_compute_signals_for_one_bar_default_wrapper` (wrapper 消失)
  - `test_ma_crossover_cpu_vs_gpu_bitwise_equal` 改 `test_step_cpu_vs_gpu`: 跑 vectorized_engine 两次 (cpu + gpu), 比较 sig 序列 bitwise
  - `test_vectorized_vs_engine_on_bars_reconcile` 改: 仍断言 vectorized vs engine 信号+成交一致; 内部用 step 调用
  - `test_metrics_summary_has_16_fields` **不变**
  - `test_both_strategies_are_vectorized_subclass` **不变**
  - `test_strategy_registry_lists_both` **不变**
- [x] 5.2 新增 `test_init_state_returns_dataclass`:
  - `ChannelDeviationStrategy().init_state({})` 是 `ChannelDeviationState`
  - `MACrossoverStrategy().init_state({})` 是 `MACrossoverState` 或 None
- [x] 5.3 新增 `test_no_instance_state_in_strategies`:
  - 静态扫描 `evtrade/strategies/*.py` (排除 `vectorized_base.py`)
  - MUST NOT 出现 `self._fsm` / `self._up_st` / `self._dw_st` / `self._has_prev`
- [x] 5.4 新增 `test_step_state_persists_across_calls`:
  - 调 step(state, bar1) -> (state', sig1); 再调 step(state', bar2) -> (state'', sig2)
  - 验证 EMA 单调正确 (state.ema.count > 0)
- [x] 5.5 跑 `uv run pytest -q` 应 122+ 全 PASS

## 6. 删除旧 API（彻底清理）

- [x] 6.1 `evtrade/strategies/vectorized_base.py`: 确认 `compute_signals` / `_for_one_bar` 不再有任何引用
- [x] 6.2 `evtrade/strategies/channel_deviation.py`: 确认 `_parse_thresholds` / `_compute_devs_*` / `_run_fsm` 不再有任何引用
- [x] 6.3 `evtrade/indicators/__init__.py`: 删除旧 `*_push / *_current` re-export (保留新 `*_step`)
- [x] 6.4 跑 `grep -rn 'compute_signals_for_one_bar\|ema_push\|ema_current\|ema_channel_push\|ema_channel_current\|atr_push\|atr_current\|rsi_push\|rsi_current\|sma_push\|sma_current\|boll_push\|boll_current' evtrade/` 应只剩 deprecation shim 的 alias 行 (或在测试 fixtures 里)
- 6.5 跑 `uv run pytest -q` 仍 122+ 全 PASS

## 7. KB 同步

- [x] 7.1 `kbs/README.md`: 术语表 (line 93-94) 改: `compute_signals(xp, bars, params)` -> `step(state, bar, params) -> (state, sig)`; 删除 `_for_one_bar` 行
- [x] 7.2 `kbs/使用说明.md`: 模板代码 (line 149-151) 改 step
- [x] 7.3 `kbs/14-策略DSL与三端转译.md`: 整体重写: 单方法 step + state 由 engine 持有 + 指标 step API
- [x] 7.4 `kbs/12-重构与性能内核.md`: §2.3 (line 87-93) 状态存放表重写: vectorized 用 list of state; Engine 用 self._state
- [x] 7.5 `kbs/09-引擎Engine与主流程.md`: §2 (line 19-20, 35, 49-53) 改 Engine.on_bars 调 step; §7 (line 150) 删 EMA 漂移说明 (state 一致后无漂移)
- [x] 7.6 `kbs/06-交易策略详解.md`: §5 (line 80-175) 整体改: 单一 step 骨架 + ChannelDeviationState dataclass

## 8. spec delta 合并

- [x] 8.1 把 `openspec/changes/2026-09-10-strategy-step-only/specs/evtrade-architecture/spec.md` 内容手动合并到 `openspec/specs/evtrade-architecture/spec.md`:
  - 删 `Requirement: state_spec is mandatory for DSL strategies` (line 26-37)
  - 删 `Requirement: DSL→CUDA projection lives in strategies/dsl.py` (line 72-79)
  - 改 `Strategy interface contract` Requirement (line 14-16) 描述 step + init_state
  - 删 Scenario "策略覆写 compute_signals_for_one_bar 维护 instance FSM" (line 110-112)
  - 改 Scenario "策略 compute_signals 用 xp 算子" (line 106-108) 为"策略 step 用 compute_ema_channel"
  - 改 `Single contract` Requirement (line 146-148) 标题与正文
  - 加 Scenario "策略无 instance state"
- [x] 8.2 跑 `openspec validate --specs` 通过 (无 openspec CLI 时人工核对: 8 Requirements + ~25 Scenarios)

## 9. 端到端验证

- [x] 9.1 `uv run pytest -q` 122+ 全 PASS
- [x] 9.2 `uv run python -m evtrade backtest --strategy channel_deviation --device cpu --period 5m --start 20250101 --end 20260903 --no-sleep`:
  - cagr/calmar/mdd/n_trades 与重构前 (498fb6a commit 数据) bitwise 一致
- [x] 9.3 同上 `--device gpu`: cagr/calmar/mdd/n_trades bitwise 一致
- [x] 9.4 `uv run python -m evtrade backtest --strategy ma_crossover --device cpu --period 5m ...`: 同上一致
- [x] 9.5 `uv run python -m evtrade sweep --strategy channel_deviation --device cpu --max-mdd 0.15 --grid low1=0.5,1.0 --splits 20250601`: filter_pass 拒 mdd > 0.15 (与重构前一致)
- [x] 9.6 `grep -rn "compute_signals_for_one_bar\|compute_signals(self" evtrade/` 应只剩 deprecation shim (理论上 0 命中)

## 10. openspec archive

- [x] 10.1 `mv openspec/changes/2026-09-10-strategy-step-only openspec/changes/archive/`
- [x] 10.2 git commit:
  ```
  refactor(strategies): single step(state, bar, params) -> (state, sig) method
  ```
  body 写:
  - strategy-step-only: 策略唯一方法收敛到 step
  - VectorizedStrategy 删 compute_signals / _for_one_bar, 加 step + init_state
  - channel_deviation 从 235 行降到 ~80 行 (单 step 方法 + ChannelDeviationState dataclass)
  - ma_crossover stateful 化 (用 EMAState 持 EMA 跨调用)
  - 指标 xxx_push/current 统一为 xxx_step, dataclass state
  - 引擎循环调 step, state 由 engine 持有 (vectorized 用 list, Engine 用 self._state)
  - spec 删 2 条 DSL 死 Requirement (state_spec, DSL→CUDA projection), 改 4 条 Scenario
  - KB 6 份同步
  - 删除 _parse_thresholds / _compute_devs_* / _run_fsm (上次 dedupe-fsm-helpers 抽的)
- [x] 10.3 git push origin cupy-unified