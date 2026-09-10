# 2026-09-10: strategy-step-only — 策略唯一方法收敛到 step(state, bar, params)

## Why

`evtrade/strategies/channel_deviation.py` 在前次 `dedupe-fsm-helpers` 后仍有：
- `compute_signals`（~22 行，批量向量化）
- `compute_signals_for_one_bar`（~22 行，逐 bar 增量）

两个方法体内算法完全一致（EMA 通道 + 4 偏离 + FSM 锁存），差异只在：
1. **循环位置**（批量 vs 逐 bar）—— 这是 engine 的事
2. **EMA 来源**（`xp_ema_channel` 全序列 vs `ema_push/current` 增量）—— 这是指标形态的事
3. **GPU 数据迁移**（`hasattr(ts,"get").get()`）—— 这是后端的事
4. **mark=0 过滤**（`if mark[i]==0: continue`）—— 这是 engine 的事

更糟的是**策略被强制持有 instance state**（`self._fsm / self._up_st / self._dw_st`），
**两个方法体无法真正共享算法**——批量路径用函数内局部变量，逐 bar 路径用 instance attr。

用户原话：「批量和逐 bar，在策略内部保留一样的算法，处理行情的差异部分放到 CPU 和 GPU 引擎里面，不要体现在策略内部」。

**目标**：把"算法"与"执行模型"彻底分层。策略作者只写 `step(state, bar, params) -> (state, sig)` 一个方法，
state 由 engine 持有，engine 决定批量/逐 bar、CPU/GPU、mark 过滤时机。

## What

| 层 | 改动 |
|---|---|
| `VectorizedStrategy` 基类 | 抽象方法改 `step(state, bar, params) -> (state, sig)`；新增 `init_state(params) -> state` 默认返 `None`；删除 `compute_signals` / `_for_one_bar` 默认 wrapper |
| `ChannelDeviationStrategy` | 删两个旧方法，**只剩一个 `step`**，state 用 `@dataclass`；无 `self._fsm / _up_st` 等 instance attr |
| `MACrossoverStrategy` | 同样改写（它现在本来就只有 `compute_signals`，影响小） |
| `indicators/{ema,atr,rsi,boll}.py` | `xxx_push / xxx_current` 双轨统一为 `xxx_step(state, value, p) -> (state, out)`；批量 xp 版 `xp_xxx` 保留供 engine fast-path 用 |
| `VectorizedEngine.run_vectorized` | 内部循环调 `strategy.step(state, bar, params)`；device=gpu 时保留 batched fast-path（一次性算全序列 EMA，state 是 GPU 数组） |
| `Engine.on_bars` | 内部循环调 `strategy.step(state, bar, params)`，state 跨调用持续 |
| `kbs/*` 6 份 | 同步 |
| `openspec/specs/evtrade-architecture/spec.md` | 4 条 Requirement/Scenario 改写；删 2 条 DSL 死代码 Requirement |

**channel_deviation.py**：从 235 行（dedupe 后）降到 ~25 行（单 step 方法 + 1 个 state dataclass）。

## Impact

- 受影响 capability：`evtrade-architecture`（4 条 Scenario 改写 + 2 条 DSL 死 Requirement 删除）
- 受影响 files：
  - 框架：`evtrade/strategies/vectorized_base.py` + `evtrade/core/vectorized_engine.py` + `evtrade/core/engine.py`
  - 策略：`evtrade/strategies/channel_deviation.py` + `evtrade/strategies/ma_crossover.py`
  - 指标：`evtrade/indicators/{ema,atr,rsi,boll}.py`
  - spec：`openspec/specs/evtrade-architecture/spec.md`（4 改 2 删）
  - KB：6 份
  - 测试：`tests/test_strategy_unified.py` 大部分重写
- 受影响 public API（**破坏性**）：
  - 删：`VectorizedStrategy.compute_signals` / `compute_signals_for_one_bar`
  - 新：`VectorizedStrategy.step(state, bar, params) -> (state, sig)`（抽象）
  - 新：`VectorizedStrategy.init_state(params) -> state | None`（默认 `None`）
  - 改：`indicators.ema.ema_push` / `ema_current` / `ema_channel_push` / `ema_channel_current`
    → `ema_step(state, value, p)` / `ema_channel_step(state, h, l, p)`
  - 改：`indicators.atr.atr_push` / `atr_current` → `atr_step(state, h, l, c, p)`
  - 改：`indicators.rsi.rsi_push` / `rsi_current` → `rsi_step(state, close, p)`
  - 改：`indicators.boll.{sma,boll}_push` / `_current` → `{sma,boll}_step`
- 保留不变：`xp_ema` / `xp_ema_channel` / `xp_atr` / `xp_rsi` / `xp_bollinger`
（engine fast-path 用，策略不直接调）
- 性能：与现状等价
  - 批量路径 GPU 加速保留（engine fast-path 用 cupy 一次性算）
  - 逐 bar 路径 host Python 循环（与现状一致）
- 行为：bitwise 一致
  - `tests/test_strategy_unified.py::test_vectorized_vs_engine_on_bars_reconcile` 必须仍 PASS
  - e2e channel_deviation cagr/calmar/mdd/n_trades 与重构前一致

## Out of Scope

- 不动指标内部算法（EMA 递推公式、Wilder/SMA seed 等都不变）
- 不动 framework 的 reconcile 机制
- 不保留 `compute_signals` / `_for_one_bar` 旧 API（避免半新半旧；一次性破坏）
- 不动 `format_signal_line` / `get_extra_bucket_columns` / `get_extra_signal_columns` 展示 hook
- 不抽公共 helpers 到 `strategies/_common.py`（过早抽象；本 change 解决"算法/执行分层"，不下沉）
- 不动 `replay` / `sweep` / `permutation` 子模块（它们调 `run_vectorized` 即可，签名不变）

## Spec delta

新增 `openspec/changes/2026-09-10-strategy-step-only/specs/evtrade-architecture/spec.md`：

### Modified Requirements

- `Strategy interface contract`：抽象方法改 `step(state, bar, params) -> (state, sig)`；新增 `init_state(params) -> state | None`；策略无 instance state；删 `_for_one_bar` 段
- `Single contract = VectorizedStrategy.compute_signals`：标题改 `Single contract = VectorizedStrategy.step`；删"MUST NOT 拥有 ... `step()` 等旧 DSL 方法"反向条款；新增"state 由调用方持有传入"
- `Indicators are private` Scenario "策略 compute_signals 用 xp 算子"：改写为"策略 step 内部调 `compute_ema_channel(state, bar, p)` 拿 up/dw"
- Scenario "vectorized vs ref 引擎 bitwise 一致"：step 路径下断言

### Removed Requirements（DSL 时代死代码，spec 没同步删）

- `Requirement: state_spec is mandatory for DSL strategies`
- `Requirement: DSL→CUDA projection lives in strategies/dsl.py`

### Removed Scenarios

- `Scenario: 策略覆写 compute_signals_for_one_bar 维护 instance FSM`（被 step 取代）