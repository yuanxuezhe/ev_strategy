# 2026-09-10: dedupe-channel-deviation-fsm-helpers — 抽取 channel_deviation 共享 helper

## Why

`evtrade/strategies/channel_deviation.py` 192 行里，`compute_signals`（38 行，批量向量化）与 `compute_signals_for_one_bar`（40 行，逐 bar 增量）各自实现了同一份逻辑：

- 参数解析（`low1/low2/high1/high2/tf1`）
- 4 个偏离公式（`(dw - l) / dw * 100` 等）
- `_fsm_step` 调用与传参顺序

实测真重复 ~32 行，占文件总行数 **17%**。进一步让"重复显得合理"的元凶：

- `kbs/06-交易策略详解.md` §5 把两个方法完整代码各列了一遍（~50 行模板）
- `kbs/11-扩展指南.md` 行 47-78 用同一范式教新增策略

修代码不改 KB/KB 同步改掉，下次有人照 KB 抄代码会**复制一份新的重复**。

## What

`channel_deviation.py` 顶部抽 3 个 module-level helper（`_` 前缀，私有）：

| helper | 签名 | 职责 |
|---|---|---|
| `_parse_thresholds(params)` | `-> tuple[float, float, float, float]` | `(low1, low2, high1, high2)` 一次性解析 |
| `_compute_devs(xp, up, dw, h, l)` | `-> tuple[float, float, float, float] 或 xp.ndarray` | 4 个偏离。xp 数组算子版与标量算术版**签名同形**（传 numpy 时返标量，传 cupy 时返 ndarray）|
| `_run_fsm(state, ts, low_dev_h, low_dev, high_dev_l, high_dev, low1, low2, high1, high2)` | `-> int` | 薄包装 `_fsm_step`，把传参顺序定死，**避免两处调用错位** |

两个方法体精简后只剩"执行形态"差异（批量 GPU vs 逐 bar 增量），共同部分消失。

`kbs/06-交易策略详解.md` §5 + `kbs/11-扩展指南.md` 行 47-78 同步改：从"两个完整方法模板"改为"helper + 两个轻量执行形态"。

## Impact

- 受影响 files：
  - 策略：`evtrade/strategies/channel_deviation.py`（减 ~40 行）
  - KB：`kbs/06-交易策略详解.md` §5 + `kbs/11-扩展指南.md` 行 47-78
  - 测试：不改
- 不改 public surface：
  - `ChannelDeviationStrategy` 类签名不变
  - `compute_signals(xp, bars, params)` / `compute_signals_for_one_bar(xp, bar, params)` 签名不变
  - 行为不变（FSM 状态语义、bitwise 一致）
- 性能：纯 Python 函数调用开销 < 微秒级，相对 150ms 总耗时无感
- GPU 加速保留：批量路径仍调 `xp_ema_channel` + `xp.where`，GPU 加速不丢

## Out of Scope

- 不合并 `compute_signals` 与 `compute_signals_for_one_bar` 为一个方法（已 explore 论证会丢 GPU 加速）
- 不动 `ma_crossover.py`（无重复）
- 不动 `vectorized_base.py` 基类
- 不动 `format_signal_line` 展示 hook
- 不重构 FSM 状态语义（state 仍是 compute_signals 函数内局部、_for_one_bar 仍是 instance）
- 不抽公共 helper 到 `strategies/_common.py`（本次只去重 1 个策略文件；过早抽象成本不划算）

## Spec delta

本次 change **不改 `openspec/specs/evtrade-architecture/spec.md`**：

- `Strategy interface contract` Requirement 没限定策略代码组织形式（只规定方法签名）
- `_` 前缀 helper 属于策略内部实现细节，不属于 framework 契约
- 行为不变（信号 bitwise 一致由 `tests/test_strategy_unified.py::test_vectorized_vs_engine_on_bars_reconcile` 锁定）

delta 文件仅声明"No Requirements Changed"，保留 artifact 完整性。