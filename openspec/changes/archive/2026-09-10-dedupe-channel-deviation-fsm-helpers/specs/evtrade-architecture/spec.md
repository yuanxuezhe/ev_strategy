# evtrade-architecture (delta)

> 本 spec 是 2026-09-10 `dedupe-channel-deviation-fsm-helpers` change 的 delta；
> 本次 change **不改 `openspec/specs/evtrade-architecture/spec.md`**。

## 决策依据

- `Strategy interface contract` Requirement（spec.md L14-16）仅规定策略方法签名
  `compute_signals(self, xp, bars, params) -> xp.ndarray[int8]`
  与可选覆写 `compute_signals_for_one_bar(self, xp, bar, params) -> int`，
  **未限定代码组织形式**（允许私有 helper / 模块级函数 / 子类复用）。
- 本次新增的 `_parse_thresholds` / `_compute_devs` / `_run_fsm` 均以下划线前缀
  为私有内部实现，不属于 framework 公共契约，spec 无需列名。
- 行为不变量（信号 bitwise 一致）由
  `tests/test_strategy_unified.py::test_vectorized_vs_engine_on_bars_reconcile`
  + `test_cpu_vs_gpu_*` 锁定，spec 已通过这两个测试间接保证。

## No Requirements Changed

- 无 Removed Requirements
- 无 Modified Requirements
- 无 Added Requirements

## 后续影响（不在本 change 范围）

- 未来若多策略复用同一组 helper（如 `_compute_devs` 对其他指标也适用），
  应新建 change 提到 `evtrade/strategies/_common.py` 或 `evtrade/strategies/helpers.py`，
  并相应更新 spec（明确"策略可调用 strategies 子包内 helpers"）。
  本次不在此范围。