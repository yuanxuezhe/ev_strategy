# Spec Delta — GPU-Batched Sweep

## ADDED Requirements

### Requirement: Optional GPU-batched sweep hook (`batched_step`)

`VectorizedStrategy` MUST 暴露一个可选 `@classmethod batched_step(cls, state, bars, params, *, n_combos, n_bars)` hook，作为 GPU 批量网格扫描的 opt-in 入口。基类默认实现 MUST raise `NotImplementedError`，表示策略未实现批量路径——`sweep()` 路由会自动 fallback 到 per-combo `ThreadPool` 路径。

实现该 hook 的策略应满足：
- `state` 为 dataclass，每个字段 MUST 为 `[N]` Tensor（`N = n_combos`），存储每 combo 的批量状态；
- `bars` MUST 为 dict[str, Tensor]，每个 value 形状 `[T]`（`T = n_bars`），键集 `{ts, o, h, l, c, v, mark}`；
- `params` MUST 为 dict[str, Tensor]，每个 value 形状 `[N]`，覆盖 `params_spec` 中声明的所有策略参数；
- 返回 `(new_state, sig)`：`sig` MUST 形状 `[N, T]` dtype `int8`，取值 ∈ {-1, 0, 1}；
- 浮点计算 MUST 用 `float64`，与 per-combo `step` 路径产出 bitwise 一致（或 float64 tolerance 1e-12 内）；
- `mark=0` 段处理 MUST 与原 `step` 语义一致（如 ma_crossover 仍推 EMA 累积、仅不产 sig）；
- 异常 MUST 立即向上抛（不延后到 sync point），保持 `test_sweep_does_not_crash_when_one_combo_fails` 语义。

`channel_deviation` MUST NOT 实现该 hook（FSM 锁存难向量化），`sweep()` 路由 MUST 自动跳过。

sweep 路由规则（`core/sweep.sweep()`）：
- `hasattr(cls, "batched_step") and cls.batched_step is not VectorizedStrategy.batched_step`（真覆写）；
- `device != "cpu"`；
- `len(combos) >= 32`（小网格 GPU 启动开销 > 收益）；
- `gpu_available()` 为 True。

任一条件不满足则走现有 per-combo ThreadPool 路径；`run_batched` 内捕获 `torch.cuda.OutOfMemoryError` 后自动 fallback 到 ThreadPool 并打 warning。`--workers` 在 batched 模式下 MUST 被忽略（不改 CLI 解析，打印 notice）。

#### Scenario: Hook default 未实现走 ThreadPool

- **WHEN** 策略类继承 `VectorizedStrategy` 但未覆写 `batched_step`
- **THEN** `hasattr(cls, "batched_step")` 为 True，调用 MUST raise `NotImplementedError`；`sweep()` 路由跳过 batched 路径，走现有 ThreadPool

#### Scenario: ma_crossover 真覆写 + 阈值满足走 batched

- **WHEN** `--strategy ma_crossover --device gpu --grid fast=3,5,10 --grid slow=20,60`（6 个 combos，< 32 个）OR `--device cpu`
- **THEN** `sweep()` 路由走 ThreadPool（device=cpu 路径；N=6 < 32 阈值）；ma_crossover 不实际跑 batched_step
- **AND WHEN** `--strategy ma_crossover --device gpu --grid fast=... --grid slow=...` 共 ≥ 32 个 combos 且 CUDA 可用
- **THEN** `sweep()` 路由走 batched 路径，调 `MACrossoverStrategy.batched_step` 一次产出 `[N, T]` sig

#### Scenario: batched_step 输出与 per-combo step 循环 bitwise 一致

- **WHEN** 同一 bars + 同一组 params 同时走 `MACrossoverStrategy.batched_step`（1 次调用）和 per-combo `MACrossoverStrategy.step` 循环
- **THEN** 产出的 sig 数组在 float64 tolerance 1e-12 内 bitwise 一致（`np.array_equal` 通过）；trades / summary 与 per-combo 路径走同一 `_execute_trades` 产出

#### Scenario: 无 CUDA / CUDA OOM / 小网格 fallback 到 ThreadPool

- **WHEN** 任一条件命中：`gpu_available() is False`、CUDA OOM at runtime、`len(combos) < 32`、`device == "cpu"`、策略未覆写 hook
- **THEN** sweep MUST 走 ThreadPool 路径，CSV 输出与纯 ThreadPool 路径 bitwise 一致；OOM 情况打印 fallback warning
