# GPU-Batched Parameter Sweep (Optional `batched_step` Hook)

## Why

`evtrade` sweep 当前走 CPU `ThreadPoolExecutor`，N 组参数各跑一遍 `run_vectorized` → `strategy.step` Python 循环。加速上限受核数限制（4-32 倍）。

GPU 真正的高价值维度是**参数间并行**：N_combos 组共享同一份 bars，作为 torch batch dim 在 1 次 kernel launch 内出 N 份信号——比 CPU 多进程快 1-2 个数量级。

当前架构**显式不实施**这层批量：
- `openspec/changes/archive/2026-09-10-2026-09-10-pytorch-unified-strategy/proposal.md` L74: "**batched grid sweep** (B=N 参数网格单次调用)：本 change 不实施, 留 TODO"
- `kbs/15` §3: "批量扫描的 'B 维' 不在单次 vectorized 调用内部：sweep 层 (core/sweep.py, ThreadPoolExecutor) 对每个参数组合各跑一次 run_vectorized"

本 change 把这个 TODO 实现掉，**最小第一刀**：
- 加一个**可选** `batched_step` hook 到 `VectorizedStrategy`；
- `ma_crossover`（纯 EMA、无 FSM）实现之、走 GPU 批量；
- `channel_deviation`（FSM 锁存）不实现、继续 CPU 多进程；
- 两条路径并存，输出格式 / score / reconcile / 现有测试**bitwise 不变**；
- 第一刀只加速**信号生成**，成交执行仍逐 combo 走现有 `_execute_trades`。

## What changes

### 新增 capability

- `VectorizedStrategy.batched_step`（默认 `NotImplementedError`）—— 路由检测点
- `MACrossoverStrategy.MABatchedState` dataclass（`Tensor[N]` 字段）+ `batched_step` 实现
- `indicators.ema.torch_ema`（1-D batched EMA，float64 bit-equal numpy 参考）
- `core.batched_sweep.run_batched`（编排）
- `core.sweep.sweep()` routing 分支（device/cuda/n_combos 阈值判断）
- spec 新 Requirement + 4 Scenario
- KB §7.4（kbs/15）+ perf note（kbs/12）+ 用法句（kbs/13）+ --device 描述（kbs/10）
- 新增 `tests/test_batched_sweep.py`

### 不变

- `VectorizedStrategy.step` / `init_state` / `_resolve_params` / `register_strategy`
- `run_vectorized` 签名（spec R10 MUST NOT 带 device）
- `MACrossoverState`（原 per-combo state dataclass 不动）
- `channel_deviation` 任何代码
- `_execute_trades` / `trade_decision`（per-combo Python 循环保持）
- spec R7 "bitwise CPU/GPU"、R10 "run_vectorized 不带 device"（已有约束不变）
- `reconcile` 路径（仍走 `run_vectorized`）
- 所有现有测试（`test_vectorized.py` / `test_strategy_unified.py` / `test_sweep_*.py` / `test_assembly_coherence.py` 等）

## Impact

- **新增 GPU 路径**（ma_crossover）：大网格（≥32 combos）+ 有 CUDA → 1 次 batched_step 出 `[N, T]` 信号
- **路由表**：sweep 设备解析后判断 `use_batched`，命中走 `run_batched`，否则现有 ThreadPool
- **失败兜底**：CUDA OOM 自动回退 ThreadPool + 提示
- **测试**：新增 11 个测试（hook 默认 / 等价性 / mark=0 语义 / 路由 / 输出格式 / 异常传播 / fallback 等）
- **KB / spec 同步**：按 CLAUDE.md §0 同步演进

## Out of Scope（留给后续 change）

- `_execute_trades` 向量化（per-combo cash/position/cur_qty → `[N]` Tensor）—— 改动面大，独立 change
- `channel_deviation` FSM 向量化（latch 难 tensor 化）—— 独立评估
- `vmap` over `tf1` —— 第一刀按 tf1 值分组够用，后续可优化
- 新策略实现 `batched_step` —— 各策略作者按需添加
