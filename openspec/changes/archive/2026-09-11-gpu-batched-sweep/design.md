# Design — GPU-Batched Sweep

## Architecture

```
sweep(bars, base, combos, ..., device, n_workers, ...)
   │
   ├─ 设备解析 (resolve_device)
   │
   ├─ cls = get_strategy_class(strategy_name)
   │
   ├─ use_batched = (
   │      hasattr(cls, "batched_step")
   │      and cls.batched_step is not VectorizedStrategy.batched_step
   │      and device != "cpu"
   │      and len(combos) >= 32
   │      and gpu_available()
   │  )
   │
   ├─ if use_batched:
   │      try:
   │          metrics = run_batched(...)        # NEW: 1 次 batched_step 出 [N,T] sig
   │      except torch.cuda.OutOfMemoryError:
   │          print("[batched] CUDA OOM -> ThreadPool fallback")
   │          metrics = _threadpool_sweep(...)
   │  else:
   │      metrics = _threadpool_sweep(...)      # 现有路径, 重命名
   │
   └─ _finalize_sweep_df(metrics, combos, wins) # 共享, CSV 列集 bitwise 不变
```

## Hook contract

```python
# evtrade/strategies/vectorized_base.py
@classmethod
def batched_step(cls, state, bars, params, *, n_combos, n_bars):
    """opt-in GPU 批量 hook; 默认未实现, sweep 自动走 ThreadPool 路径"""
    raise NotImplementedError
```

| 参数 | shape | 说明 |
|---|---|---|
| `cls` | — | 类方法, 无 self 耦合 |
| `state` | dataclass 字段 `[N]` Tensor | 每 combo 一份 batched state |
| `bars` | dict[str, Tensor] `[T]` | `{"ts","o","h","l","c","v","mark"}` 全部 1-D |
| `params` | dict[str, Tensor] `[N]` | 每 combo 一份 param |
| `n_combos`, `n_bars` | int | shape 元数据 |
| 返回 | `(state, sig [N, T] int8)` | sig 第一维 N（combo），第二维 T（bar） |

## State dataclass: MABatchedState

```python
@dataclass
class MABatchedState:
    fast_sum: Tensor  # [N] float64
    fast_count: Tensor  # [N] int64
    fast_ema: Tensor  # [N] float64
    slow_sum: Tensor  # [N] float64
    slow_count: Tensor  # [N] int64
    slow_ema: Tensor  # [N] float64
    prev_diff: Tensor  # [N] float64
    has_prev: Tensor  # [N] bool (uint8 内部)
```

跟现有 `MACrossoverState` **并存**，batched 路径 opt-in。

## torch_ema kernel

```python
def torch_ema(values: torch.Tensor, p: int) -> torch.Tensor:
    """[T] -> [T] float64; 前 p-1 个 0.0 (跟 ema_step 一致); device/dtype 保留
    跟 numpy ema() 参考版 float64 bit-equal (无 vmap, 按 p 调用)"""
    out = torch.zeros_like(values, dtype=torch.float64)
    cumsum = torch.cumsum(values, dim=0)
    out[p-1] = cumsum[p-1] / p
    k = 2.0 / (p + 1.0)
    # 递推尾段: out[i] = values[i]*k + out[i-1]*(1-k), i in [p, T)
    ...
```

按 `tf1` 值分组调用（`tf1` 整数取值集小，分组后每组一次 kernel）。

## Routing rules

| 条件 | 路由 |
|---|---|
| `cls.batched_step is VectorizedStrategy.batched_step`（未覆写） | ThreadPool |
| `device == "cpu"` | ThreadPool |
| `len(combos) < 32` | ThreadPool（小网格 GPU 启动开销 > 收益） |
| `not gpu_available()` | ThreadPool（`device=auto` 降级 cpu） |
| CUDA OOM at runtime | ThreadPool fallback + warning |
| 其它 | `run_batched` |

`--workers` 在 batched 模式下打印忽略 notice，CLI 解析不变。

## Equivalence invariant

batched_step 输出 sig 跟 per-combo step 循环 bit-equal（`np.array_equal`）：

- 全部 float64
- EMA 累积用同一 `cumsum/p` seed + 递推尾段（跟 numpy 参考同公式）
- mark=0 段复刻 ma_crossover 原 step 语义（推 EMA、不产 sig）
- 双线同时就绪检查、首桶记 prev_diff 不产 sig、cross 检测（跟原 step 同公式）

测试 `tests/test_batched_sweep.py::test_ma_crossover_batched_step_equivalence` 锁。

## Failure modes

| 失败 | 处理 |
|---|---|
| batched_step 抛 RuntimeError | `run_batched` 立即透传（不延后），`sweep` 同步抛 |
| CUDA OOM | try/except 回退 ThreadPool + warning |
| batched_step 输出 shape 不对 | runtime assertion（不靠 type check） |
| bars 长度 < max(p_fast, p_slow) | 仍能跑（EMA seed 阶段全 0，sig 全 0） |

## Files

修改：
- `evtrade/strategies/vectorized_base.py`（+默认 hook）
- `evtrade/strategies/ma_crossover.py`（+MABatchedState + batched_step）
- `evtrade/indicators/ema.py`（+torch_ema）
- `evtrade/core/sweep.py`（routing 分支 + 抽出 _finalize_sweep_df）

新增：
- `evtrade/core/batched_sweep.py`（run_batched）
- `tests/test_batched_sweep.py`（11 个测试）

复用（不改）：
- `vectorized_engine.py::_aggregate_buckets`
- `vectorized_engine.py::_execute_trades`（per-combo 调）
- `vectorized_engine.py::_compute_signals`（per-combo path 不动）
- `backends.py::{resolve_device, gpu_available, to_tensor, to_host}`
- `_defaults_loader.py::save_best_from_sweep`
- `sweep.py::{parse_grid, _neighbor_decay, _pareto_flag, _ann_net}`
- `permutation.py::permutation_test`
- `metrics.py::summarize`
