# Proposal: PyTorch 统一 CPU/GPU 策略算子（替换 cupy）

## Why

`evtrade` 当前 CPU 走 numpy / GPU 走 cupy 双后端。每个指标 `ema/atr/rsi/boll` 既要写
numpy 版又要写 cupy 版（即使语法相近），且变周期 batched EMA 用 `searchsorted+reduceat`
笨拙。cupy 探测需要补 DLL 路径，维护成本高。

**目标**：用 PyTorch 单一库替换 cupy，统一 CPU/GPU/批量/实盘四种场景：

1. CPU/GPU 透明：`tensor.to(device)` 一行切换
2. **变周期 batched EMA**：`unfold + gather` 支持 per-row period
3. **实盘/批量代码同源**：`compute_signals(xp, bars, params)` 接受 `(B, T)` 形状，
   B=1 实盘 / B=N 网格扫描用同一份策略代码

## What Changes

### A. backends 换 torch
- `evtrade/backends.py::get_xp(device) -> torch.device`
  - `"cpu"` → `torch.device("cpu")`
  - `"gpu"` → `torch.device("cuda")`
  - `"auto"` → cuda 可用则 cuda，否则 cpu
- 删 `evtrade/core/gpu.py::_ensure_cupy` 与 cupy DLL 探测
- `gpu_available() -> torch.cuda.is_available()`
- `gpu_info()` 改读 `torch.cuda` API

### B. indicators 重写为 torch 算子 + batch 维
- `xp_ema(xp, values, period)` 形状 `(B, T) -> (B, T)`，period 支持 `(B,)`
  - 实现：`torch.cumsum` + 索引切片 + per-row period gather
- `xp_ema_channel(xp, highs, lows, period)` 同上
- `xp_atr / xp_rsi / xp_bollinger` 同样支持 batch 维
- 纯 Python 增量版 `ema_push / ema_current` 保留（B=1 实盘 step 内用）

### C. 策略基类适配 (B, T) 形状
- `VectorizedStrategy.compute_signals(xp, bars, params)`:
  - `bars["c"]` 等数组形状 `(B, T)`
  - 返回 `(B, T) int8`
- `compute_signals_for_one_bar(xp, bar, params) -> int`:
  - 内部 `bar = {k: torch.tensor([v]).unsqueeze(0) for k, v in bar.items()}` → `(1, 1)`
  - 调 `compute_signals`，取 `[0, 0]`

### D. 引擎双模式共享 compute_signals
- **Engine.on_bars (实盘/逐桶)**：维护 `(1, T_lookback)` 滑动 buffer，
  桶 CLOSE 时把新 OHLCV 拼到末尾 → 调 `compute_signals` → `signal[0, -1]`
- **VectorizedEngine.run_vectorized (批量)**：构造 `(1, T)` bars → `compute_signals` →
  撮合（`_execute_trades` 改 torch tensor，但 cash/position 标量在 host）

### E. 删 cupy 依赖，加 torch
- `pyproject.toml`: 删 `cupy-cuda11x/12x`；`dependencies` 加 `torch>=2.0`
- `gpu` extra 仍保留，语义改为"装 CUDA 后端"，但 torch 一行切换 cuda 即可，
  不再需要 cupy 风格独立 wheel

### F. CLI / sweep
- `--device {cpu, gpu, auto}` → 内部映射 `gpu → cuda`，仍接受旧字符串
- `sweep.run_one_vectorized` 适配 B=N（grid 维度）—— 暂不实施 batched grid（B=1 × N 次串行）；
  留 TODO
- 旧 `--engine {kernel, ref, vectorized}` 兼容层删除（2026-09-09 已下线）

## Impact

- 受影响 capability：`evtrade-architecture` (3 Requirements 改写 + 2 删除)
- 受影响 files（核心 8 个）：见 design
- 受影响 public API（破坏性）：
  - `bars` 数组形状 `(T,)` → `(B, T)`
  - `xp` 类型 `numpy/cupy.ndarray` → `torch.Tensor`
  - `get_xp("gpu")` 返回 `torch.device("cuda")`
- 受影响下游：
  - `evtrade/cli.py` 仅 device 字符串映射调整
  - `evtrade/core/sweep.py` 不变（仍 B=1）
  - 测试全部需要 import 改为 torch；核心算法测试可保留 numpy 断言（CPU 端 bitwise）

## Out of Scope

- **batched grid sweep** (B=N 参数网格单次调用)：本 change 不实施，留 TODO
- **state 增量 step 接口** (避免 B=1 实盘每次重算窗口)：本 change 不实施
- **CPU/GPU 数值精度差异** 兜底：tolerance 设 1e-5

## Verification

- `uv run python -c "import torch; print(torch.__version__)"` 正常
- `uv run pytest tests -q` 维持基线（110+ passed；cupy 测试替换为 torch 测试）
- `python -m evtrade backtest --device cpu` 跑通
- `python -m evtrade backtest --device cuda` 跑通（CUDA 不可用 fallback）
- `python -m evtrade replay --log ... --device cpu --against-ref` PASS

## Risk

1. **PyTorch 包大小**：torch CPU ~200MB / CUDA ~2GB。`pyproject.toml` 单依赖会膨胀
   短期可接受；长期拆 `[cpu]` / `[gpu]` extras
2. **变周期 batched EMA 性能**：`unfold+gather` 复杂度 O(B·T·P_max)
   实测 vs cupy reduceat：可能略慢；GPU 内存足够时摊销
3. **Engine.on_bars 滑动窗口重算**：B=1 每次回调 O(T) 重算 EMA 全序列
   实盘低延迟场景可能卡；短期接受，长期做 state 增量接口
