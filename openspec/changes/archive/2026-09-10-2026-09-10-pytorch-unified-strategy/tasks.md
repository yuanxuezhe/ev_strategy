# Tasks: PyTorch 统一 CPU/GPU 策略算子

## 1. spec 落地
- [x] 1.1 改 `openspec/specs/evtrade-architecture/spec.md`（Requirements + Scenarios）
- [x] 1.2 `openspec validate --specs` 通过

## 2. backends 换 torch
- [x] 2.1 `evtrade/backends.py::get_xp(device) -> torch.device`
  - `"cpu"` → `torch.device("cpu")`
  - `"gpu"` → `torch.device("cuda")` (无 cupy 探测)
  - `"auto"` → cuda 可用则 cuda，否则 cpu
- [x] 2.2 `gpu_available() -> torch.cuda.is_available()`
- [x] 2.3 `evtrade/core/gpu.py`: 删 `_ensure_cupy()`，保留 `gpu_info()`（改用 torch API）
- [x] 2.4 `evtrade/core/capability.py::gpu_available` 委托 backends（thin wrapper）

## 3. indicators 改 torch + batch 维
- [x] 3.1 `evtrade/indicators/ema.py`:
  - `xp_ema(xp, values: (B, T), period: int | (B,)) -> (B, T)`
  - 用 cumsum + per-row period 切片实现
  - `xp_ema_channel(xp, highs, lows, period)` 同形态
  - `ema_push / ema_current` 增量版保留（实盘 step 内用）
- [x] 3.2 `evtrade/indicators/atr.py`: `xp_atr(xp, h, l, c, period) -> (B, T)`
- [x] 3.3 `evtrade/indicators/rsi.py`: `xp_rsi(xp, close, period) -> (B, T)`
- [x] 3.4 `evtrade/indicators/boll.py`: `xp_sma / xp_bollinger` 支持 batch 维
- [x] 3.5 `evtrade/indicators/__init__.py`: 重导出

## 4. 策略基类适配
- [x] 4.1 `VectorizedStrategy.compute_signals(xp, bars, params) -> torch.Tensor (B, T) int8`
- [x] 4.2 `compute_signals_for_one_bar` 默认 wrapper: `bar -> (1, 1) tensor -> compute_signals[0, 0]`
- [x] 4.3 `ChannelDeviationStrategy`: 适配 (B, T) 形态；内部用 `xp_ema_channel`
- [x] 4.4 `MACrossoverStrategy`: 适配 (B, T)

## 5. 引擎
- [x] 5.1 `Engine.__init__` 新增 `state` (B=1 torch tensor buffer)
- [x] 5.2 `Engine.on_bars` 桶 CLOSE 时维护滑动 buffer (1, T_lookback)
- [x] 5.3 `Engine._flush_final_bucket` 同样
- [x] 5.4 `VectorizedEngine.run_vectorized`: bars 构造成 `(1, T)` tensor
- [x] 5.5 `_execute_trades`: 改 torch tensor；cash/position 标量在 host
- [x] 5.6 `metrics.summarize`: 接受 torch tensor 输入（兼容 numpy 自动转换）

## 6. 依赖
- [x] 6.1 `pyproject.toml`: 删 `cupy-cuda11x/12x`，`dependencies` 加 `torch>=2.0`
- [x] 6.2 `[project.optional-dependencies]` gpu 段删 cupy；保留 gpu 段以备未来

## 7. CLI
- [x] 7.1 `cli.py`: `--device` choices 加 `"cuda"`（保留 cpu/gpu/auto 兼容）
- [x] 7.2 `sweep.py`: `device=gpu` 自动转 cuda

## 8. 测试
- [x] 8.1 替换 `tests/test_gpu.py` → `tests/test_torch_backend.py`
  - B=1 vs B=10 bitwise 一致 (cpu)
  - cupy/GPU 测试改 cuda 测试
- [x] 8.2 `tests/test_indicators_torch.py`: ema/atr/rsi/boll batch 维正确性
- [x] 8.3 `tests/test_strategy_unified.py`:
  - 验证 `compute_signals` 返回 (B, T)
  - B=1 vs B=2 (不同 params) bitwise 一致于各自 row
- [x] 8.4 `tests/test_vectorized.py`:
  - `run_vectorized` 返回 summary 字段集不变
  - reconcile (vectorized vs engine) PASS
- [x] 8.5 `uv run pytest -q` 通过（基线 ≥110 passed）

## 9. KB 同步
- [x] 9.1 新建 `kbs/15-PyTorch统一策略.md`：bars (B,T) 形态、双模式示例、变周期 EMA
- [x] 9.2 `kbs/02-系统架构.md`: 架构图改 torch 后端
- [x] 9.3 `kbs/06-交易策略详解.md`: 策略基类 + (B, T) 形态
- [x] 9.4 `kbs/12-重构与性能内核.md`: 写 torch 替换 cupy 段
- [x] 9.5 `kbs/14-统一策略契约.md`: bars 形状 (B, T) 契约
- [x] 9.6 `kbs/README.md`: 索引加 15

## 10. archive
- [x] 10.1 `openspec archive 2026-09-10-pytorch-unified-strategy`
