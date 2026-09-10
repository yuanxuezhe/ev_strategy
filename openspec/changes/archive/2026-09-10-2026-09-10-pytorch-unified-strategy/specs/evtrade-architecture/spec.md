# Spec Delta: PyTorch 统一 CPU/GPU 策略算子（替换 cupy）

> 本 spec 是 2026-09-10 `pytorch-unified-strategy` change 的 delta；落地后由 `openspec archive` 合并到 `openspec/specs/evtrade-architecture/spec.md`。

## MODIFIED Requirements

### Requirement: Strategy interface contract

策略 MUST 实现 `VectorizedStrategy.compute_signals(self, xp, bars: dict, params: dict) -> torch.Tensor`。
`xp` MUST 为 `torch.device`（framework 通过 `evtrade.backends.get_xp(device)` 传入；`"cpu"` / `"cuda"` / `"auto"` 三种字符串仍兼容）。
**pytorch-unified-strategy, 2026-09-10 变化**：cupy 已下线，CPU/GPU 统一走 PyTorch；
策略代码 MUST 仅使用 torch / 框架提供的算子，禁止 `import cupy`（已删除依赖）。

#### Scenario: 框架传入 torch.device
- **WHEN** `run_vectorized` / `Engine.on_bars` 调 `strategy.compute_signals(xp, bars, params)`
- **THEN** `xp` MUST 是 `torch.device`（不是 numpy/cupy 模块）

#### Scenario: bars 数组为 numpy 或 torch.Tensor
- **WHEN** 策略读 `bars["c"]` 等
- **THEN** 框架 MUST 接受 numpy ndarray 与 torch.Tensor 两种输入；策略 MUST NOT 假设特定后端

#### Scenario: 策略 compute_signals 签名不变
- **WHEN** 一个策略类继承 `VectorizedStrategy` 并实现 `compute_signals(self, xp, bars, params)`
- **THEN** framework MUST 按新签名调用；策略 body 内仍可调 `xp_ema(xp, c, p)` 等算子（保留向后兼容）

### Requirement: Indicators are private to strategies (framework MUST NOT compute any indicator)

框架 MUST NOT 预计算任何指标（EMA / ATR / RSI / Bollinger 等）。所有指标计算由策略在 `compute_signals` 或 `step` body 内通过 `evtrade/indicators/` 子包导出的 API 完成。
**2026-09-10 变化**：`evtrade/indicators/` 子包 MUST 提供三类算子：
1. **xp 版**（兼容旧 `xp_ema(xp, close, p)` 签名，`xp` 为 numpy/cupy/torch 模块；保留向后兼容）
2. **torch 版**（`xp_ema_torch(values: torch.Tensor, p) -> torch.Tensor`，新 PyTorch 后端路径用）
3. **step 增量版**（`ema_step(state, value, p)` 纯 Python 标量 in/out；策略 step() 用）
子包 MUST NOT import `cupy` / `numba`；MUST NOT 使用 `@njit`；MUST NOT 包含 CUDA C99 字符串。
`pyproject.toml` MUST NOT 声明 `cupy` 或 `numba` 为依赖；MUST 声明 `torch>=2.0`。

#### Scenario: indicators 提供 xp 版与 torch 版双形态
- **WHEN** 策略用 `from evtrade.indicators import xp_ema` 调用
- **THEN** `xp_ema(xp, c, p)` 兼容旧 numpy/cupy/torch 模块；`xp_ema_torch(c_torch, p)` 新 PyTorch 路径用

#### Scenario: 策略 step 用 ema_step 维护增量
- **WHEN** 策略覆写 `step(state, bar, params)` 维护 EMA 增量
- **THEN** MUST 用 `ema_step(EMAState, value, p)` 纯 Python 接口；与批量路径 `xp_ema` 信号轨迹 MUST bitwise 一致

### Requirement: CLI surface = `--device {cpu, gpu, auto}`

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto, cuda}`（默认 `auto`）；
`gpu` 与 `cuda` 同义（cupy 时代遗留的 `gpu` 字符串保留），`auto` 优先 cuda（cupy/torch 都已不用，device 由 PyTorch CUDA 自动探测）。
`evtrade.backends.get_xp("gpu")` 在 CUDA 不可用时 MUST fallback 到 cpu 并打 `RuntimeWarning`。

#### Scenario: 旧 --device gpu 仍接受
- **WHEN** `python -m evtrade backtest --device gpu ...`
- **THEN** 自动映射为 cuda；CUDA 不可用时打 RuntimeWarning 并继续跑 cpu

#### Scenario: PyTorch 后端端到端跑通
- **WHEN** `python -m evtrade backtest --device auto --strategy channel_deviation ...`
- **THEN** MUST 跑通并打印 26 字段盈亏汇总（含 cagr / sharpe / calmar / win_rate / profit_factor / baseline_max_dd / dd_excess / x_mdd 等）

## ADDED Requirements

### Requirement: PyTorch 统一后端 (NEW 2026-09-10)

`evtrade` MUST 使用 PyTorch 作为唯一 array 后端，CPU/GPU 透明路由由 `tensor.to(device)` 完成。
`evtrade.backends.get_xp(device) -> torch.device` 是统一入口。
`evtrade.core.gpu.py` 的 `cupy` 探测路径 MUST NOT 存在；`gpu_info()` MUST 改用 `torch.cuda` API。
`evtrade.core.capability.py::gpu_available()` MUST 委托 `torch.cuda.is_available()`。

#### Scenario: cupy 不再是可选依赖
- **WHEN** 用户执行 `grep -r "import cupy" evtrade/`
- **THEN** MUST 0 命中（cupy 已完全下线）

#### Scenario: torch 后端单测通过
- **WHEN** `pytest tests/test_torch_backend.py -v`
- **THEN** `get_xp` 返 `torch.device`；`gpu_available` 与 `torch.cuda.is_available()` bitwise 一致；`xp_ema_torch` 接受 (B, T) tensor + per-row period 返 (B, T)
