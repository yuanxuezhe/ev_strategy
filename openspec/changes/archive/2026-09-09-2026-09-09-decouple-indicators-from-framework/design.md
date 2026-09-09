# Design: 框架不假定任何指标

## 1. 架构概览

### 1.1 三层职责重划

| 层 | 旧职责 | 新职责 |
|---|---|---|
| `core/engine.py` `Engine.on_bars` | 算 EMA → 调 strategy.check | 只调 strategy.check(cur, indicators)；EMA 由 strategy 自己算 |
| `core/kernel.py` `KernelState.step` | 维护 up_sum/up_ema 等 6 字段 + 调 _ema_push | 只调 strategy_check(st, indicators)；EMA 由 strategy DSL body 自己维护 |
| `core/gpu.py` CUDA 模板 | 内联 EMA push/current | 不再含 EMA 逻辑；strategy DSL body 内联 push/current 或调 __device__ helper |
| `indicators/` | 纯函数批量版 (jupyter 用) | **增量版（@njit，DSL 可调用）+ 批量版** 双接口 |
| `strategies/dsl.py` | 白名单 = {min,max,abs} | 白名单 += indicators 所有 *_push / *_current |
| `strategies/channel_deviation.py` | 用 ctx.up / ctx.dw (framework 算好) | state_spec 加 up_st/dw_st；DSL body 调 ema_channel_push / ema_channel_current |

### 1.2 数据流（EMA 责任从 framework 下放到 strategy）

**旧**：
```
Engine.on_bars → ema_ch.channel(h,l) → (up,dw) → strategy.check(cur, up, dw)
```

**新**：
```
Engine.on_bars → strategy.check(cur, {})
  └─ strategy DSL body (Python runner / numba njit / CUDA __device__):
       ctx.up_st, ctx.dw_st = ema_channel_push(ctx.up_st, ctx.dw_st, ctx.cur_high, ctx.cur_low, ctx.p_N)
       ctx.up, ctx.dw = ema_channel_current(ctx.up_st, ctx.dw_st, ctx.cur_high, ctx.cur_low, ctx.p_N)
       # 后续信号逻辑不变
```

### 1.3 incremental state 形状

EMA 通道的双 rail 增量 state 设计为 Python `@dataclass`（Python 端）和 numba `namedtuple`（numba 端）：

```python
# Python 端
@dataclass
class EMAState:
    sum: float = 0.0
    count: int = 0
    ema: float = 0.0  # 用 inf 表示未就绪

@dataclass
class EMAChannelState:
    up: EMAState
    dw: EMAState
```

numba 端由于不支持 dataclass，使用 `UniTuple` 嵌套。**接口契约**：`ema_channel_push(st, h, l, p) -> st`（state 在 numba 端是 namedtuple，按值返回）。三端接口同形，函数名一致。

### 1.4 DSL→三端转译**

`strategies/dsl.py::_CALL_WHITELIST` 增加：
```
ema_push, ema_current, ema_channel_push, ema_channel_current,
atr_push, atr_current, rsi_push, rsi_current,
boll_push, boll_current, sma_push, sma_current
```

- **Python 端**：白名单函数在 `strategies/dsl.py` 内 import 后直接 `exec` 原 DSL body
- **numba 端**：白名单函数在 `core/kernel_dsl.py::_build_dsl_kernel_impl` 内 import 后直接调 `@njit` 函数（已通过 `from evtrade.indicators.ema import ema_channel_push`）
- **CUDA 端**：白名单函数在 `strategies/dsl.py::render_cuda_device_function` 内转换为对应的 `__device__` 调用（C++ inline 实现）

## 2. 数据契约

### 2.1 删除的 framework 符号

| 符号 | 位置 | 替代方案 |
|---|---|---|
| `IncrementalEMA` | `core/incremental_indicators.py` | `indicators.ema.ema_push` + `ema_current` |
| `EMAChannel` | `core/incremental_indicators.py` | `indicators.ema.ema_channel_push` + `ema_channel_current` |
| `ema` / `ema_channel` (incremental_indicators 版) | `core/incremental_indicators.py` | `indicators.ema.ema` / `ema_channel` (批量版，保留) |
| `tf1` 参数 | `Engine.__init__` / `KernelState.__init__` / `replay_kernel` / `replay_engine` / `sweep.GRID_KEYS` / `config.TF1` | 策略 `params_spec` 自声明 |
| EMA 6 字段 | `KernelState` | 策略 `state_spec` 自声明 |
| `_ema_push` / `_ema_current` | `core/kernel.py` | `indicators.ema.ema_push` / `ema_current`（@njit） |

### 2.2 新增的 indicators 增量 API

```python
# evtrade/indicators/ema.py (新增)
@njit(cache=True)
def ema_push(s_sum: float, s_count: int, s_ema: float, value: float, p: int) -> Tuple[float, int, float]:
    """EMA 增量推入;返回新 state (sum, count, ema)"""

@njit(cache=True)
def ema_current(s_sum: float, s_count: int, s_ema: float, p: int, pending: float) -> float:
    """EMA 当前值;未就绪返回 inf (numba 不支持 None)"""

@njit(cache=True)
def ema_channel_push(up_sum, up_count, up_ema, dw_sum, dw_count, dw_ema, h, l, p):
    """EMA 通道增量推入"""

@njit(cache=True)
def ema_channel_current(up_sum, up_count, up_ema, dw_sum, dw_count, dw_ema, h, l, p):
    """EMA 通道当前值 -> (up, dw)"""
```

ATR / RSI / Boll / SMA 类似（设计实现细节见 tasks.md）。

### 2.3 顶层 API 变更

`evtrade/__init__.py`：
- 删除：`EMAChannel` / `IncrementalEMA` / `ema` / `ema_channel`（来自 incremental_indicators.py）
- 保留：`ema_fn`（来自 indicators.ema.py，纯函数批量版）/`atr` / `rsi` / `bollinger` / `sma` / `true_range`
- 新增：`ema_push` / `ema_current` / `ema_channel_push` / `ema_channel_current` 等增量 API
- 删除 shim：`sys.modules.setdefault("evtrade.incremental_indicators", ...)` 和 `sys.modules.setdefault("evtrade._incremental_indicators", ...)`

## 3. 迁移路径

### 3.1 策略层

`strategies/channel_deviation.py`：

```python
# state_spec 加 EMA 增量状态 (state_spec 默认值同时承担 numba/KernelState 的初值)
state_spec = {
    ...,
    "up_st_sum":  {"type": float, "default": 0.0},
    "up_st_count": {"type": int,   "default": 0},
    "up_st_ema":  {"type": float, "default": float("inf")},
    "dw_st_sum":  {"type": float, "default": 0.0},
    "dw_st_count": {"type": int,   "default": 0},
    "dw_st_ema":  {"type": float, "default": float("inf")},
}

# params_spec 加 tf1 (作为策略私有参数)
params_spec = {
    ...,
    "tf1": {"default": 21, "type": int, "min": 2, "max": 1000},
}

# DSL body 第一段: 维护 EMA 增量状态 + 取当前通道值
_CHANNEL_DEVIATION_DSL = """
# === EMA 通道增量更新 (DSL 可调用) ===
ctx.up_st_sum, ctx.up_st_count, ctx.up_st_ema, ctx.dw_st_sum, ctx.dw_st_count, ctx.dw_st_ema = ema_channel_push(
    ctx.up_st_sum, ctx.up_st_count, ctx.up_st_ema,
    ctx.dw_st_sum, ctx.dw_st_count, ctx.dw_st_ema,
    ctx.cur_high, ctx.cur_low, ctx.p4
)
ctx.up, ctx.dw = ema_channel_current(
    ctx.up_st_sum, ctx.up_st_count, ctx.up_st_ema,
    ctx.dw_st_sum, ctx.dw_st_count, ctx.dw_st_ema,
    ctx.cur_high, ctx.cur_low, ctx.p4
)
# === 桶切换清锁 ===
...
"""
```

### 3.2 Framework 层

`core/engine.py` `Engine.on_bars`：直接调 `strategy.check(cur, {})`，删 EMAChannel / _sync_ema / tf1 字段。

`core/kernel.py` `KernelState`：删 6 EMA 字段；删 _ema_push / _ema_current；step() 删 EMA 内联；make_state_general 删 tf1 参数。

`core/gpu.py` CUDA 模板：删 tf1s / k_ema / 6 register / 内联 push-current；template 不变。

`core/replay.py`：`replay_kernel` / `replay_engine` / `reconcile` 删 tf1。

`core/sweep.py`：GRID_KEYS 删 tf1。

`core/config.py`：删 TF1 常量。

### 3.3 Test 适配

- `tests/test_differential.py`：删 up/dw 数组断言；`test_differential_tf1` 改用 `params["tf1"]`；保留 signal/trade/cash bitwise 断言
- `tests/test_replay.py`：删 tf1
- `tests/test_kernel_unit.py`：删 `_ema_push` / `_ema_current` 直测；新增 `indicators.ema.ema_push` / `ema_channel_push` 单测
- `tests/test_dsl_cuda.py`：保留通用 CUDA 模板测试（按策略自管指标路径）

## 4. 风险与回退

### 4.1 风险

- **bitwise 一致性**：incremental EMA 公式必须在三端字面一致（同一 k=2/(p+1)、同一 SMA seed）
- **numba state 形状**：incremental state 必须能用 numba `UniTuple` 表达（不用 dataclass）
- **CUDA 端 state 类型**：必须能用 C 标量表达（不能用 Python 对象）

### 4.2 回退

- 删除 `core/incremental_indicators.py` 是不可逆的（git revert 即可）
- 框架解耦后若三端 bitwise 一致性破，git revert 单个 commit
- spec delta 同步：落地前 `openspec validate --specs` 必须通过；落地后 `openspec archive`

## 5. 验证策略

```bash
# 1. spec 校验
openspec validate --specs
openspec validate 2026-09-09-decouple-indicators-from-framework

# 2. KB ↔ 代码一致性 (CLAUDE.md §6)
grep -rn "IncrementalEMA\|EMAChannel\|_ema_push\|_ema_current\|up_sum\|up_count\|up_ema\|dw_sum\|dw_count\|dw_ema\|tf1\|k_ema" evtrade/core/   # 必须 0 命中

# 3. 测试
uv run pytest -q                       # 全部通过
uv run pytest -q tests/test_differential.py -k channel_deviation   # bitwise 一致
uv run pytest -q tests/test_kernel_unit.py -k "ema_push or ema_current"
uv run pytest -q tests/test_dsl_cuda.py

# 4. 三引擎端到端 (无 GPU 跳过)
python -m evtrade backtest --engine ref    --strategy channel_deviation --params "tf1:21;..." --code XXX
python -m evtrade backtest --engine kernel --strategy channel_deviation --params "tf1:21;..." --code XXX
# 两者输出应 bitwise 一致
```