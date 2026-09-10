# 15 PyTorch 统一策略（pytorch-unified-strategy, 2026-09-10）

> 回答的问题：**CPU/GPU/批量回测/实盘逐棒** 如何用**同一套策略代码**统一表达。
> 本文档覆盖 PyTorch 后端的双形态算子、bars 契约（B/T 维）、批量 vs 实盘的策略接口。

---

## 1. 背景

### 1.1 重构前的问题（cupy 时代）

| 痛点 | 表现 |
|---|---|
| 双份算子维护 | 每个指标（ema/atr/rsi/boll）既要 numpy 版又要 cupy 版 |
| 变周期 batched EMA 笨拙 | `xp_ema_channel` 用 `searchsorted+reduceat`，B 维 + per-row period 表达困难 |
| 实盘/批量代码双轨 | `step(state, bar, params)` 与 `compute_signals(xp, bars, params)` 算法相同但循环结构不同 |

### 1.2 目标

用 **PyTorch 作为唯一 array 后端**，统一：

- **CPU 回测**：`tensor.to("cpu")`
- **GPU 回测**：`tensor.to("cuda")`
- **批量扫描**：B 维扩展（同时跑 N 组参数网格）
- **实盘逐棒**：B=1 滑动窗口

PyTorch 单库同时提供：

- CPU/GPU 透明
- `unfold + gather` 支持每行变长卷积核 → 变周期 batched EMA
- `torch.where / cumsum / conv1d / clamp` 已是矢量化一等公民

---

## 2. 双形态算子

`evtrade/indicators/` 提供三类算子，按调用方选择：

| 形态 | 函数签名 | 适用场景 |
|---|---|---|
| **xp 版**（保留兼容） | `xp_ema(xp, values, p)` | 兼容旧 numpy 模块签名；策略 body 内 `xp_ema(np, c, p)` 直接用 |
| **torch 版**（新 PyTorch 后端） | `xp_ema_torch(values, p)` | PyTorch 路径；输入输出都是 `torch.Tensor`；支持 (B, T) + per-row period |
| **step 增量版** | `ema_step(state, value, p)` | 策略 step() 用，标量 in/out |

### 2.1 xp 版

```python
import numpy as np
from evtrade.indicators import xp_ema

values = np.random.randn(100)
ema = xp_ema(np, values, p=21)
# ema 形状 (100,), 前 20 根 NaN, 之后递推
```

兼容：

- `xp = numpy` — CPU
- `xp = cupy` — 已下线（cupy 已从依赖中删除）
- `xp = torch` — PyTorch 模块名（不推荐；用 `xp_ema_torch` 更直接）

### 2.2 torch 版

```python
import torch
from evtrade.indicators import xp_ema_torch

# 单线 (T,)
v1d = torch.randn(100, dtype=torch.float64)
ema1 = xp_ema_torch(v1d, p=21)   # 形状 (100,)

# 批量 (B, T)
v2d = torch.randn(10, 100, dtype=torch.float64)
ema_batch = xp_ema_torch(v2d, p=21)   # 形状 (10, 100)

# 变周期 (B,) per-row period
periods = torch.tensor([5, 21, 50, 60, 10, 8, 13, 100, 30, 7])  # 10 行不同 period
ema_var = xp_ema_torch(v2d, p=periods)   # 形状 (10, 100); 每行按各自 period
```

### 2.3 step 增量版

```python
from evtrade.indicators.ema import EMAState, ema_step

state = EMAState()
for v in values:
    state, ema = ema_step(state, v, p=21)
# state 由 engine 持有, 跨调用持续
```

---

## 3. bars 契约（B 维 + T 维）

引擎在 `_compute_signals` 内逐桶调 `strategy.step(state, bar, params)`；
桶级 bars dict（`_aggregate_buckets` 产出）：

```python
bars = {
    "ts":    tensor (B, T) int64,     # 桶时间戳 (14 位整数)
    "o":     tensor (B, T) float32,    # 每桶 open (首根 1m bar)
    "h":     tensor (B, T) float32,
    "l":     tensor (B, T) float32,
    "c":     tensor (B, T) float32,
    "v":     tensor (B, T) float32,
    "mark":  tensor (B, T) int8,       # 1=策略期, 0=预热段
    "n_bars": tensor (B,) int64,       # 每行桶数
}
返回: 每桶一次 step → sig ∈ {-1, 0, 1} (引擎聚合成 (B, T) int8)
```

### 3.1 B 维语义

| B | 场景 | engine |
|---|---|---|
| **1** | 单线回测 / 实盘逐棒 | Engine.on_bars 每桶调一次 |
| **N** | 网格扫描（N 组参数） | VectorizedEngine 一次性传 (N, T) bars，撮合也矢量化 |

策略代码**只写一份**，不需要 `if B == 1`。

### 3.2 T 维

时间序列长度。Engine.on_bars 维护 `(1, T_lookback)` 滑动 buffer；VectorizedEngine 一次性传 `(1, T)` 全部历史。

---

## 4. 批量 vs 实盘：同代码

### 4.1 批量 GPU 网格回测

```python
import torch
from evtrade.indicators import xp_ema_torch

T = 100_000  # 10 万根 5m 桶
B = 5_000    # 5000 组参数

# (1, T) 桶 close; (B, T) 广播
close_1m = torch.randn(1, T, dtype=torch.float64)
batch_close = close_1m.expand(B, -1)

# 网格参数 (B,) per-row period
tf1_grid = torch.randint(5, 60, (B,), dtype=torch.int64)

# 一次算整段 EMA
ema = xp_ema_torch(batch_close, p=tf1_grid)   # (B, T)

# 批量撮合（_execute_trades 支持 (B,)）
# 引擎逐桶循环: state, sig = strategy.step(state, bar_i, {"tf1": tf1_grid[i]})
```

### 4.2 实盘 / 单线回测（逐棒）

```python
import torch
from evtrade.indicators import EMAState, ema_step
from evtrade.strategies import get_strategy

strat = get_strategy("channel_deviation", low1=1.5, low2=1.0)
state = strat.init_state(strat.params)  # 引擎持有

# 每根 5m 桶 CLOSE 时调一次
def on_bucket_close(bar: dict):
    global state
    state, sig = strat.step(state, bar, strat.params)
    if sig == 1:
        broker.buy(...)
    elif sig == -1:
        broker.sell(...)
```

`step` 内调用 `ema_step` 维护 EMA 增量；信号轨迹与批量路径 bitwise 一致。

---

## 5. CPU/GPU 透明路由

### 5.1 `get_xp(device)`

```python
from evtrade.backends import get_xp

xp_cpu = get_xp("cpu")          # torch.device("cpu")
xp_cuda = get_xp("cuda")        # torch.device("cuda") if available, else cpu + warning
xp_auto = get_xp("auto")        # cuda 可用则 cuda, 否则 cpu
xp_gpu = get_xp("gpu")          # 别名 = cuda
```

CUDA 不可用时 fallback 到 cpu + `RuntimeWarning`。

### 5.2 策略代码不应直接判断 device

```python
# 错 (耦合 torch.cuda)
if torch.cuda.is_available():
    values = values.cuda()

# 对 (框架层处理)
state, sig = strategy.step(state, bar, params)   # 引擎循环调用, 不感知 device
```

---

## 6. 策略契约示例（step 唯一入口 + torch 批量算子）

```python
import torch
from dataclasses import dataclass, field
from evtrade.indicators import EMAState, ema_step
from evtrade.strategies import VectorizedStrategy, register_strategy


@dataclass
class DualMAStrategyState:
    fast: EMAState = field(default_factory=EMAState)
    slow: EMAState = field(default_factory=EMAState)


@register_strategy("dual_ma")
class DualMAStrategy(VectorizedStrategy):
    """双均线交叉策略: 唯一入口 step, CPU/GPU 同代码"""
    params_spec = {
        "fast": {"default": 10, "type": int},
        "slow": {"default": 30, "type": int},
    }

    def init_state(self, params):
        return DualMAStrategyState()

    def step(self, state, bar, params):
        if bar["mark"] == 0:
            return state, 0
        state.fast, fast = ema_step(state.fast, bar["c"], params["fast"])
        state.slow, slow = ema_step(state.slow, bar["c"], params["slow"])
        if state.fast.count < params["fast"]:
            return state, 0
        sig = 1 if fast > slow else (-1 if fast < slow else 0)
        return state, sig
```

> 批量路径（大网格扫描）不写第二个方法：引擎在 `_compute_signals` 里对每个桶循环调
> 同一个 `step`。若需要纯 torch 批量算子做离群分析/加速，直接调 `xp_ema_torch(values, p)`
> （`(B,T)` tensor in/out），与 `step` 的 `ema_step` 逐桶累积在信号轨迹上保持一致。

---

## 7. 性能与边界

### 7.1 变周期 batched EMA

`xp_ema_torch` 实现：`torch.cumsum` + per-row period 切片 + 递推；B 维并行，T 维串行（递推依赖前一根）。

复杂度：O(B·T·P_max)，B=10000, T=100000, P_max=60 时 ~6e10 浮点运算。
GPU 内存足够时摊销；CPU batched 仍较慢。

### 7.2 实盘滑动窗口

`Engine.on_bars` 维护 `(1, T_lookback)` buffer；每桶 CLOSE 重算整段 EMA。

短期：T_lookback ≤ 500 时 O(T) 重算 < 1ms（CPU）。  
长期：state 增量接口 `step_incremental(state, bar)` 跳过窗口重算（未来 work）。

### 7.3 与 cupy 时代对比

| 指标 | cupy (旧) | PyTorch (新) |
|---|---|---|
| CPU 单测 | numpy 1.x | torch 2.x |
| GPU 单测 | cupy 11.x/12.x | torch 2.x + CUDA runtime |
| 变周期 EMA | searchsorted + reduceat 笨拙 | cumsum + per-row slice 直观 |
| 包大小 | cupy ~1GB (CUDA 12) | torch ~2GB (CUDA) / ~200MB (CPU) |
| 依赖管理 | cupy-cuda11x / cupy-cuda12x 双 wheel | torch 单一 wheel (CPU + CUDA 二选一) |

---

## 8. 与现有文档的对应

| 主题 | 文档 |
|---|---|
| 策略唯一入口 `step` / `init_state` | [14-统一策略契约.md](14-策略DSL与三端转译.md) |
| CPU/GPU 数据流 + 架构图 | [02-系统架构.md](02-系统架构.md) |
| 通道偏离策略实现 | [06-交易策略详解.md](06-交易策略详解.md) |
| 指标公式 | [05-指标计算-EMA通道.md](05-指标计算-EMA通道.md) |
| 重构历史与对比 | [12-重构与性能内核.md](12-重构与性能内核.md) |
| CLI `--device` 参数 | [10-配置参数与运行指南.md](10-配置参数与运行指南.md) |

---

## 9. 验证清单

```bash
# 安装 torch CPU (或 CUDA 版本)
pip install torch

# 跑全部测试
pytest tests -q
# 预期: 147 passed, 2 skipped (GPU 测试)

# CLI 跑通
python -m evtrade backtest --device cpu --strategy channel_deviation --synthetic-days 30 --no-sleep
python -m evtrade backtest --device auto --strategy channel_deviation --synthetic-days 30 --no-sleep

# spec 校验
openspec validate --specs
```

测试覆盖：

- `tests/test_torch_backend.py` — get_xp / gpu_available / xp_ema_torch batch 维 / per-row period
- `tests/test_pyproject.py` — torch 依赖 + cupy 已删除
- `tests/test_metrics_v3.py` / `test_metrics_units.py` — 30 字段 metrics + 单位约定
- 既有 `test_strategy_unified.py` / `test_vectorized.py` — vectorized vs Engine.on_bars reconcile
