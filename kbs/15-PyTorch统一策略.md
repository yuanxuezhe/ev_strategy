# 15 PyTorch 统一策略 (pytorch-unified-strategy, 2026-09-10)

> 回答的问题：**CPU/GPU/批量回测/实盘逐棒** 如何用**同一套策略代码**统一表达。
> 本文档覆盖 PyTorch 单一后端 (`evtrade/backends.py`)、bars 契约 (桶级 numpy 数组)、
> 批量 vs 实盘的策略接口 (同一条 `step` 路径)。

---

## 1. 背景

### 1.1 重构前的问题（cupy 时代, 已下线）

| 痛点 | 表现 |
|---|---|
| 双份算子维护 | 每个指标（ema 等）既要 numpy 版又要 cupy 版 |
| 变周期 batched EMA 笨拙 | 旧 `xp_ema_channel` 用 `searchsorted+reduceat`，B 维 + per-row period 表达困难 |
| 实盘/批量代码双轨 | 批量路径与逐 bar 路径循环结构不同，同一算法要写两遍 |

### 1.2 目标与现状

用 **PyTorch 作为唯一 array 后端**（CPU + GPU 同一库同一 wheel），统一：

- **CPU 回测**：`torch.device("cpu")`
- **GPU 回测**：`torch.device("cuda")`
- **批量扫描**：sweep 层按参数组合循环，每组走同一条 vectorized 路径
- **实盘逐棒**：`Engine.on_bars` 逐桶 CLOSE 调 `step`

**现状要点（2026-09-10）**：

- 后端唯一旋钮是 `evtrade.backends.get_xp(device) -> torch.device`
  （cpu/cuda 路由 + CUDA 不可用时 fallback）。
- 策略代码**只写一份** numpy/Python 标量运算，cpu/cuda 上行为一致；
  **没有** `xp_ema_torch` 之类的 torch 批量指标算子 —— vectorized 批量路径
  (`core/vectorized_engine` 内部逐桶信号循环) 与实盘路径
  (`Engine.on_bars`) 调用**同一个标量 `step(state, bar, params)`**，
  指标由 `ema_step` / `ema_channel_step` 增量维护。
- 不存在 `xp` 别名要传给策略：策略直接 `from evtrade.indicators import ema_step`，
  用 numpy / Python 标量即可。

---

## 2. 指标算子（evtrade/indicators/ema.py）

当前 `evtrade/indicators/` 只有 `ema.py`，提供两类形态：

| 形态 | 函数签名 | 适用场景 |
|---|---|---|
| **step 增量版** | `ema_step(state, value, p)` / `ema_channel_step(state, h, l, p)` | 策略 `step` 用；state 为 `@dataclass`（`EMAState` / `EMAChannelState`），标量 in/out |
| **numpy 批量版** | `ema(values, p)` / `ema_channel(highs, lows, p)` | 返回 ndarray；jupyter / 复盘 / 测试对拍用 |

```python
# step 增量版 (策略内用; state 由 engine 持有, 跨调用持续)
from evtrade.indicators import EMAState, ema_step

state = EMAState()
state, ema = ema_step(state, 1.234, p=21)

# numpy 批量版 (复盘/测试对拍)
import numpy as np
from evtrade.indicators import ema, ema_channel

highs = np.random.rand(100) + 1.0
lows = np.random.rand(100) + 0.5
up, dw = ema_channel(highs, lows, 21)
```

旧 `xp_ema` / `xp_ema_channel` / `xp_ema_torch` 批量 xp 算子已随 cupy 一并删除；
旧 `ema_push` / `ema_current` 也已删除，统一为 `ema_step` 增量入口。

---

## 3. bars 契约（桶级数组）

vectorized 路径在**桶级 finalized OHLCV**（numpy 数组）上跑策略：

```python
# _aggregate_buckets 产出 (numpy 数组, 每行 = 一个闭合桶)
buckets = {
    "ts":     np int64   (T,),   # 桶时间戳 (14 位整数)
    "o":      np float64 (T,),   # 每桶 open (首根 1m bar)
    "h":      np float64 (T,),
    "l":      np float64 (T,),
    "c":      np float64 (T,),
    "v":      np float64 (T,),
    "mark":   np int     (T,),   # 1=策略期, 0=预热段
    "n_bars": np int64   (T,),   # 每桶含 1m 根数
}
# 引擎逐桶取标量喂给 step:
#   bar = {"ts","o","h","l","c","v","mark"}  ->  strategy.step(state, bar, params)
# 返回 sig ∈ {-1, 0, 1}, 引擎聚合成 (T,) int8 数组
```

- **批量扫描的 "B 维"** 不在单次 vectorized 调用内部：sweep 层
  (`core/sweep.py`, ThreadPoolExecutor) 对每个参数组合各跑一次
  `run_vectorized`，策略代码无需感知 B 维、不需要 `if B == 1`。
- **实盘/对账路径** `Engine.on_bars` 在桶 CLOSE 时（桶切换时）用上一桶
  finalized OHLCV 调同一次 `step`，语义与 vectorized 路径逐桶一致。

---

## 4. 批量 vs 实盘：同一条 step 路径

### 4.1 批量回测 / 网格扫描

```python
from evtrade.core.vectorized_engine import run_vectorized
from evtrade.strategies import get_strategy

strat = get_strategy("channel_deviation", params={"low1": 1.5, "low2": 1.0})
res = run_vectorized(bars_1m=bars, period="5m", warmup_until=warmup_until,
                     strategy=strat, params=strat.params)
# res["summary"]: metrics 全套字段; res["trades"]: 逐笔

# 网格扫描: python -m evtrade sweep --strategy channel_deviation --device auto ...
# sweep 内部: ThreadPoolExecutor 并发, 每组参数各跑一次 run_vectorized
```

### 4.2 实盘 / 单线回测（逐棒）

```python
from evtrade.strategies import get_strategy

strat = get_strategy("channel_deviation", low1=1.5, low2=1.0)
state = strat.init_state(strat.params)  # engine 持有, 跨调用持续

# 每根 5m 桶 CLOSE 时调一次 (Engine.on_bars 内部语义)
def on_bucket_close(bar: dict):
    global state
    state, sig = strat.step(state, bar, strat.params)
    if sig == 1:
        broker.buy(...)
    elif sig == -1:
        broker.sell(...)
```

`step` 内调用 `ema_step` / `ema_channel_step` 维护 EMA 增量；批量路径
（vectorized 内部逐桶信号循环）与逐 bar 路径调的是**同一个 `step`**，信号轨迹一致
（`replay --against-ref` reconcile 锁定，`bucket_diff_cap=8` 容忍 EMA 累积漂移）。

---

## 5. CPU/GPU 透明路由（evtrade/backends.py）

### 5.1 唯一后端旋钮

```python
from evtrade.backends import get_xp, gpu_available, resolve_device

get_xp("cpu")        # torch.device("cpu")
get_xp("gpu")        # torch.device("cuda"); CUDA 不可用 fallback cpu + warning
get_xp("auto")       # cuda 可用则 cuda, 否则 cpu
gpu_available()      # == torch.cuda.is_available()
resolve_device("auto", gpu_ok=False)   # "cpu"
```

CUDA 不可用时 fallback 到 cpu + `RuntimeWarning`。
（旧 `core/capability.py::select_device` 已删除，能力探测收敛到
`backends.resolve_device` + `gpu_available`；旧 `core/gpu.py` 删除，
桶 ts/mark 预计算迁至 `core/tsbucket.py`。）

### 5.2 策略代码不应直接判断 device

```python
# 错 (耦合 torch.cuda)
if torch.cuda.is_available():
    values = values.cuda()

# 对 (框架层处理)
state, sig = strategy.step(state, bar, params)   # 引擎循环调用, 不感知 device
```

策略里既不需要 `xp` 变量，也不需要 `import torch`；`to_tensor` / `to_host`
仅供框架层 (数据进出 torch 设备) 使用。

---

## 6. 策略契约示例（step 唯一入口）

```python
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

> 批量路径（回测 / 大网格扫描）不写第二个方法：vectorized 引擎在内部
> 逐桶信号循环里对每个桶调同一个 `step`
> （state 为 Python dataclass 对象, 逐桶增量递推 EMA）。

---

## 7. 性能与边界

### 7.1 信号循环

vectorized 内部逐桶信号循环 = Python 逐桶循环 + 标量 EMA 递推（`ema_step`），
**没有** numba JIT 也没有 torch 批量指标算子；单组回测吞吐比旧内核慢，
大网格扫描建议 `--workers 16` + `--device auto` 保证吞吐。
GPU 的价值在数据量大的 tensor 侧操作与未来扩展，当前策略信号路径 CPU/GPU
数值行为一致。

### 7.2 实盘滑动窗口

`Engine.on_bars` 在桶 CLOSE 时用 finalized 桶 OHLCV 调一次 `step`；
state 增量接口（`ema_step`）已跳过窗口重算，每桶 O(1) 递推。

### 7.3 与 cupy 时代对比 (2026-09-10 历史存档)

| 指标 | cupy (旧, 已删) | PyTorch (现行) |
|---|---|---|
| CPU 单测 | numpy 1.x | torch 2.x (CPU) |
| GPU 单测 | cupy 11.x/12.x | torch 2.x + CUDA runtime |
| 后端选择 | numpy / cupy 双算子 | `get_xp` 单旋钮 (torch CPU/CUDA) |
| 策略算子 | 旧 `xp_ema_channel` 批量 (已删) | 无批量指标算子; 两路径同一条标量 `step` |
| 包大小 | cupy ~1GB (CUDA 12) | torch ~2GB (CUDA) / ~200MB (CPU) |
| 依赖管理 | cupy-cuda11x / cupy-cuda12x 双 wheel | torch 单一 wheel (CPU + CUDA 二选一) |

> 注：本表为历史对比；当前 `pyproject.toml` 只声明 `torch>=2.0`，
> 无 cupy、无 numba。

---

## 8. 与现有文档的对应

| 主题 | 文档 |
|---|---|
| 旧 DSL / 三端转译 (已下线存档) | [14-策略DSL与三端转译.md](14-策略DSL与三端转译.md) |
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

# CLI 跑通
python -m evtrade backtest --device cpu --strategy channel_deviation --synthetic-days 30
python -m evtrade backtest --device auto --strategy channel_deviation --synthetic-days 30

# 对账 (vectorized vs Engine.on_bars)
python -m evtrade replay --log <log.csv> --strategy channel_deviation --device cpu --against-ref

# spec 校验
openspec validate --specs
```

测试覆盖：

- `tests/test_torch_backend.py` — get_xp (cpu/gpu/auto + fallback) / gpu_available /
  to_tensor / to_host / cupy import 已删断言
- `tests/test_pyproject.py` — torch 依赖 + numba/cupy 已删除断言
- `tests/test_metrics_v3.py` / `test_metrics_units.py` — metrics 字段集 + 单位约定
- `tests/test_device_resolution.py` — device 解析 (resolve_device / gpu_available)
- `tests/test_tsbucket_cache.py` — tsbucket 预计算缓存
- 既有 `test_strategy_unified.py` / `test_vectorized.py` — step 状态跨调用持续 +
  vectorized vs Engine.on_bars reconcile
