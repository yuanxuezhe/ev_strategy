# 07 PyTorch 与性能

> PyTorch 统一后端(`backends.get_xp`) + CPU/GPU 双端透明路由 + 性能内核 +
> batched_step hook(opt-in GPU 批量 sweep)
>
> 相关源码: `backends.py` (`evtrade/backends.py`)、`batched_sweep.py` (`evtrade/core/batched_sweep.py`)、
> `vectorized_engine.py` (`evtrade/core/vectorized_engine.py`)、`tsbucket.py` (`evtrade/core/tsbucket.py`)

## 1. 背景:为什么 PyTorch

### 1.1 cupy 时代的痛点(已下线)

| 痛点 | 表现 |
|---|---|
| 双份算子维护 | 每个指标(EMA 等)既要 numpy 版又要 cupy 版 |
| 变周期 batched EMA 笨拙 | 旧 `xp_ema_channel` 用 `searchsorted+reduceat`, B 维 + per-row period 表达困难 |
| 实盘/批量代码双轨 | 批量路径与逐 bar 路径循环结构不同, 同一算法要写两遍 |

### 1.2 目标与现状(2026-09-10 起)

用 **PyTorch 作为唯一 array 后端**(CPU + GPU 同一库同一 wheel), 统一:

- **CPU 回测**: `torch.device("cpu")`
- **GPU 回测**: `torch.device("cuda")`
- **批量扫描**: sweep 层按参数组合循环, 每组走同一条 vectorized 路径
- **实盘逐棒**: `Engine.on_bars` 逐桶 CLOSE 调 `step`

要点:

- 后端唯一旋钮是 `evtrade.backends.get_xp(device) -> torch.device`
- 策略代码**只写一份** numpy/Python 标量运算, cpu/cuda 上行为一致
- vectorized 路径与实盘路径调**同一个标量 `step(state, bar, params)`**, 指标由 `ema_step` /
  `ema_channel_step` 增量维护
- 策略直接 `from evtrade.indicators import ema_step` 用 numpy / Python 标量即可, 无需感知 device
- **GPU-batched sweep opt-in 第二路径**(`batched_step`): 详见 §5

## 2. 后端路由(`evtrade/backends.py`)

```python
from evtrade.backends import get_xp, gpu_available, resolve_device, to_tensor, to_host

get_xp("cpu")        # torch.device("cpu")
get_xp("gpu")        # torch.device("cuda"); CUDA 不可用 fallback cpu + warning
get_xp("auto")       # cuda 可用则 cuda, 否则 cpu
gpu_available()      # == torch.cuda.is_available()
resolve_device("auto", gpu_ok=False)   # "cpu"

# to_tensor / to_host: 框架层进出 torch 设备用, 策略代码不直接调
to_tensor(np_array, device="cpu")   # numpy.ndarray -> torch.Tensor
to_host(tensor)                     # torch.Tensor(任意 device) -> torch.Tensor("cpu")
```

CUDA 不可用时 fallback 到 cpu + `RuntimeWarning`。

旧 `core/capability.py::select_device` 已删除, 能力探测收敛到 `backends.resolve_device` +
`gpu_available`; 旧 `core/gpu.py` 删除, 桶 ts / mark 预计算迁至 `core/tsbucket.py`。

### 2.1 策略代码不应直接判断 device

```python
# 错 (耦合 torch.cuda)
if torch.cuda.is_available():
    values = values.cuda()

# 对 (框架层处理)
state, sig = strategy.step(state, bar, params)   # 引擎循环调用, 不感知 device
```

策略里既不需要 `xp` 变量, 也不需要 `import torch`;
`to_tensor` / `to_host` 仅供框架层(数据进出 torch 设备)使用。

## 3. `--device` 口径(2026-09-12 更新)

当前回测热路径为设备无关的 numpy / Python 标量实现,
`--device {cpu, gpu, auto}`(默认 `auto`)选择的是 `backends.get_xp` 的 torch 后端设备,
为未来 tensor 热路径预留。

- `--device gpu` 在 CUDA 不可用时**抛错**(提示改用 `auto`/`cpu`; GPU 机器请 `uv sync --extra gpu`
  或 `bash scripts/sync-torch-cu.sh`)
- `--device auto` 无 CUDA 时自动降级 cpu 并打 warning

### 3.1 GPU wheel 安装约定(`add-gpu-extra-pyproject`)

- `pyproject.toml` 声明 `[project.optional-dependencies].gpu = ["torch==2.9.0+cu128"]`,
  由 `[tool.uv.sources]` + `[[tool.uv.index]]` 指向 `https://download.pytorch.org/whl/cu128`
- GPU 协作者 `uv sync --extra gpu` 一次锁住(lockfile 已 commit)
- CPU 协作者 `uv sync` 默认行为不变(仍拉 pypi CPU wheel; 若 lockfile 强制 cu128 则接受)
- 防御性 helper: `bash scripts/sync-torch-cu.sh` 检测 torch 是 CPU wheel 时自动 reinstall 到
  cu128(接受 `EVT_TORCH_CU_TAG` 环境变量覆写, 默认 `cu128`)
- `uv lock --upgrade-package torch==2.9.0+cu128 --index-strategy unsafe-best-match`
  用于升级 / 切换 cu tag 后重生 lockfile 并 commit

### 3.2 bars 契约(桶级数组)

vectorized 路径在**桶级 finalized OHLCV**(numpy 数组)上跑策略:

```python
buckets = {
    "ts":     np int64   (T,),   # 桶时间戳 (14 位整数)
    "o":      np float64 (T,),
    "h":      np float64 (T,),
    "l":      np float64 (T,),
    "c":      np float64 (T,),
    "v":      np float64 (T,),
    "mark":   np int     (T,),   # 1=策略期, 0=预热段
    "n_bars": np int64   (T,),   # 每桶含 1m 根数 (诊断用, 策略 step 不可见)
}
# 引擎逐桶取标量喂给 step:
#   bar = {"ts","o","h","l","c","v","mark"}  ->  strategy.step(state, bar, params)
```

`batched_step` 看到的 bars 契约: `batched_sweep._build_bars_tensor` 把上述 numpy 桶数组
转成 `dict[str, Tensor]` `[T]`(不含 `n_bars`); `mark` 转 `int8`, 价格转 `float64`。

## 4. 批量 vs 实盘:同一条 step 路径

### 4.1 批量回测 / 网格扫描

```python
from evtrade.core.vectorized_engine import run_vectorized
from evtrade.strategies import get_strategy

strat = get_strategy("channel_deviation", params={"low1": 1.5, "low2": 1.0})
res = run_vectorized(bars_1m=bars, period="5m", warmup_until=warmup_until,
                     strategy=strat, params=strat.params)
# res["buckets"]: 桶级 OHLCV; res["sig"]: (T,) int8 信号数组; res["final_state"]: 策略 state 透传
```

### 4.2 实盘 / 单线回测(逐棒)

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

`step` 内调用 `ema_step` / `ema_channel_step` 维护 EMA 增量; 批量路径与逐 bar 路径调的是
**同一个 `step`**。

### 4.3 与 cupy 时代对比(历史存档)

| 指标 | cupy (旧, 已删) | PyTorch (现行) |
|---|---|---|
| CPU 单测 | numpy 1.x | torch 2.x (CPU) |
| GPU 单测 | cupy 11.x/12.x | torch 2.x + CUDA runtime |
| 后端选择 | numpy / cupy 双算子 | `get_xp` 单旋钮 (torch CPU/CUDA) |
| 策略算子 | 旧 `xp_ema_channel` 批量(已删) | 无批量指标算子; 两路径同一条标量 `step` |
| 包大小 | cupy ~1GB (CUDA 12) | torch ~2GB (CUDA) / ~200MB (CPU) |
| 依赖管理 | cupy-cuda11x / cupy-cuda12x 双 wheel | torch 单一 wheel (CPU + CUDA 二选一) |

> 当前 `pyproject.toml` 只声明 `torch>=2.0`, 无 cupy、无 numba。

## 5. Batched sweep hook(opt-in)

### 5.1 动机

`sweep()` 当前走 `ThreadPoolExecutor` + `--workers` 多进程, 每组参数独立跑一遍 `run_vectorized`
→ `strategy.step` Python 循环。扩展性受核数限制(4~32 倍)。
**参数间并行**才是 GPU 真正的高价值维度 —— N_combos 组共享同一份 bars,
作为 torch batch dim 在 1 次 kernel launch 内出 N 份信号。

第一刀只加速**信号生成**; 成交执行仍逐 combo 调现有逻辑(`batched_step` 仅产 sig),
改动面小、bitwise 兼容。

### 5.2 Hook 契约

```python
@classmethod
def batched_step(cls, state, bars, params, *, n_combos, n_bars):
    """opt-in GPU 批量 hook; 默认未实现, sweep 自动走 ThreadPool"""
    raise NotImplementedError
```

| 参数 | shape | 说明 |
|---|---|---|
| `state` | dataclass 字段 `[N]` Tensor | 每 combo 一份批量 state |
| `bars` | `dict[str, Tensor]` `[T]` | `{"ts","o","h","l","c","v","mark"}` 全 1-D |
| `params` | `dict[str, Tensor]` `[N]` | 每 combo 一份 param |
| `n_combos`, `n_bars` | int | shape 元数据 |
| 返回 | `(new_state, sig [N, T] int8)` | sig 第一维 combo, 第二维 bar |

### 5.3 路由规则(`core/sweep.sweep()`)

```
use_batched = (
    hasattr(cls, "batched_step")
    and cls.batched_step is not VectorizedStrategy.batched_step  # 子类真覆写
    and device != "cpu"
    and len(combos) >= 32                                       # 小网格 GPU 启动开销 > 收益
    and gpu_available()
)
```

命中走 `run_batched`(`core/batched_sweep.py`), 否则现有 ThreadPool 路径。
`run_batched` 内捕获 `torch.cuda.OutOfMemoryError` → 自动 fallback ThreadPool + warning。
`--workers` 在 batched 模式下被忽略, 打印 notice。

### 5.4 当前实现状态

| 策略 | 是否实现 `batched_step` | 原因 |
|---|---|---|
| `ma_crossover` | ✅ 已实现 | 纯 EMA 增量、无 FSM、可向量化 |
| `channel_deviation` | ❌ 不实现 | FSM 锁存跨桶 latch 难 tensor 化; 继续走 ThreadPool |
| `filtered_mr` | ❌ 不实现 | 大周期桶跟踪 + ADX Wilder 平滑 + FSM latch 混合, torch 收益小 |

### 5.5 关键约束

- **浮点必须 float64**, 跟 per-combo `step` 循环产出 bitwise 一致(容差 1e-12)
- `mark=0` 段复刻原 step 语义(如 `ma_crossover` 仍推 EMA 累积)
- 异常立即透传(不延后到 sync point), 保持 `test_sweep_does_not_crash_when_one_combo_fails` 语义
- `run_vectorized` 签名**不动**(spec R10 MUST NOT 带 device)

### 5.6 EMA kernel(`torch_ema`)

新增 `indicators.ema.torch_ema(values: Tensor, p: int) -> Tensor`:

- 输入 `[T]`、输出 `[T]`、`dtype/device` 保留(自动转 float64)
- 前 `p-1` 个返回 `0.0`(跟 `ema_step` 一致, 不是 numpy `ema()` 的 NaN)
- 算法: `cumsum` seed(`cumsum[p-1]/p`) + 递推尾段 `out[i] = values[i]*k + out[i-1]*(1-k)`,
  `k = 2/(p+1)`
- **跟 numpy `ema()` 参考版 float64 bit-equal**(`np.array_equal` 通过)
- seed 必须用 `numpy.sum` 取首段避免 GPU 串行 cumsum 与 numpy pairwise 求和的 1-bit 末位差

`batched_sweep` 按 `tf1` 值**分组**调用 `torch_ema`(同 `tf1` 的 combo 一次 kernel 出整段)——
第一刀不用 `vmap`, 按整数取值分组够用。

### 5.7 收益范围

信号生成从 `N_combos × T_bars` 个 Python `step` 调用 → 1 个 `batched_step` 调用。
成交仍 `N_combos` 次(每窗每 combo 一次), 但信号部分是大头。

### 5.8 不做的事(留给后续 change)

- 撮合数学向量化(per-combo `cash/position/cur_qty` → `[N]` Tensor)
- `channel_deviation` / `filtered_mr` FSM 向量化
- `vmap` over `tf1`(按值分组够用)

## 6. 性能特性(2026-09-09 重构后)

旧版 numba 加速特性(`tests/test_kernel_unit.py::test_live_step_equals_batch` 期望微秒级 step)
已作废 —— 本版无 JIT 内核。

- **速度特性**:
  - torch CPU 路径: batched 向量化吞吐 ≈ 0.05~0.2 M bar/s(取决于策略复杂度,
    channel_deviation 含 FSM Python 循环较慢)
  - torch CUDA 路径: 桶聚合走 torch 算子; 主要优势在大网格并发(单 GPU batch)
  - 实盘路径(`Engine.on_bars`): 逐 bar Python 调用 + dataclass state 维护,
    吞吐 ≈ 1~5 K bar/s, 满足实盘 1m bar 节奏
- **栈**: 消费级 NVIDIA GPU + `torch`(CUDA runtime 版 wheel)
- **设计**: torch 算子(`cumsum` / `where` / `unfold` 等)在 CUDA 后端映射到预编译 kernel;
  桶 ts / mark 用整数化历法预计算(numpy 向量化), 避免 GPU 上 int64 除法
- **何时用 GPU**: 与 CPU 路径吞吐差距不大; 主要价值在大网格并发(单 GPU 一次处理数千组参数)
- **失败回退**: torch CUDA 不可用时 `--device auto` 自动 fallback cpu 并打 warning

旧 numba 基准(168K bar / 9 ms)已不适用; 本版无 JIT 内核, 吞吐以 torch 算子为准。

## 7. CPU/GPU 一致性保证(差分测试)

`tests/test_strategy_unified.py` 锁定:

1. **CPU vs GPU 容差**: 同一策略同参数, `np.array_equal(sig_cpu, sig_gpu)`,
   summary 浮点字段 1e-6 容差(`test_ma_crossover_cpu_vs_gpu`)
2. **vectorized vs Engine 桶级对账**: 两条入口调同一份 `step`, 信号轨迹一致
   (少量桶信号漂移属 EMA 累积顺序差异, 已由策略 step 实现吸收)
3. **MA 交叉语义**: numpy 独立算 EMA + 交叉, 与 vectorized 引擎输出对比
4. **三策略 × CPU/GPU/vectorized/Engine 四路径**: 6 项核心断言
   (signature / dtype / cpu-vs-gpu / vectorized-vs-Engine / state 字段集 / signal bitwise)

## 8. 重构历史(2026-09 大改)

| 日期 | change | 关键变更 |
|---|---|---|
| 2026-09-04 | (git 5585371) | 单文件 `mysql_analyze_demo.py` 685 行 |
| 2026-09-05 | 包拆分 | `evtrade/` 子包化 + BarAggregator + WFO 评分 |
| 2026-09-08 | state-spec 工厂化 | kernel/gpu 按策略 state spec 工厂化; 桶表去指标化 |
| 2026-09-09 | `decouple-indicators-from-framework` | framework 完全不假定指标; `core/incremental_indicators.py` 整文件删除 |
| 2026-09-09 | **`unify-strategy-contract`** | **DSL 渲染层 + numba 流式内核 + NVRTC CUDA 编译整体下线**; 策略唯一入口 `step(state, bar, params) -> (state, sig)` |
| 2026-09-10 | `consolidate-simplify-core` | 删行情源子包(数据加载内联 `core/data.py`); 删 atr/rsi/boll 指标(仅留 EMA); GPU 探测模块收敛为 `core/tsbucket.py`; 删 `--engine` / sleep 关闭 / 分段步长等 CLI 死 flag |
| 2026-09-10 | **`pytorch-unified-strategy`** | 后端从 cupy/numpy 双端统一为 PyTorch 单端; `pyproject.toml` 删 `cupy`/`numba`, 声明 `torch>=2.0`; GPU 环境探测改 `torch.cuda`; 新增 `evtrade/backends.py` 后端选择器 |
| 2026-09-10 | `strategy-step-only` | 唯一抽象方法收敛到 `step(state, bar, params) -> (state, sig)`; 状态由 engine 持有(`@dataclass`) |
| 2026-09-11 | `gpu-batched-sweep` | `VectorizedStrategy.batched_step` opt-in hook + `indicators.torch_ema` 落地; sweep 在大网格 GPU 自动路由 batched 路径 |
| 2026-09-12 | `add-gpu-extra-pyproject` | pyproject 加 `[project.optional-dependencies].gpu = ["torch==2.9.0+cu128"]` + sync-torch-cu.sh helper |
| 2026-09-13 | `drop-engine-finance` | framework 删 execution/metrics/replay/permutation/config; 业务概念全部下放策略 step; framework 只驱动 step |
| 2026-09-13 | `drop-scale-residue` | 收尾 SimulatedExecutor / `_TradeStateTracker` 等 scale state machine 残留 |

## 9. 验证清单

```bash
# 安装 torch CPU (或 CUDA 版本)
pip install torch
# 或 GPU: uv sync --extra gpu

# 跑全部测试
uv run pytest -q          # 166 passed, 1 skipped (CUDA 不可用时)

# CLI 跑通
python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device cpu
python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device auto

# spec 校验
openspec validate --specs
```

测试覆盖:

- `tests/test_torch_backend.py` — get_xp(cpu/gpu/auto + fallback) / gpu_available /
  to_tensor / to_host
- `tests/test_tsbucket_cache.py` — tsbucket 预计算缓存
- `tests/test_batched_sweep.py` — batched_step hook + 路由规则 + OOM fallback
- `tests/test_sync_torch_cu.py` — bash helper 行为契约
- `tests/test_strategy_unified.py` — step 状态跨调用持续 + CPU/GPU 对账
- `tests/test_pyproject.py` — torch 依赖 + numba/cupy 已删除断言
