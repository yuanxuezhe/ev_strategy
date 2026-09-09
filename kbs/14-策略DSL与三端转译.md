# 14 · 统一策略契约 (VectorizedStrategy, 2026-09-09 重写)

> **本文为 2026-09-09 重构后版本。** 旧的"DSL → 三端转译"章节整体废弃。
>
> 一份策略主逻辑 (Python), 两端统一执行: **CPU (numpy) / GPU (cupy)**。
> 不再有 numba `@njit`、CUDA `__device__` 函数渲染器、AST 白名单、state_spec 三端投影。
> 性能由 CuPy 高阶封装承担 (cupy.where / cupy.cumsum / cupy.searchsorted 等
> 映射到 cuBLAS / cuDNN 预编译 kernel, 无需手写 CUDA C99)。
>
> 写新策略前必读 11 号文档第 3 节。

## 1. 设计动机 (旧 DSL 痛点)

重构前 (`evtrade/strategies/dsl.py` 907 行 + `core/kernel.py` 889 行 numba
流式内核 + `core/gpu.py::cuda_sweep_window_generic` NVRTC 编译) 的形态:

- 策略主逻辑写在 `compute_signal` 的 **docstring** (DSL body), 由 AST
  白名单 + 三端渲染器 (Python exec / numba @njit / CUDA C99) 自动转译。
- 同一份 AST 三端共用, 浮点路径 bitwise 一致 (依赖 `--fmad=false`)。
- 痛点: AST 白名单维护成本高; 三端语义需手动对齐; numba 编译开销影响冷启动;
  CUDA 通用 kernel 只支持 ≤8 参数 (字段上限限制); 策略新增一个 `state_spec` 字段
  要同步改 DSL AST + numba jitclass + CUDA device 三处。

本次重构 (2026-09-09, change `2026-09-09-unify-strategy-contract`):
DSL 渲染管线整体下线。**策略唯一抽象方法 = `compute_signals(xp, bars, params)`**;
CPU/GPU 两端都是 Python + xp (numpy/cupy) 数组算子, 框架 0 渲染逻辑。

## 2. 策略契约 (VectorizedStrategy)

```python
# evtrade/strategies/vectorized_base.py
class VectorizedStrategy:
    params_spec: dict[str, dict] = {}   # 参数 schema 校验
    strategy_key: str = ""

    def compute_signals(self, xp, bars: dict, params: dict) -> "xp.ndarray[int8]":
        """策略唯一入口; 返回 1=BUY / -1=SELL / 0=hold 的桶级信号数组"""
        raise NotImplementedError

    def compute_signals_for_one_bar(self, xp, bar: dict, params: dict) -> int:
        """framework 包装: 单 bar -> 单元素 xp 数组 -> compute_signals[0]
        子类可覆写以维护 instance-level FSM (如 channel_deviation 的桶切换累积 EMA)。"""
        single = {k: xp.asarray([v]) for k, v in bar.items()}
        return int(self.compute_signals(xp, single, params)[0])
```

`bars` 契约 (vectorized 引擎传入的桶聚合 dict, 来自 `run_vectorized`):

| 键 | dtype | 说明 |
|---|---|---|
| `ts` | int64[N] | 桶右端点 ts (YYYYMMDDHHmmss) |
| `o` / `h` / `l` / `c` | float64[N] | 桶的 OHLCV |
| `v` | float64[N] | 桶累计成交量 |
| `mark` | int8[N] | 1=策略期, 0=预热段 |
| `n_bars` | int64[N] | 桶内 1m bar 根数 (供调试/统计用) |

约定:
- `xp` 是 `numpy` 或 `cupy` 模块; **策略代码禁止直接 `import numpy` 或 `import cupy`**, 必须通过 `xp` 抽象。
- 返回数组长度 == `len(bars["ts"])`; 元素 ∈ {-1, 0, 1}。
- `mark == 0` 桶策略 SHOULD 输出 0; framework 内部再做一次 `sig *= mark` 兜底。
- 框架层 MUST NOT 在 `bars` 里塞指标键 (无 `up` / `dw` / `low_dev` 等); 指标自维护。

## 3. CPU/GPU 双端统一路径

```
run_vectorized(bars_1m, period, warmup_until,
               strategy=VectorizedStrategy, params, ..., device)
├─ xp = evtrade.backends.get_xp(device)              # "cpu"→numpy / "gpu"→cupy
├─ buckets = _aggregate_buckets_xp(xp, bars_1m, ...) # 向量化桶聚合
├─ sig = strategy.compute_signals(xp, buckets, params)
├─ sig = sig * buckets["mark"]                       # 预热段清0
└─ trades / summary = metrics 顺序执行
```

`Engine.on_bars` (实盘/对账路径) 调 `strategy.compute_signals_for_one_bar(xp, cur, params)`;
桶切换时用上一桶 finalized OHLCV 驱动一次 (见 kbs/09)。

## 4. 指标 (evtrade.indicators)

`evtrade/indicators/{ema,atr,rsi,boll}.py` 全部重写为**纯 xp 版 + 纯 Python 增量版**:

| 形态 | 函数 | 用途 |
|---|---|---|
| 批量 (xp) | `xp_ema(xp, values, p)` | `compute_signals` 主体使用 |
| 批量 (xp) | `xp_ema_channel(xp, h, l, p)` | 通道策略 |
| 增量 (纯 Python, 标量 in/out) | `ema_push / ema_current` | `compute_signals_for_one_bar` 覆写 |
| 增量 (纯 Python, 6 标量) | `ema_channel_push / ema_channel_current` | 通道逐 bar |

- 文件无 `@njit`、无 `CUDA_DEVICE_*` 字符串常量、无 `numba` import。
- 表达式与浮点路径与原 numba 版逐位一致 (同一份 k=2/(p+1) SMA seed + 递推)。
- `pyproject.toml` 已删 `numba` 依赖。

## 5. 策略示例 (channel_deviation)

混合策略 = **xp 数组算子 (EMA 通道 + 偏离) + Python FSM (锁存/桶切换)**:
- `compute_signals`: xp 算 up/dw + 4 个偏离; Python 循环跑 FSM
- `compute_signals_for_one_bar`: 覆写, 维护 instance `_up_st / _dw_st / _fsm` 增量 EMA

`ma_crossover` (纯数组算子, 无 FSM):
```python
class MACrossoverStrategy(VectorizedStrategy):
    def compute_signals(self, xp, bars, params):
        fast = int(params["fast"]); slow = int(params["slow"])
        ema_fast = xp_ema(xp, bars["c"], fast)
        ema_slow = xp_ema(xp, bars["c"], slow)
        diff = ema_fast - ema_slow
        sig = xp.where(diff > 0, xp.int8(1), xp.int8(0))
        sig = xp.where(diff < 0, xp.int8(-1), sig)
        return sig
```

## 6. 与旧 DSL 的对比

| 维度 | 旧 DSL (2026-09-08) | 统一契约 (2026-09-09) |
|---|---|---|
| 策略写法 | docstring DSL body | Python class 覆写 `compute_signals` |
| 持久状态声明 | `state_spec` (DSL AST 白名单) | 实例字段 (Python 属性) |
| 渲染层 | Python exec + numba @njit + CUDA C99 | 无 (Python + xp) |
| 性能路径 | numba (CPU 7000+ bar/s) / CUDA | numpy (CPU) / cupy (GPU) |
| 指标调用 | DSL 白名单 + 三端 stub | 普通 Python 函数 (xp 兼容) |
| 参数上限 | CUDA 通用 kernel 限 8 字段 | 无上限 |
| 三端一致性 | bitwise (依赖 --fmad=false) | 浮点路径 (xp 算子) |
| `params_spec` 校验 | DSL 编译期 | `_resolve_params` 实例化时 |

## 7. 写新策略的步骤

1. 复制 `tests/test_strategy_template.py` 改 class 名 + 注册 key。
2. `compute_signals(xp, bars, params)` 内用 `xp_ema / xp_atr / xp_rsi / ...` 算子
   (见 evtrade.indicators); 无需 import numpy/cupy, 直接用 `xp`。
3. 需要逐 bar 路径 (Engine.on_bars) 维护 instance 状态? 覆写 `compute_signals_for_one_bar`,
   用增量 API (`ema_push / ema_current / ...`)。
4. 测试: `tests/test_strategy_unified.py` 提供 6 项锁定 (子类/注册表/dtype/cpu-vs-gpu/
   vectorized-vs-Engine reconcile/16 字段 metrics); 复制其中子集即可。

## 8. 跨文档索引

- kbs/02 §2 — 统一路径分层 (CPU 向量化 + Engine 逐 bar)
- kbs/05 — 指标纯 xp 版 + 纯 Python 增量版实现
- kbs/06 §5 — 策略主逻辑写法 (channel_deviation / ma_crossover 双范式)
- kbs/09 §2 — `Engine.on_bars` 桶 CLOSE 语义 (桶切换时驱动)
- kbs/11 §3 — 新策略开发模板
- kbs/12 — 重构与性能内核 (旧 numba/CUDA 段已废)
- spec.md — `Strategy interface contract` + `Indicators are private to strategies`