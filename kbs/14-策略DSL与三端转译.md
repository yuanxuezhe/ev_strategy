# 14 · 统一策略契约 (VectorizedStrategy, strategy-step-only 2026-09-10)

> **本文为 2026-09-10 重构版。** 唯一抽象方法 = `step(state, bar, params) -> (state, sig)`;
> 状态由 engine 持有 (`@dataclass`), 策略**无 instance attr** (无 `self._fsm`)。
>
> 一份策略主逻辑 (Python), 两端统一执行: **CPU (torch) / GPU (torch CUDA)**。
> 不再有 numba `@njit`、CUDA `__device__` 函数渲染器、AST 白名单、state_spec 三端投影。
> 性能由 PyTorch 高阶算子承担 (torch.where / cumsum / unfold / gather 等
> 映射到 cuBLAS 预编译 kernel, 无需手写 CUDA C99)。
>
> 写新策略前必读 11 号文档第 3 节。

## 1. 设计动机 (旧 DSL 痛点 → step-only 重构)

重构前 (`evtrade/strategies/dsl.py` 907 行 + `core/kernel.py` 889 行 numba
流式内核 + `core/gpu.py::cuda_sweep_window_generic` NVRTC 编译) 的形态:

- 策略主逻辑写在 `compute_signal` 的 **docstring** (DSL body), 由 AST
白名单 + 三端渲染器 (Python exec / numba @njit / CUDA C99) 自动转译。
- 同一份 AST 三端共用, 浮点路径 bitwise 一致 (依赖 `--fmad=false`)。
- 痛点: AST 白名单维护成本高; 三端语义需手动对齐; numba 编译开销影响冷启动;
CUDA 通用 kernel 只支持 ≤8 参数 (字段上限限制); 策略新增一个 `state_spec` 字段
要同步改 DSL AST + numba jitclass + CUDA device 三处。

**第一次重构** (2026-09-09, change `unify-strategy-contract`):
DSL 渲染管线整体下线, 改用 `compute_signals(xp, bars, params)` 一个抽象入口;
但**策略仍持有 instance state** (`self._up_st` / `self._dw_st` / `self._fsm`),
导致**算法和执行模型混在一起**: 批量路径用函数内局部 FSM, 逐 bar 路径用 instance attr,
同一份算法必须写两遍。

**第二次重构** (2026-09-10, change `strategy-step-only`):
唯一抽象方法收敛到 `step(state, bar, params) -> (state, sig)`;
state 由 engine 持有传入传出, **策略无 instance attr**; 批量/逐 bar 两条路径都
循环调同一份 `step`, 信号 bitwise 一致 (reconcile 锁定)。

## 2. 策略契约 (VectorizedStrategy, strategy-step-only)

```python
# evtrade/strategies/vectorized_base.py
class VectorizedStrategy:
    params_spec: dict[str, dict] = {}      # 参数 schema 校验
    strategy_key: str = ""

    def init_state(self, params: dict) -> state | None:
        """返回 state 初值; 默认 None (无状态策略)"""
        return None

    def step(self, state, bar: dict, params: dict) -> tuple[state, int]:
        """策略唯一入口; bar = {ts, o, h, l, c, v, mark} 单桶
        返回 (new_state, sig), sig ∈ {-1, 0, 1} (SELL/无/BUY)
        策略 MUST NOT 持有 instance attr; state 由 engine 持有。"""
        raise NotImplementedError
```

`bar` 契约 (engine 传入的**单桶** dict, 由 vectorized 引擎聚合桶后逐桶传入):

| 键 | dtype | 说明 |
|---|---|---|
| `ts` | int | 桶右端点 ts (YYYYMMDDHHmmss) |
| `o` / `h` / `l` / `c` | float | 桶的 OHLCV |
| `v` | float | 桶累计成交量 |
| `mark` | int | 1=策略期, 0=预热段 |

约定:
- **state 由 engine 持有传入传出**, 策略无 instance attr (无 `self._fsm` 等)。
- `mark == 0` 桶 (预热段) step MUST 直接返 `(state, 0)`, framework 不做兜底。
- 框架层 MUST NOT 在 `bar` 里塞指标键 (无 `up` / `dw` / `low_dev` 等); 指标自维护。
- 策略代码 MUST NOT `import numba` / `import cupy`; CPU/GPU 由 `backends.get_xp(device)` 统一路由到 torch.device。

## 3. CPU/GPU 双端统一路径 (engine 循环调 step)

```
run_vectorized(bars_1m, period, warmup_until,
               strategy=VectorizedStrategy, params, ..., device)
├─ xp = evtrade.backends.get_xp(device)              # "cpu"→torch cpu / "gpu"→torch cuda
├─ buckets = _aggregate_buckets_xp(xp, bars_1m, ...) # 向量化桶聚合
├─ state = strategy.init_state(params)
├─ for i in range(n_buckets):                       # engine 循环调 step
│   state, sig[i] = strategy.step(state, bar_i, params)
├─ sig *= buckets["mark"]                           # 预热段清0
└─ trades / summary = metrics 顺序执行
```

```
Engine.on_bars (实盘/对账):
├─ state = strategy.init_state(strategy.params)
└─ for bar in feed (桶切换时):
    state, sig = strategy.step(state, bar, params)
```

**两条路径调同一份 step**, state 由 engine 持有 (vectorized 用 list, Engine 用
`self._state`), 跨调用持续; 信号 bitwise 一致 (reconcile 测试锁定)。

## 4. 指标 (evtrade.indicators, strategy-step-only)

`evtrade/indicators/{ema,atr,rsi,boll}.py` 提供**三类形态**:

| 形态 | 函数 | 用途 |
|---|---|---|
| 批量 (xp) | `xp_ema(xp, values, p)` / `xp_ema_channel(xp, h, l, p)` | engine fast-path 算全序列 (策略不直接调) |
| step 增量 | `ema_step(EMAState, value, p) -> (EMAState, ema)` | 策略 `step` body 内调 |
| step 增量 | `ema_channel_step(EMAChannelState, h, l, p) -> (state, up, dw)` | 通道策略 step |
| 纯函数版 | `ema(values, p)` / `ema_channel(h, l, p)` | numpy 返 ndarray, jupyter / 复盘用 |

- step API state 是 `@dataclass EMAState(sum, count, ema)` / `EMAChannelState(up, dw)`,
  in/out 单 dataclass, 无需 6 标量 / 3 标量拆开传
- 文件无 `@njit`、无 `CUDA_DEVICE_*` 字符串常量、无 `numba` import
- `pyproject.toml` 已删 `numba` 依赖
- 旧 `ema_push / ema_current` 等作 deprecated shim (后续删除)

## 5. 策略示例 (channel_deviation)

stateful 策略 = **EMA 通道 (step) + 偏离 (stateless 算) + 锁存 FSM (step)**:

```python
@dataclass
class ChannelDeviationState:
    ema: EMAChannelState              # EMA 通道 state
    fsm: dict                         # FSM 锁存
    prev_ts: int = 0
    cur_high: float = 0.0
    cur_low: float = 0.0
    has_prev: bool = False

@register_strategy("channel_deviation")
class ChannelDeviationStrategy(VectorizedStrategy):
    params_spec = {"low1": ..., "low2": ..., "high1": ..., "high2": ..., "tf1": ...}

    def init_state(self, params):
        return ChannelDeviationState()        # 返回 state 初值

    def step(self, state, bar, params):
        if bar["mark"] == 0:
            return state, 0                  # 预热段

        # 1) EMA 通道 (stateful step)
        if state.has_prev and state.prev_ts != bar["ts"]:
            state.ema, _, _ = ema_channel_step(state.ema, state.cur_high, state.cur_low, tf1)
        state.prev_ts = bar["ts"]; state.cur_high = bar["h"]; state.cur_low = bar["l"]
        state.has_prev = True
        state.ema, up, dw = ema_channel_step(state.ema, bar["h"], bar["l"], tf1)

        # 2) 偏离 (stateless 算)
        low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs(up, dw, bar["h"], bar["l"])

        # 3) FSM (stateful step)
        sig = _fsm_step(state.fsm, bar["ts"], low_dev_h, low_dev, high_dev_l, high_dev, ...)
        return state, sig
```

`ma_crossover` (stateful step): 持 EMA fast/slow state, 每次 step 推一根 + 检测交叉。

## 6. 与旧 DSL / 旧统一契约的对比

| 维度 | 旧 DSL (2026-09-08) | 旧统一契约 (2026-09-09) | strategy-step-only (2026-09-10) |
|---|---|---|---|
| 策略写法 | docstring DSL body | Python class 覆写 `compute_signals` | Python class 覆写 `step` |
| 持久状态 | `state_spec` AST 白名单 | instance 属性 (`self._up_st`) | **state 参数 + engine 持有** (dataclass) |
| 渲染层 | Python exec + numba @njit + CUDA C99 | 无 (Python + xp) | 无 (Python + xp) |
| 性能路径 | numba (CPU) / CUDA | torch (CPU) / torch (CUDA) | torch (CPU) / torch (CUDA) |
| 指标调用 | DSL 白名单 + 三端 stub | 普通 Python 函数 | 普通 Python 函数 |
| 参数上限 | CUDA 通用 kernel 限 8 字段 | 无上限 | 无上限 |
| 三端一致性 | bitwise (--fmad=false) | 浮点路径 | 浮点路径 |
| 算法/执行分层 | 混 | 混 (instance attr 两处) | **分** (engine 决定执行模型) |
| `params_spec` 校验 | DSL 编译期 | `_resolve_params` 实例化时 | `_resolve_params` 实例化时 |

## 7. 写新策略的步骤

1. 复制 `tests/test_strategy_template.py` 改 class 名 + 注册 key。
2. 声明 `params_spec` (类型/默认值/范围); 继承 `VectorizedStrategy`。
3. stateful 策略: 声明 `@dataclass MyState`, 覆写 `init_state` 返 `MyState()`。
4. 覆写 `step(self, state, bar, params) -> (state, sig)`:
   - `mark == 0` 直接返 `(state, 0)`
   - 调 `ema_step / ema_channel_step / atr_step / ...` 算指标
   - 算信号, 返 `(state, sig)`
5. 测试: `tests/test_strategy_unified.py::test_step_state_persists_across_calls` +
   `test_cpu_vs_gpu_*` 提供锁定; 复制其中子集即可。

## 8. 跨文档索引

- kbs/02 §2 — 统一路径分层 (CPU 向量化 + Engine 逐 bar)
- kbs/05 — 指标 step API (`EMAState` / `ema_step` / `ema_channel_step`)
- kbs/06 §5 — 策略 `step` 骨架 (channel_deviation / ma_crossover 范式)
- kbs/09 §2 — `Engine.on_bars` 桶 CLOSE 语义 (桶切换时调 step)
- kbs/11 §3 — 新策略开发模板
- kbs/12 — 重构与性能内核 (旧 numba/CUDA 段已废)
- spec.md — `Strategy interface contract` + `Single contract = VectorizedStrategy.step`