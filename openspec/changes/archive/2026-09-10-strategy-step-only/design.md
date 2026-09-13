# Design: strategy-step-only — 策略唯一方法收敛到 step

## 1. 架构目标

**算法与执行模型彻底分离**：

```
┌────────────────────────────────────────────────────────────┐
│   Engine 层 (cpu/gpu + 批量/逐 bar 都由它决定)              │
│                                                            │
│   VectorizedEngine.run_vectorized                          │
│     if device == "gpu":                                    │
│       - batched fast-path: 用 xp_ema_channel 一次算全序列 │
│       - state = GPU 数组, 在 fast-path 里维护              │
│     else:                                                  │
│       - 循环调 strategy.step(state, bar, params)            │
│       - state 在 host (numpy)                               │
│                                                            │
│   Engine.on_bars                                           │
│     - 循环调 strategy.step(state, bar, params)              │
│     - state 跨调用持续 (instance 持有)                     │
│                                                            │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│   Strategy 层 (纯算法, 不感知执行模型)                      │
│                                                            │
│   class ChannelDeviationStrategy(VectorizedStrategy):     │
│     params_spec = {"tf1": ..., "low1": ..., ...}           │
│                                                            │
│     def init_state(self, params) -> ChannelState:          │
│         return ChannelState(...)                           │
│                                                            │
│     def step(self, state, bar, params) -> (state, sig):    │
│         if bar["mark"] == 0:  # 预热段                     │
│             return state, 0                                 │
│         up, dw = ema_channel_step(state, h, l, p)          │
│         devs = compute_devs(up, dw, h, l)                  │
│         sig = fsm(state.fsm, bar["ts"], devs, params)      │
│         return state, sig                                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 2. 基类 API 变化

### 2.1 `VectorizedStrategy`（`evtrade/strategies/vectorized_base.py`）

**删除**：
```python
def compute_signals(self, xp, bars, params): raise NotImplementedError  # 删
def compute_signals_for_one_bar(self, xp, bar, params) -> int: ...      # 删
```

**新增抽象**：
```python
def step(self, state, bar: dict, params: dict) -> tuple[Any, int]:
    """策略唯一入口: 拿到 state + 单桶 bar -> 返回 (new_state, signal)
    
    state: 上一步状态 (init_state 返回值; 无状态策略传 None)
    bar:   {"ts", "o", "h", "l", "c", "v", "mark"}
    返回: (new_state, signal), signal ∈ {-1, 0, 1}
    
    策略 MUST:
      - 若 bar["mark"] == 0: return state, 0  (预热段)
      - 仅做"算法": 算指标 + FSM, 不持有 instance state
      - 不 import numba / cupy
    """
    raise NotImplementedError

def init_state(self, params: dict) -> Any:
    """返回 state 初值; 无状态策略默认返 None
    
    框架在 strategy 实例化时调一次, 把 state 存到 engine 层
    (不是 self._state, 是 engine 持有, 策略无感)
    """
    return None
```

### 2.2 为什么 `init_state` 默认返 `None` 而不是 `{}`

- 无状态策略（`ma_crossover`）不需要 state
- 框架统一处理：`if state is None` → 用一个 sentinel `EMPTY_STATE` 透传
- stateful 策略覆写 `init_state` 返回 `@dataclass` 实例

## 3. state 类型：dataclass

```python
from dataclasses import dataclass, replace

@dataclass
class ChannelDeviationState:
    ema_up: tuple[float, int, float]      # (sum, count, ema)  for up rail
    ema_dw: tuple[float, int, float]      # (sum, count, ema)  for dw rail
    fsm: dict[str, Any]                   # {"low_hit": ..., "lock_ts": ..., ...}
    prev_ts: int = 0
    cur_high: float = 0.0
    cur_low: float = 0.0
    has_prev: bool = False
```

**为什么 dataclass 而不是 dict**：
- IDE 自动补全 + 类型检查
- 默认值在字段定义里, `init_state` 不用手填
- `dataclasses.replace(state, x=y)` 提供 immutable update 语义

## 4. 指标 step API 形态

### 4.1 现状

```python
# 当前 ema.py
def ema_push(s_sum, s_count, s_ema, value, p) -> tuple:
    """state 拆 3 个标量传入; 返回新 (sum', count', ema')"""

def ema_current(s_sum, s_count, s_ema, pending, p) -> float:
    """返回当前 ema 值"""

def ema_channel_push(us, uc, ue, ds, dc, de, h, l, p) -> tuple:
    """6 标量 in/out"""

def ema_channel_current(us, uc, ue, ds, dc, de, h, l, p) -> tuple:
    """返回 (up, dw)"""
```

### 4.2 重构后

```python
from dataclasses import dataclass

@dataclass
class EMAState:
    sum: float = 0.0
    count: int = 0
    ema: float = 0.0

def ema_step(state: EMAState, value: float, p: int) -> tuple[EMAState, float]:
    """state + 1 标量 -> (new_state, ema_value)
    
    规则 (与 ema_push + ema_current 同式):
      - count < p: sum += value; count += 1; ema 仍 0
      - count == p: ema = sum / p (SMA seed)
      - count > p: ema = value*k + ema*(1-k)
    
    返回: (new_state, ema_value)
    """
    s = state.sum + value if state.count < p else state.sum
    c = state.count + 1 if state.count < p else state.count
    if c < p:
        return EMAState(s, c, 0.0), 0.0
    if c == p:
        new_ema = s / p
        return EMAState(s, c, new_ema), new_ema
    k = 2.0 / (p + 1.0)
    new_ema = value * k + state.ema * (1.0 - k)
    return EMAState(state.sum, c, new_ema), new_ema
```

**`ema_channel_step`**：内部调两次 `ema_step`，返回 `(state, up, dw)`。

类似 API 形态应用 `atr_step` / `rsi_step` / `sma_step` / `boll_step`。

### 4.3 保留 `xp_xxx` 批量版

`xp_ema_channel(xp, h, l, p)` 等保留不变 —— **engine fast-path 用**，策略**不直接调**。

## 5. 引擎层改动

### 5.1 `VectorizedStrategy.run_vectorized`（`evtrade/core/vectorized_engine.py`）

**现状**：调 `strategy.compute_signals(xp, buckets, params)` 拿全序列 sig。

**重构后**：分两种路径：

```python
def run_vectorized(bars_1m, period, warmup_until, strategy, params,
                   init_cash, init_position, trade_qty, scale,
                   buy_pct, sell_pct, device="cpu"):
    xp = get_xp(device)
    buckets = _aggregate_buckets_xp(xp, bars_1m, period, warmup_until)
    sig_arr, exec_state, equity, baseline, first_ts, last_ts = (
        _compute_signals_xp(xp, strategy, params, buckets))

    summary = _summarize(...)
    return {...}


def _compute_signals_xp(xp, strategy, params, buckets):
    """vectorized 路径: 循环调 strategy.step"""
    state = strategy.init_state(params)
    n = len(buckets["ts"])
    sig_arr = xp.zeros(n, dtype=xp.int8)

    for i in range(n):
        bar = {
            "ts": int(buckets["ts"][i]),
            "o": float(buckets["o"][i]),
            "h": float(buckets["h"][i]),
            "l": float(buckets["l"][i]),
            "c": float(buckets["c"][i]),
            "v": float(buckets["v"][i]),
            "mark": int(buckets["mark"][i]),
        }
        state, sig = strategy.step(state, bar, params)
        sig_arr[i] = sig

    # 拉回 host
    sig_np = _to_host(sig_arr)
    ...
    return sig_np, exec_state, ...
```

**GPU 加速保留说明**：当策略 step 内部调 `ema_channel_step(state, h, l, p)` 时，state 可以是 dataclass（含 `EMAState(sum, count, ema)`）或 GPU 数组。

**简化方案**：stateful 策略在 GPU 路径下，state dataclass 也持有 host 副本，engine fast-path 用 `xp_ema_channel` 一次性算全序列 EMA 然后再循环 step。这样：
- GPU 加速在 fast-path 内部（算 EMA）
- step 循环调一次拿 sig
- state dataclass 始终在 host

### 5.2 `Engine.on_bars`（`evtrade/core/engine.py`）

**现状**：调 `strategy.compute_signals_for_one_bar(np, bar, params)`。

**重构后**：

```python
class Engine:
    def __init__(self, feed, aggregator, strategy, executor, verbose=False):
        ...
        self.strategy = strategy
        self._state = strategy.init_state(strategy.params)  # 跨调用持有
        ...

    def on_bars(self, bars):
        cur = bars[-1]
        if self._last_cur and self._last_cur["ts"] != cur["ts"]:
            prev = self._last_cur
            if prev.get("mark", 1) == 1:
                bar = {...}  # 从 prev 构 bar dict
                self._state, sig_int = self.strategy.step(
                    self._state, bar, self.strategy.params)
                self.bucket_signals.append(int(sig_int))
                if sig_int != 0:
                    self.executor.trade({1:"BUY",-1:"SELL"}[sig_int], price, prev["ts"])
        self._last_cur = dict(cur)

    def _flush_final_bucket(self):
        ...
        self._state, sig_int = self.strategy.step(self._state, bar, ...)
        ...
```

**关键**：`self._state` 由 Engine 持有，跨调用持续；策略只接收/返回。

## 6. 策略代码改写

### 6.1 `channel_deviation.py`：从 235 行降到 ~25 行

```python
from dataclasses import dataclass, field, replace
from ..indicators import ema_channel_step, EMAChannelState
from .vectorized_base import VectorizedStrategy, register_strategy


@dataclass
class ChannelDeviationState:
    ema: EMAChannelState = field(default_factory=EMAChannelState)
    fsm: dict = field(default_factory=lambda: {
        "low_hit": False, "high_hit": False, "lock_ts": 0,
        "low_acted": False, "high_acted": False})
    prev_ts: int = 0
    cur_high: float = 0.0
    cur_low: float = 0.0
    has_prev: bool = False


def _fsm(state_fsm: dict, cur_ts: int, low_dev_h, low_dev,
         high_dev_l, high_dev, low1, low2, high1, high2) -> int:
    """FSM 步进; 与原版语义逐行一致"""
    if cur_ts != state_fsm["lock_ts"]:
        state_fsm["lock_ts"] = cur_ts
        state_fsm["low_acted"] = False
        state_fsm["high_acted"] = False
    sig = 0
    if state_fsm["low_hit"] and low_dev_h < low2 and not state_fsm["low_acted"]:
        sig = 1; state_fsm["low_hit"] = False; state_fsm["low_acted"] = True
    elif state_fsm["high_hit"] and high_dev_l < high2 and not state_fsm["high_acted"]:
        sig = -1; state_fsm["high_hit"] = False; state_fsm["high_acted"] = True
    if low_dev > low1 and not state_fsm["low_acted"]:
        state_fsm["low_hit"] = True; state_fsm["low_acted"] = True
    if high_dev > high1 and not state_fsm["high_acted"]:
        state_fsm["high_hit"] = True; state_fsm["high_acted"] = True
    return sig


def _devs(up, dw, h, l):
    """4 个偏离; 通道未就绪返 0"""
    if up == 0.0 or dw == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return ((dw - l) / dw * 100.0,
            (h - up) / up * 100.0,
            (dw - h) / dw * 100.0,
            (l - up) / up * 100.0)


@register_strategy("channel_deviation")
class ChannelDeviationStrategy(VectorizedStrategy):
    params_spec = {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "low2":  {"default": 1.0, "type": float, "min": 0.0, "max": 100.0},
        "high1": {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "high2": {"default": 0.5, "type": float, "min": 0.0, "max": 100.0},
        "tf1":   {"default": 21,  "type": int,   "min": 2,   "max": 1000},
    }

    def init_state(self, params):
        return ChannelDeviationState()

    def step(self, state, bar, params):
        if bar["mark"] == 0:
            return state, 0  # 预热段

        tf1 = int(params["tf1"])
        low1, low2, high1, high2 = (float(params[k]) for k in
                                    ("low1", "low2", "high1", "high2"))
        cur_ts = int(bar["ts"])
        cur_high = float(bar["h"])
        cur_low = float(bar["l"])

        # 桶切换 push 旧桶 high/low 进 EMA
        if state.has_prev and state.prev_ts != cur_ts:
            state.ema, _ = ema_channel_step(state.ema,
                                            state.cur_high, state.cur_low, tf1)
        state.prev_ts = cur_ts
        state.cur_high = cur_high
        state.cur_low = cur_low
        state.has_prev = True

        # 当前通道值 (含 pending)
        state.ema, up, dw = ema_channel_step(state.ema, cur_high, cur_low, tf1)
        low_dev, high_dev, low_dev_h, high_dev_l = _devs(up, dw, cur_high, cur_low)
        sig = _fsm(state.fsm, cur_ts, low_dev_h, low_dev,
                   high_dev_l, high_dev, low1, low2, high1, high2)
        return state, sig

    def format_signal_line(self, ts, sig, info=None):
        from ..primitives import fmt
        info = info or {}
        prefix = f"{'BUY' if sig==1 else 'SELL' if sig==-1 else '':>6} >>> " if sig else "             "
        return (f"{prefix}[{ts}] | UP={fmt(info.get('up'))} DW={fmt(info.get('dw'))} | "
                f"low_dev(L/DW)={fmt(info.get('low_dev'))}% "
                f"high_dev(H/UP)={fmt(info.get('high_dev'))}%")
```

**对比原版**（235 行 → ~80 行包含 helper + dataclass + step + format_signal_line）。**核心 step 只有 ~15 行**。

### 6.2 `ma_crossover.py`：本就简单

```python
@register_strategy("ma_crossover")
class MACrossoverStrategy(VectorizedStrategy):
    params_spec = {
        "fast": {"default": 5,  "type": int, "min": 2, "max": 1000},
        "slow": {"default": 20, "type": int, "min": 2, "max": 1000},
    }

    def init_state(self, params):
        return None  # 无状态

    def step(self, state, bar, params):
        if bar["mark"] == 0:
            return state, 0
        # 与原版 compute_signals 同式 (单 bar)
        ...
```

注：`ma_crossover` step 内需要 EMA —— 但它**没有 instance state**，所以无法跨调用保持 EMA。

**这是个真问题**！解决：
- 选项 A：`ma_crossover` 用 `EMAState` 持跨调用 EMA（stateful 化）
- 选项 B：ma_crossover 走 batched 路径（`run_vectorized` 不循环调 step，而是调 `compute_signals_array` 拿全序列）—— 但这违反"step 是唯一入口"
- 选项 C：保留 ma_crossover 在 batched 路径下用 `xp_ema` 一次算全序列，由 engine 维护 state

**采纳 C**：engine 在调用策略 step 之前，先用 `xp_xxx` 批量算出所有桶的指标，**预计算并塞进 state**（state 此时含 `bars["fast_ema"] / bars["slow_ema"]`）。这是 engine 的职责。

**实现**：engine 检测策略是否有 `step` 而无 `_pure_batch_only` 标记，默认 batched 算指标，state 携带预计算结果。

更简单：**保留 ma_crossover 的 stateless 写法**，但要求它的 step 接收一个完整 `bars` 数组（不限单 bar）。

**反对**：这又破坏"bar = 单桶"的契约。

**最终方案**：ma_crossover 改 stateful —— 用 `EMAState` 持 EMA state，每桶 step 推一次。**这是算法上的小调整，但能保留 step 契约**。

## 7. 指标 step 实现细节

### 7.1 EMAState dataclass

```python
@dataclass
class EMAState:
    """EMA 增量 state; ema_step 进出 dataclass 实例"""
    sum: float = 0.0
    count: int = 0
    ema: float = 0.0


@dataclass
class EMAChannelState:
    """EMA 通道增量 state; ema_channel_step 进出 dataclass 实例"""
    up: EMAState = field(default_factory=EMAState)
    dw: EMAState = field(default_factory=EMAState)
```

### 7.2 ema_step 实现

```python
def ema_step(state: EMAState, value: float, p: int) -> tuple[EMAState, float]:
    """与旧 ema_push + ema_current 等价的单函数"""
    if state.count < p:
        new_sum = state.sum + value
        new_count = state.count + 1
        if new_count < p:
            return EMAState(new_sum, new_count, 0.0), 0.0
        elif new_count == p:
            new_ema = new_sum / p  # SMA seed
            return EMAState(new_sum, new_count, new_ema), new_ema
        else:
            # 理论上 count+1 > p 不可能 (因为 if count < p 阻断了)
            ...
    else:
        k = 2.0 / (p + 1.0)
        new_ema = value * k + state.ema * (1.0 - k)
        return EMAState(state.sum, state.count + 1, new_ema), new_ema
```

### 7.3 ema_channel_step 实现

```python
def ema_channel_step(state: EMAChannelState, h: float, l: float,
                     p: int) -> tuple[EMAChannelState, float, float]:
    new_up, up = ema_step(state.up, h, p)
    new_dw, dw = ema_step(state.dw, l, p)
    return EMAChannelState(up=new_up, dw=new_dw), up, dw
```

### 7.4 atr/rsi/sma/boll step 同形态

```python
# atr.py
@dataclass
class ATRState:
    prev_close: float = 0.0
    ema_tr: float = 0.0
    count: int = 0

def atr_step(state: ATRState, h: float, l: float, c: float,
             p: int) -> tuple[ATRState, float]:
    """与旧 atr_push + atr_current 等价"""
    if state.count == 0:
        tr = h - l  # 简化 (首根 tr)
    else:
        pc = state.prev_close
        tr = max(h - l, abs(h - pc), abs(l - pc))
    if state.count < p:
        return ATRState(prev_close=c, ema_tr=0.0, count=state.count+1), 0.0
    if state.count == p:
        # 第一根凑齐: ATR = SMA of last p trs (这里简化为 seed = tr)
        ...
    k = 1.0 / p
    new_ema = state.ema_tr + k * (tr - state.ema_tr)
    return ATRState(prev_close=c, ema_tr=new_ema, count=state.count+1), new_ema
```

类似 `rsi_step` / `sma_step` / `boll_step`。

## 8. 测试矩阵

| 测试 | 改动 |
|---|---|
| `test_compute_signals_returns_int8_xp_array` | 删（方法消失）；改 `test_step_returns_int` |
| `test_compute_signals_for_one_bar_default_wrapper` | 删（wrapper 消失） |
| `test_ma_crossover_cpu_vs_gpu_bitwise_equal` | 改 `test_step_cpu_vs_gpu`；跑两遍 engine, 比较 sig 序列 |
| `test_vectorized_vs_engine_on_bars_reconcile` | 改: 仍断言 vectorized vs engine 信号+成交一致 |
| `test_metrics_summary_has_16_fields` | **不变** |
| `test_both_strategies_are_vectorized_subclass` | **不变** |
| `test_strategy_registry_lists_both` | **不变** |
| **新增** `test_init_state_returns_dataclass` | ChannelDeviationStrategy.init_state 返回 ChannelDeviationState |
| **新增** `test_no_instance_state_in_strategies` | 静态扫描 `evtrade/strategies/*.py`，MUST NOT 出现 `self._fsm` 等 |
| **新增** `test_step_state_persists_across_calls` | 连续调 step(state, bar1)、step(state, bar2)，验证 EMA 累积 |

## 9. 端到端验证

| 命令 | 期望 |
|---|---|
| `uv run pytest -q` | 122+ 全 PASS |
| `uv run python -m evtrade backtest --strategy channel_deviation --device cpu --period 5m --start 20250101 --end 20260903 --no-sleep` | cagr/calmar/mdd/n_trades 与重构前 bitwise 一致 |
| 同上 device=gpu | 同上 |
| `uv run python -m evtrade backtest --strategy ma_crossover --device cpu --period 5m ...` | 同上 |
| `uv run python -m evtrade sweep --strategy channel_deviation --device cpu --max-mdd 0.15 --grid low1=0.5,1.0 --splits 20250601` | filter_pass 行为不变 |

## 10. 风险

| 风险 | 缓解 |
|---|---|
| channel_deviation FSM 行为漂移（state容器从 dict 变 dataclass）| dataclass 字段对位 + test_step_state_persists_across_calls 锁定 |
| ma_crossover 必须 stateful 化（每桶 EMA 增量）| 改用 EMAState；与 channel_deviation 共享状态语义 |
| API breaking（删 compute_signals / _for_one_bar）| 本项目目前无外部用户（CLAUDE.md 自描述）；一次性破坏 |
| dataclass 创建开销（state 实例化频繁）| dataclass 比 dict 稍快；engine 复用同一 state 引用避免拷贝 |
| GPU fast-path 与 step 循环冲突 | 简化方案：stateful 策略 state 在 host（dataclass）；engine batched 路径仍用 xp_ema_channel 一次性算指标，state 携带预计算 EMA 数组 |
| `xp_xxx` 仍被 engine 使用但策略不应直接调 | spec Scenario "策略 step 用 compute_ema_channel 拿 up/dw" 锁定 |
| KB 6 份需改 | tasks 里分节；先 spec/KB 落地再改代码 |

## 11. 落地顺序

按依赖逆序：

1. **指标层**（`indicators/{ema,atr,rsi,boll}.py`）：新 dataclass + step 函数，**保留旧 push/current 作 deprecated shim**
2. **基类**（`vectorized_base.py`）：加 `step` 抽象 + `init_state` 默认 `None`
3. **策略**（`channel_deviation.py` + `ma_crossover.py`）：改写为 step 方法
4. **引擎**（`vectorized_engine.py` + `engine.py`）：循环调 step
5. **测试**（`test_strategy_unified.py`）：重写
6. **删除旧 API**：compute_signals / _for_one_bar / ema_push / ema_current 等彻底从基类 + 策略 + 引擎中删
7. **KB** 同步
9. **archive**