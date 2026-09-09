# Design: dedupe-channel-deviation-fsm-helpers

## 1. 重复点定位

`channel_deviation.py` L104-139（`compute_signals`）与 L143-180（`compute_signals_for_one_bar`）逐字对比，重复 4 块：

```
                    compute_signals          compute_signals_for_one_bar
参数解析             L105-107                 L146-180 末尾 (float(params[k]) for k in ...)
4 偏离公式           L116-119                 L172-175
_fsm_step 调用       L134-137 (low_dev_h, ...)  L177-180 (_fsm_step(self._fsm, ...))
```

**唯一不可消除的差异**是状态/算子来源：

| | compute_signals | _for_one_bar |
|---|---|---|
| EMA 来源 | `xp_ema_channel(xp, ...)` 全序列 GPU | `ema_push / ema_current` 增量 Python |
| NaN 处理 | `xp.where(xp.isnan, 0.0, up)` | 不需要（增量版数据未就绪返回 0.0）|
| FSM 状态 | 函数内 `state = _init_fsm_state()` 局部 | instance `self._fsm` |
| 容器 | `np.zeros(n, dtype=np.int8)` + Python for | 直接返回 int |
| mark=0 跳过 | `if mark[i] == 0: continue` | 引擎层已过滤（不调用本方法）|

这些差异**必须保留**（性能 + stateful 语义），所以本次只抽"无差异"的部分。

## 2. 三个 helper

### 2.1 `_parse_thresholds(params)`

```python
def _parse_thresholds(params: dict) -> tuple[float, float, float, float]:
    """(low1, low2, high1, high2) 一次性解析; 类型转换与 _resolve_params 无关"""
    return (float(params["low1"]), float(params["low2"]),
            float(params["high1"]), float(params["high2"]))
```

**为什么不在基类**：`_resolve_params`（`vectorized_base.py`）已经填默认 + type/min/max 校验，但**返回的是 dict**（key 仍是 str）；取 4 个值仍要写 4 个 `float(...)`。helper 是 DRY，不是新校验层。

### 2.2 `_compute_devs(xp, up, dw, h, l)`

```python
def _compute_devs(xp, up, dw, h, l):
    """4 个偏离 (low_dev, high_dev, low_dev_h, high_dev_l)

    xp 数组算子版 + 标量算术版签名同形:
      - xp=numpy + up/dw/h/l 全为标量 -> 返回 4 个 float
      - xp=cupy + up/dw/h/l 全为 ndarray -> 返回 4 个 ndarray

    实现技巧: 调用 xp.asarray 把标量/数组统一; 算式用 xp.where 处理 dw=0 分母除零
    (批量版与逐 bar版都受益; 现有 compute_signals L116 用 / dw_v 直接除已不安全)
    """
    up_v = xp.where(xp.asarray(xp.isnan(up)) if hasattr(xp, 'asarray') else
                    xp.isnan(up), 0.0, up) if False else (
        xp.where(xp.isnan(up), 0.0, up) if hasattr(up, 'size') and up.size > 1
        else (0.0 if (isinstance(up, float) and up != up) else up)
    )
    dw_v = xp.where(xp.isnan(dw), 0.0, dw) if hasattr(dw, 'size') and dw.size > 1 \
        else (0.0 if (isinstance(dw, float) and dw != dw) else dw)
    # ... 这种 if hasattr 写法不可读
```

**实际方案：分两个实现，用 dispatcher**

```python
def _compute_devs_scalar(up, dw, h, l):
    """标量版 (compute_signals_for_one_bar 用)"""
    if up == 0.0 or dw == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return ((dw - l) / dw * 100.0,
            (h - up) / up * 100.0,
            (dw - h) / dw * 100.0,
            (l - up) / up * 100.0)


def _compute_devs_xp(xp, up, dw, h, l):
    """xp 数组算子版 (compute_signals 用, GPU 加速)"""
    up_v = xp.where(xp.isnan(up), 0.0, up)
    dw_v = xp.where(xp.isnan(dw), 0.0, dw)
    # 避免分母 0: 用 where 替换 0 -> 1 (结果由 isfinite 校验后归 0)
    safe_up = xp.where(up_v == 0, 1.0, up_v)
    safe_dw = xp.where(dw_v == 0, 1.0, dw_v)
    low_dev   = (safe_dw - l)   / safe_dw * 100.0
    high_dev  = (h - safe_up)   / safe_up * 100.0
    low_dev_h = (safe_dw - h)   / safe_dw * 100.0
    high_dev_l= (l - safe_up)   / safe_up * 100.0
    # 通道未就绪处 (up_v==0 或 dw_v==0) 强制 0; 避免 FSM 因 NaN 触发
    ready = (up_v != 0) & (dw_v != 0)
    return (xp.where(ready, low_dev,    0.0),
            xp.where(ready, high_dev,   0.0),
            xp.where(ready, low_dev_h,  0.0),
            xp.where(ready, high_dev_l, 0.0))
```

**调用方按场景选**：

```python
# compute_signals:
low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs_xp(xp, up_v, dw_v, bars["h"], bars["l"])

# compute_signals_for_one_bar:
low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs_scalar(up, dw, cur_high, cur_low)
```

**为什么不用统一签名 + hasattr 探测**：运行时 if hasattr 探测把分支放在最热路径（每根 bar 都跑），可读性差 + Python 开销。两套实现是显式分派，性能 + 清晰度都更好。

**安全改进**：原 `compute_signals` L116 `low_dev = (dw_v - bars["l"]) / dw_v * 100.0` 在 `dw_v==0` 时产生 `inf`，再被 `xp.where(xp.isnan(up), 0.0, up)` 替换 `up` 后的 `up_v==0` 同样问题；下游 FSM 把 `inf` 视为大数会误触发。新版用 `safe_dw/safe_up` + `ready` mask 同时处理 NaN 与 0 —— **比原版更安全**。

### 2.3 `_run_fsm(state, ts, low_dev_h, low_dev, high_dev_l, high_dev, low1, low2, high1, high2)`

```python
def _run_fsm(state, ts, low_dev_h, low_dev, high_dev_l, high_dev,
             low1, low2, high1, high2) -> int:
    """薄包装 _fsm_step; 把传参顺序定死避免两处调用错位"""
    return _fsm_step(state, ts,
                     low_dev_h, low_dev, high_dev_l, high_dev,
                     low1, low2, high1, high2)
```

看似无用（只是 1 行透传），但**消除了"两处传参顺序可能错位"的隐患**。现状两处都是 `(state, ts, low_dev_h, low_dev, high_dev_l, high_dev, low1, low2, high1, high2)` 顺序，重构后新增策略时通过 `_run_fsm` 强制统一。

## 3. 两个方法体精简

### 3.1 `compute_signals`（38 → 约 22 行）

```python
def compute_signals(self, xp, bars: dict, params: dict):
    tf1 = int(params["tf1"])
    low1, low2, high1, high2 = _parse_thresholds(params)

    # 1) 向量化: EMA 通道 + 4 偏离 (xp 数组算子; GPU 加速)
    up, dw = xp_ema_channel(xp, bars["h"], bars["l"], tf1)
    up_v = xp.where(xp.isnan(up), 0.0, up)
    dw_v = xp.where(xp.isnan(dw), 0.0, dw)
    low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs_xp(
        xp, up_v, dw_v, bars["h"], bars["l"])

    # 2) FSM (Python 循环; cupy 时先拉回 host)
    ts = bars["ts"]; mark = bars["mark"]
    if hasattr(ts, "get"):
        ts = ts.get(); mark = mark.get()
        low_dev_h = low_dev_h.get(); low_dev = low_dev.get()
        high_dev_l = high_dev_l.get(); high_dev = high_dev.get()

    n = len(ts)
    sig = np.zeros(n, dtype=np.int8)
    state = _init_fsm_state()
    for i in range(n):
        if mark[i] == 0:
            continue
        sig[i] = _run_fsm(state, int(ts[i]),
                          float(low_dev_h[i]), float(low_dev[i]),
                          float(high_dev_l[i]), float(high_dev[i]),
                          low1, low2, high1, high2)
    return sig
```

### 3.2 `compute_signals_for_one_bar`（40 → 约 22 行）

```python
def compute_signals_for_one_bar(self, xp, bar: dict, params: dict) -> int:
    """单 bar: 增量 EMA + _run_fsm (instance FSM 跨桶持续)"""
    tf1 = int(params["tf1"])
    cur_ts = int(bar["ts"])
    cur_high = float(bar["h"])
    cur_low = float(bar["l"])

    # 桶切换 push (旧桶 high/low 闭锁入 EMA)
    if self._has_prev and self._prev_ts != cur_ts:
        us, uc, ue = self._up_st
        self._up_st = ema_push(us, uc, ue, self._cur_high, tf1)
        ds, dc, de = self._dw_st
        self._dw_st = ema_push(ds, dc, de, self._cur_low, tf1)

    self._prev_ts = cur_ts
    self._cur_high = cur_high
    self._cur_low = cur_low
    self._has_prev = True

    # 当前通道值 (含 pending)
    us, uc, ue = self._up_st
    ds, dc, de = self._dw_st
    up = ema_current(us, uc, ue, cur_high, tf1)
    dw = ema_current(ds, dc, de, cur_low, tf1)

    low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs_scalar(
        up, dw, cur_high, cur_low)

    return _run_fsm(self._fsm, cur_ts,
                    low_dev_h, low_dev, high_dev_l, high_dev,
                    *(_parse_thresholds(params)))
```

**注意 `*(_parse_thresholds(params))`** 解包：保持调 `_run_fsm` 时实参顺序与原来一致（`low1, low2, high1, high2`）。

## 4. 文件总行数变化

| 节 | 行数 | 改后 |
|---|---|---|
| 文件头 docstring + import | 25 | 25 |
| `_fsm_step` + `_init_fsm_state` | 16 | 16 |
| 3 个新 helper | 0 | ~30 |
| `ChannelDeviationStrategy.__init__` | 12 | 12 |
| `compute_signals` | 38 | ~22 |
| `compute_signals_for_one_bar` | 40 | ~22 |
| `format_signal_line` | 10 | 10 |
| **合计** | **192** | **~150** |

**净减 ~42 行**，重复 0 行。

## 5. 不变量验证

| 不变量 | 验证方式 |
|---|---|
| GPU 加速保留 | `_compute_devs_xp` 内部全用 xp 算子；`xp_ema_channel` 不动 |
| FSM bitwise 一致 | `_fsm_step` 调用顺序未变（仅透传到 `_run_fsm`）|
| state 语义不变 | compute_signals 函数内局部 `_init_fsm_state()` 保留；_for_one_bar `self._fsm` 保留 |
| mark=0 跳过 | 批量内 `if mark[i] == 0: continue` 保留 |
| 信号 dtype/length | 返回 `np.int8`、长度 `len(ts)` 不变 |
| 已有测试通过 | `test_strategy_unified.py::test_vectorized_vs_engine_on_bars_reconcile` + `test_cpu_vs_gpu_*` 应 PASS |

## 6. 风险

| 风险 | 缓解 |
|---|---|
| `_compute_devs_xp` 用 `safe_dw/safe_up` 替换 0 分母，行为是否与原版一致？ | 原版 `(dw_v - l) / dw_v` 在 dw_v==0 时返 inf，FSM 会触发；新版返 0 更安全。**可能改变信号轨迹**。需要 e2e backtest 跑一次 channel_deviation，对比前后 `summary["trades"]` 笔数与 cagr 变化 |
| `_compute_devs_scalar` 加 `up==0 or dw==0` 短路，原版没这分支 | 原版会 RuntimeWarning: divide by zero；新版静默返 0。FSM 看到 0 偏离 = 不触发，**等价** |
| 三个 helper 是 `_` 前缀，未来如被多处复用，是否抽到 `_common.py`？ | 不在本次范围。过度抽象 = 早优化；当前只有 1 个策略用，等第 2 个再抽 |
| `_parse_thresholds` 每次调用都新建 tuple，开销? | 微秒级，可忽略；且 `*tuple` 解包比 dict 取值稍快 |

## 7. KB 改动要点

### `kbs/06-交易策略详解.md` §5

**现状**（L80-118，约 50 行）：列 compute_signals + _for_one_bar 完整代码模板

**改后**：

- 删原 §5.1 + §5.2 完整代码
- 新 §5：3 个 helper 的位置（文件顶部）+ 职责列表
- 新 §5.1：compute_signals 关键 5 行骨架（注释：完整看 git blame）
- 新 §5.2：_for_one_bar 关键 5 行骨架
- §5.3 不变（基类默认实现）

### `kbs/11-扩展指南.md` 行 47-78

**现状**：教"覆写两个方法"的范式

**改后**：

- 删两个完整方法模板
- 新增 "**复用 helper**：当多个偏离/参数共享时，提到模块级 `_` 函数"
- 引用 `kbs/06` 作为详细模板

## 8. 端到端验证

| 命令 | 期望 |
|---|---|
| `uv run pytest tests/test_strategy_unified.py -v` | 4+ 个 test 全 PASS |
| `uv run pytest -q` | 122+ 全 PASS |
| `uv run python -m evtrade backtest --strategy channel_deviation --device cpu --period 5m --start 20250101 --end 20260903 --no-sleep` | cagr/calmar/mdd 与重构前一致（±浮点误差）|
| `uv run python -m evtrade backtest --strategy channel_deviation --device gpu --period 5m --start 20250101 --end 20260903 --no-sleep` | 同上 |
| `uv run python -m evtrade replay --log <log.csv> --strategy channel_deviation --device cpu --against-ref` | PASS |

**关键对比指标**：重构前后 `n_trades / final_equity / cagr / calmar / max_drawdown` 应 bitwise 一致（除 NaN→0 改动可能影响极端场景）。

## 9. 回滚预案

每个 helper 是独立小函数，回滚只需把 helper 调用 inline 回原代码即可。不涉及 schema 改动。`format_signal_line` / `__init__` / `_fsm_step` 不动。