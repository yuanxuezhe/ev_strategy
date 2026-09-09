# Design: 统一策略契约

## 1. 策略契约

### 1.1 唯一基类

`evtrade/strategies/vectorized_base.py::VectorizedStrategy`：

```python
class VectorizedStrategy:
    params_spec: dict[str, dict[str, Any]] = {}

    def __init__(self, params=None, **kwargs):
        merged = dict(params or {})
        for k, v in kwargs.items():
            merged.setdefault(k, v)
        self.params = self._resolve_params(merged)
        for k, v in self.params.items():
            setattr(self, k, v)

    @classmethod
    def _resolve_params(cls, params):
        # 与旧 StrategyBase 同式: 填默认 + type/min/max 校验
        ...

    def compute_signals(self, xp, bars: dict, params: dict) -> xp.ndarray:
        """子类必实现: 数组算子 -> 信号数组 (int8)
        xp: numpy 或 cupy 模块
        bars: 桶聚合 dict {ts, o, h, l, c, v, mark, n_bars}
        """
        raise NotImplementedError

    def compute_signals_for_one_bar(self, xp, bar: dict, params: dict) -> int:
        """framework 包装: 单 bar -> 单元素数组 -> compute_signals[0]
        channel_deviation 等有 instance FSM 的策略可覆写此方法维护状态。
        """
        single = {k: xp.asarray([v]) for k, v in bar.items()}
        return int(self.compute_signals(xp, single, params)[0])
```

注册表 `_STRATEGIES: dict[str, type[VectorizedStrategy]]` + `register_strategy` / `get_strategy` / `get_strategy_class` / `get_strategy_param_spec` / `available_strategies` 全部搬到 `vectorized_base.py`。

### 1.2 bars 契约

```python
{"ts":   int64[N],    # 桶时间戳 (14 位整数)
 "o":    float64[N],  # 每桶 open
 "h":    float64[N],  # 每桶 high
 "l":    float64[N],  # 每桶 low
 "c":    float64[N],  # 每桶 close
 "v":    float64[N],  # 每桶 volume
 "mark": int8[N],     # 1=策略期 (stime >= warmup), 0=预热期
 "n_bars": int64[N]}  # 每桶含 1m bar 根数 (调试用)
```

## 2. 引擎与 CLI

### 2.1 唯一执行入口

`evtrade/core/vectorized_engine.py::run_vectorized`：

```python
def run_vectorized(bars_1m, period, warmup_until, strategy, params,
                   init_cash, init_position, trade_qty, scale,
                   buy_pct, sell_pct, device="cpu") -> dict:
    xp = get_xp(device)                       # numpy 或 cupy
    buckets = _aggregate_buckets_xp(xp, ...)  # 桶聚合 (向量化)
    sig = strategy.compute_signals(xp, buckets, params)
    # 拉回 host 顺序执行成交 + 累积 equity_curve
    sig_np, close_np, ts_np = _to_host(sig, buckets["c"], buckets["ts"])
    mark_np = _to_host(buckets["mark"])
    sig_np = sig_np * mark_np                 # 预热段信号清零
    exec_state, equity_curve, baseline_curve = _execute_trades(...)
    summary = metrics.summarize(
        equity_curve=equity_curve,
        baseline_curve=baseline_curve,
        years=years, n_trades=exec_state["n_trades"], ...,
    )
    return {"sig": sig_np, "trades": exec_state["trades"],
            "summary": summary, "buckets": _to_host(buckets)}
```

### 2.2 Engine.on_bars 重写

`evtrade/core/engine.py::Engine.on_bars`：

```python
def on_bars(self, bars):
    closed = bars[:-1]
    cur = bars[-1]
    # 删 _sync_strategy_state (DSL ctx 已无)
    if cur.get("mark", 1) == 0:
        return  # 预热: 不驱动策略
    price = float(cur["close"])
    self.executor.update_price(price)
    xp = np  # 引擎始终 CPU
    sig_int = self.strategy.compute_signals_for_one_bar(xp, cur, self.strategy.params)
    if sig_int != 0:
        side = {1: "BUY", -1: "SELL"}[sig_int]
        self.executor.trade(side, price, cur["ts"])
    if self.verbose and sig_int != 0:
        # 策略覆写 format_signal_line 自维护输出
        info = ...  # 策略可覆写 compute_signals_for_one_bar 返回 (sig, info)
        print(self.strategy.format_signal_line(cur, side, info), flush=True)
```

注：`format_signal_line` 当前签名只接 (cur, signal, info) 三参；策略若要展示 per-bar info，可在 `compute_signals_for_one_bar` 子类实现里捕获后存 instance 字段供 hook 读。

### 2.3 CLI 收口

`evtrade/cli.py`：

```python
# backtest
ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
# 删 --engine
# backtest_main: 统一 _run_backtest(args)

# sweep
ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
# sweep_main: from .metrics import bars_to_arrays; 调 run_sweep(..., device=args.device)
```

`--engine` 兼容：在 argparse 顶层解析时检测到 `--engine` 打 `DeprecationWarning` 并自动转换（`kernel→auto / ref→cpu / vectorized→cpu`），然后删除该 argv 项继续正常解析。**仅本次回归期保留**，后续 change 移除。

## 3. metrics 补齐

### 3.1 新 `metrics.summarize`

```python
def summarize(equity_curve, baseline_curve, years,
              n_trades, n_buy, n_sell, turnover,
              init_cash, init_position, final_price) -> dict:
    """vectorized 引擎终态 + equity 序列 -> 16 字段绩效字典"""
    cash = ...  # 由 caller 算或参数传
    position = ...
    final_equity = cash + position * final_price
    baseline_final = init_cash + init_position * final_price
    excess = final_equity - baseline_final
    excess_pct = excess / baseline_final * 100
    ann_excess_pct = excess_pct / years if years > 0 else 0.0

    # cagr: (equity[-1]/equity[0])^(1/years) - 1
    cagr = ((equity_curve[-1] / equity_curve[0]) ** (1.0 / years) - 1.0) * 100 \
        if years > 0 and equity_curve[0] > 0 else 0.0

    # 日收益 (按 1d 窗口; 简化版: 桶收益)
    ret_eq = np.diff(equity_curve) / equity_curve[:-1]
    ret_bl = np.diff(baseline_curve) / baseline_curve[:-1]
    excess_ret = ret_eq - ret_bl

    sharpe_excess = excess_ret.mean() / excess_ret.std() * np.sqrt(252) \
        if excess_ret.std() > 0 else 0.0
    downside = excess_ret[excess_ret < 0]
    sortino_excess = excess_ret.mean() / downside.std() * np.sqrt(252) \
        if len(downside) and downside.std() > 0 else 0.0

    # max_drawdown + max_dd_days
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_drawdown = float(-dd.min()) if len(dd) else 0.0
    trough_idx = int(dd.argmin()) if len(dd) else 0
    peak_idx = int(peak[:trough_idx + 1].argmax()) if trough_idx > 0 else 0
    max_dd_days = (trough_idx - peak_idx) / 252.0
    max_dd_recovered = bool(equity_curve[-1] >= peak[trough_idx])

    calmar = cagr / (max_drawdown * 100) if max_drawdown > 0 else 0.0

    x_dd = (excess_ret.cumsum() - np.maximum.accumulate(excess_ret.cumsum()))
    x_mdd = float(-x_dd.min()) if len(x_dd) else 0.0

    return {
        "final_price": final_price, "n_trades": n_trades, "n_buy": n_buy,
        "n_sell": n_sell, "final_cash": cash, "final_position": position,
        "final_equity": final_equity, "baseline": baseline_final,
        "excess": excess, "excess_pct": excess_pct, "years": years,
        "ann_excess_pct": ann_excess_pct,
        "cagr": cagr, "sharpe_excess": sharpe_excess,
        "sortino_excess": sortino_excess, "calmar": calmar,
        "max_dd_days": max_dd_days, "max_dd_recovered": max_dd_recovered,
        "x_mdd": x_mdd, "max_drawdown": max_drawdown, "turnover": turnover,
    }
```

### 3.2 vectorized_engine._execute_trades 累积 equity

```python
def _execute_trades(sig_np, close_np, ts_np, init_cash, init_position,
                    trade_qty, scale, buy_pct, sell_pct):
    n = len(sig_np)
    cash = init_cash
    position = init_position
    equity_curve = np.empty(n)
    baseline_curve = np.empty(n)
    base_init = init_cash + init_position * close_np[0] if n else init_cash
    last_side = 0
    cur_qty = trade_qty
    trades = []
    n_trades = n_buy = n_sell = 0
    turnover = 0.0
    for i in range(n):
        s = int(sig_np[i])
        price = float(close_np[i])
        if s != 0:
            # 倍投
            if s == last_side:
                cur_qty *= scale
            else:
                cur_qty = trade_qty
                last_side = s
            if s == 1:  # BUY
                if price > 0:
                    if buy_pct > 0:
                        q = min(buy_pct * cash / price, cash / price)
                    else:
                        q = min(cur_qty, cash / price)
                    if q > 0:
                        cash -= q * price; position += q
                        n_trades += 1; n_buy += 1; turnover += q * price
                        trades.append({"ts": int(ts_np[i]), "side": "BUY",
                                       "qty": float(q), "price": float(price),
                                       "cash_after": float(cash)})
            else:  # SELL
                if sell_pct > 0:
                    q = min(sell_pct * position, position)
                else:
                    q = min(cur_qty, position)
                if q > 0:
                    cash += q * price; position -= q
                    n_trades += 1; n_sell += 1; turnover += q * price
                    trades.append({"ts": int(ts_np[i]), "side": "SELL",
                                   "qty": float(q), "price": float(price),
                                   "cash_after": float(cash)})
        # 累计 equity（按桶盯市）
        equity_curve[i] = cash + position * price
        # baseline: 期初买入 init_position 持有不动
        baseline_curve[i] = base_init * (price / close_np[0]) if close_np[0] > 0 and n else base_init
    last_price = float(close_np[-1]) if n else 0.0
    return {"cash": cash, "position": position, "last_price": last_price,
            "n_trades": n_trades, "n_buy": n_buy, "n_sell": n_sell,
            "turnover": turnover, "trades": trades}, equity_curve, baseline_curve
```

## 4. indicators xp 化

### 4.1 增量版改纯 Python

```python
# evtrade/indicators/ema.py
def ema_push(s_sum, s_count, s_ema, value, p):
    """EMA 增量推入: state=(s_sum, s_count, s_ema) -> 新 (sum', count', ema')
    表达式: count < p 累加 sum; count == p 时 ema = sum/p; count > p 时
            ema = value*k + ema*(1-k), k = 2/(p+1)。
    """
    if s_count < p:
        s_sum = s_sum + value
        s_count = s_count + 1
        if s_count == p:
            s_ema = s_sum / p
    else:
        k = 2.0 / (p + 1.0)
        s_ema = value * k + s_ema * (1.0 - k)
        s_count = s_count + 1
    return s_sum, s_count, s_ema

def ema_current(s_sum, s_count, s_ema, pending, p):
    """EMA 当前值 (不改 state); 数据不足返回 0.0"""
    if s_count < p - 1:
        return 0.0
    if s_count == p - 1:
        return (s_sum + pending) / p
    k = 2.0 / (p + 1.0)
    return pending * k + s_ema * (1.0 - k)
```

ema_channel_push / atr_push / rsi_push / boll_push / sma_push 同式（保留 3-6 标量 in/out API）。

### 4.2 批量版改 np.ndarray

```python
def ema(values: np.ndarray, p: int) -> np.ndarray:
    """EMA(values, p): 返回长度 len(values) 的 ndarray (前 p-1 个 NaN)"""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < p:
        return out
    k = 2.0 / (p + 1.0)
    seed = np.sum(values[:p]) / p
    out[p - 1] = seed
    e = float(seed)
    for i in range(p, n):
        e = float(values[i]) * k + e * (1.0 - k)
        out[i] = e
    return out
```

## 5. sweep / replay 收口

### 5.1 sweep.run_one_vectorized

```python
def run_one_vectorized(bars, period, warmup_until, strategy_name,
                       strategy_params, init_cash, init_position,
                       trade_qty, scale=1.0, buy_pct=0.0, sell_pct=0.0,
                       all_in=False, device="cpu") -> dict:
    """单组参数单窗: 走 run_vectorized; 返回 metrics.summarize 16 字段"""
    from .metrics import bars_to_arrays as _b2a
    from .vectorized_engine import run_vectorized
    from ..strategies import get_strategy

    if len(bars["stime"]) == 0:
        return _empty_metrics(init_cash, init_position)

    strat = get_strategy(strategy_name, params=strategy_params or {})
    result = run_vectorized(
        bars, period=period, warmup_until=warmup_until,
        strategy=strat, params=strat.params,
        init_cash=init_cash, init_position=init_position,
        trade_qty=trade_qty, scale=scale,
        buy_pct=buy_pct, sell_pct=sell_pct, device=device,
    )
    # 累计 equity / baseline 用于 metrics
    # 注: run_vectorized 已经在 summary 内算齐 16 字段, 直接 return
    return result["summary"]
```

### 5.2 replay_engine / reconcile

```python
def replay_engine(bars, period, warmup_until, strategy_name, strategy_params,
                  ..., device="cpu"):
    """Engine.on_bars 逐 bar 路径 (与 vectorized 对账用)"""
    from ..strategies import get_strategy
    from evtrade.account import Account
    from evtrade.aggregator import BarAggregator
    from evtrade.engine import Engine
    from evtrade.execution import SimulatedExecutor
    from ._harness import ListBarFeed

    feed = ListBarFeed(bars)
    strat = get_strategy(strategy_name, params=strategy_params or {})
    account = Account(cash=init_cash, position=init_position)
    executor = SimulatedExecutor(account, qty=trade_qty, verbose=False, ...)
    aggregator = BarAggregator(resolve_period_seconds(period), on_bars=None,
                               warmup_until=str(warmup_until) if warmup_until else None)
    engine = Engine(feed, aggregator, strat, executor, verbose=False)
    engine.run()
    # Engine 路径下 Account 已累积 trades; 计算 equity + summary 16 字段
    ...

def reconcile(bars, period, warmup_until, strategy_name, strategy_params, ...,
              device="cpu"):
    """vectorized vs Engine.on_bars 逐 bar 对账"""
    v = replay_vectorized(bars, period, warmup_until, strategy_name,
                          strategy_params, ..., device=device)
    e = replay_engine(bars, period, warmup_until, strategy_name,
                      strategy_params, ...)
    # 信号 + 成交 逐笔对比
    ...
```

## 6. 测试矩阵

| 测试 | 锁定目标 |
|---|---|
| `test_strategy_unified.py::test_unified_signature` | channel_deviation / ma_crossover 都是 VectorizedStrategy 子类 |
| `test_strategy_unified.py::test_cpu_vs_gpu_ma_crossover` | 同一份数据 cpu vs gpu 信号 bitwise 一致 |
| `test_strategy_unified.py::test_cpu_vs_gpu_channel_deviation` | 同上 |
| `test_strategy_unified.py::test_vectorized_vs_ref_channel_deviation` | vectorized 引擎 vs Engine.on_bars 信号 + 成交逐笔一致 |
| `test_strategy_unified.py::test_unified_metrics_fields` | summary 含 16 字段 |
| `test_metrics_v2.py::test_cagr_sharpe_sortino_calmar_manual` | metrics 字段与手算一致 |
| `test_replay_v2.py::test_reconcile_pass` | reconcile 跑通且 pass=True |
| `test_sweep_v2.py::test_sweep_runs_vectorized` | sweep 全 vectorized, metrics 字段非 0 |
| `test_strategy_params_v2.py` | 字典/kwargs/默认/范围校验 |

## 7. 性能特性

| 路径 | 时间复杂度 | 空间复杂度 |
|---|---|---|
| vectorized 引擎 (CPU) | O(N_bars) | O(N_buckets) |
| vectorized 引擎 (GPU) | O(N_bars) 在 device | O(N_buckets) 在 device |
| Engine.on_bars (ref) | O(N_bars) (单 bar compute_signals_for_one_bar) | O(1) per bar |

channel_deviation 在 ref 路径下覆写 `compute_signals_for_one_bar` 维护 instance FSM + 桶切换累积 EMA，单 bar O(1) 增量维护；与 vectorized 路径的 O(N_buckets) 全序列重算在 EMA 数值上 bitwise 一致（同公式同初值）。
