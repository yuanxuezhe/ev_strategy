# 09 引擎 Engine 与主流程

> 相关源码：`Engine`（`evtrade/core/engine.py`）、向量化引擎
> （`evtrade/core/vectorized_engine.py`）、`metrics.summarize`（`evtrade/core/metrics.py`）、
> `main`（`evtrade/cli.py`）。
> 本文档为 2026-09-10 核心精简重构版（consolidate-simplify-core）：数据加载入口
> `core/data.py`、`--device {cpu,gpu,auto}` 统一后端、盈亏汇总归 `metrics.summarize`。

## 1. Engine 的装配（构造函数）

```python
engine = Engine(feed, aggregator, strategy, executor, verbose=True)
```

构造时做三件事：

1. 持有四个组件引用（bar 流对象 / aggregator / strategy / executor）
2. **初始化策略 state**：`self._state = strategy.init_state(strategy.params)` —— state 由引擎持有、跨调用持续
3. **接线**：`self.aggregator.on_bars = self.on_bars` —— 聚合器桶回调改道到引擎（`main` 里构造 aggregator 时传的 `on_bars=None` 就是为了在这里被覆盖）

> framework **不再持有** `tf1` / EMA 增量 / FSM 等任何策略相关字段。
> `tf1` 由策略 `params_spec` 自声明；EMA / FSM 持久状态由策略 `step(state, bar, params)`
> 内部的 `@dataclass` state 维护（引擎持有跨调用），framework 不感知字段。

Engine 是唯一的装配点：CLI/replay 都是"构造 aggregator → 构造 Engine"，aggregator 的
`on_bars` 由 Engine 覆盖，不存在独立的全局 build 函数。

## 2. `on_bars` —— 桶 CLOSE 时驱动策略

```
on_bars(bars):                                   # aggregator 每根 1m bar 回调
  cur = bars[-1]
  # 桶切换: 上一桶 finalize 时, 用上一桶的 finalized OHLCV 驱动策略
  if self._last_cur is not None and self._last_cur["ts"] != cur["ts"]:
      self._process_bucket(self._last_cur)       # 上一桶快照
  self._last_cur = cur                           # 记录 cur 供下一根 bar 检测切换

_process_bucket(rec):
  if rec.get("mark", 1) != 1: return            # 预热桶不驱动
  price = float(rec["close"])
  executor.update_price(price)
  self._state, sig = strategy.step(self._state, {ts,o,h,l,c,v,mark}, strategy.params)
  bucket_signals.append(sig)
  if sig != 0:
      executor.trade(sig_to_side(sig), price, rec["ts"])
      if verbose: print(strategy.format_signal_line(rec["ts"], sig, info))
```

要点：

- **桶 CLOSE 语义**：信号触发时点 = 桶**切换**那一刻（看到当前根 `ts` ≠ 上一根 `ts`），
  价格 = 上一桶 finalized `close`。
  与 vectorized 路径在"桶级 finalized OHLCV"上算指标完全对齐（replay/reconcile 测试
  通过 `bucket_diff_cap=8` 容忍少量桶信号漂移）。
- **mark=0（预热）不驱动**：`_process_bucket` 直接 return，不调策略、不成交。
- **信息流**：`step(state, bar, params)` 内部维护 state（增量 EMA、FSM，`@dataclass`），
  state 由引擎持有（`Engine._state`）跨调用持续。
  vectorized 路径（`_compute_signals`）同样循环调 step, state 在循环内持续。
  两种入口走同一份 `step` 算法, 信号 bitwise 一致。
- `info` dict 字段集由策略自由控制（经 `strategy._last_info` 读出），framework 不命名也不假设。

## 3. `run()` 与收尾

```python
def run(self):
    total = 0
    try:
        for bar in self.feed.stream():
            self.aggregator.update(bar)
            total += 1
        self.aggregator.flush()
        # flush 不产生"下一桶切换", 手动补驱动最后一个桶
        if self._last_cur is not None:
            self._process_bucket(self._last_cur)
    except KeyboardInterrupt:
        print("\n已停止")
    if self.verbose:
        print(f"\n完成, 共处理 {total} 根 1m bar")
    return total
```

- `KeyboardInterrupt` 被捕获打印"已停止"；`flush()` 与最后一桶补驱动在流结束后执行（若流已被中断则跳过）。
- 最后一桶补驱动用最后一个桶的 finalized OHLCV 驱动策略一次（防止漏掉最后一桶的信号）。
- `run()` 返回处理过的 1m bar 根数，无其它产出。

## 4. 盈亏汇总（`metrics.summarize`）

盈亏汇总**不是 Engine 的方法**，统一由 `evtrade/core/metrics.summarize(final_state,
init_cash, init_position, equity_curve=None, baseline_curve=None, first_ts, last_ts,
trades=None, bucket_seconds=300) -> dict` 给出，vectorized 路径
（`vectorized_engine._summarize`）在逐 bar 执行中累积 `equity_curve` / `baseline_curve`
后调用。

基础口径（详见 07 文档）：

```
期末价   = final_state.last_price
策略总资产 = cash + position × 期末价
不操作基线 = init_cash + init_position × 期末价
盈亏差额   = 策略总资产 − 基线
盈亏比例   = 盈亏差额 / 基线 × 100
```

该口径消除了标的本身涨跌的影响，衡量的是**择时的贡献**；未含手续费/滑点。

输出约 30 个字段，分四组（完整清单与单位约定见 kbs/13 §1）：

- **终态/基础**：`final_price` / `final_cash` / `final_position` / `final_equity` /
  `baseline` / `n_trades` / `n_buy` / `n_sell` / `turnover` / `excess_pct`
- **时间序列（基于 equity/baseline 曲线）**：`years` / `cagr` / `cagr_excess`（年化超额
  CAGR，复合）/ `sharpe_excess` / `sortino_excess` /
  `calmar` / `ir` / `max_drawdown` / `max_dd_days` / `max_dd_recovered`（-1 = 从未恢复）/
  `x_mdd`（策略 vs 基准最大回撤差）/ `baseline_max_dd` / `dd_excess`
- **持仓行为（基于 BUY/SELL 配对成交）**：`win_rate` / `profit_factor` / `avg_pnl` /
  `max_consecutive_wins` / `max_consecutive_losses` / `avg_hold_bars` / `max_hold_bars`
- 未提供 equity 曲线时时间序列字段退化为 0.0；未提供 trades 时持仓行为字段退化为
  0.0 / inf 哨兵——Engine 路径（逐 bar 记账、不累积曲线）调用时即属此情形，
  对账（`reconcile`）只比信号 + 成交，不比 summary

> **字段单位约定**（详见 kbs/13 §1）：
> - 百分数（`cagr` / `max_drawdown` / `excess_pct` / `cagr_excess`）CLI 用 `:.2%`；
> - 金额元（`final_equity` / `baseline` / `excess` / `final_cash` / `turnover`）CLI 用 `:.2f`；
> - 无量纲比率（`calmar` / `sharpe_excess` / `sortino_excess`）CLI 用 `:.3f`；
> - 自然日（`max_dd_days`）CLI 用 `:.1f 天`。
> 任何"金额元"被 `:.2%` 打印（如 `15734173.45%`）即视为 broken，由 `tests/test_metrics_units.py` 锁定。

## 5. 主流程装配（CLI）

`backtest` 子命令（vectorized 引擎，`--device {cpu,gpu,auto}` 选后端）：

```
1. bars    = load_bars(code, start, end, warmup_days, cache_dir)   # core/data.py
2. strategy= get_strategy(strategy_name, params=strategy_params)
3. result  = run_vectorized(bars, period, warmup_until=int(start)*1_000_000,
                            strategy, params, init_cash, init_position,
                            trade_qty, scale, buy_pct, sell_pct, verbose)
4. → 打印成交行 → 打印 metrics.summarize 盈亏汇总
```

Engine 全链路（`replay_engine`，供 `replay --against-ref` 对账）：

```
1. feed      = ListBarFeed(bars)                    # bars: list[Bar]
2. account   = Account(init_cash, init_position); executor = SimulatedExecutor(...)
3. aggregator= BarAggregator(resolve_period_seconds(period), on_bars=None, warmup_until=str(...))
4. strategy  = get_strategy(strategy_name, params=strategy_params)
5. engine    = Engine(feed, aggregator, strategy, executor, verbose=True)
   → engine.run() → engine.bucket_signals
```

要点：

- 周期解析统一走 `resolve_period_seconds(period)`（int 秒），aggregator 收 **int period-seconds**；
  `config` 里不再保留周期名→秒数的周期表（旧周期常量已删）
- 预热阈值 `warmup_until` 来自 `load_bars` 的 `start_ymd`（`int(start)*1_000_000`），
  不是 bar 流对象上的属性
- 实盘 = 自供一个 `.stream()` 对象（见 08 文档）替换 `ListBarFeed`，其余装配不变

启动头打印：证券/周期/策略日期/预热天数/策略/scale/资金模式/device，便于复核每次回测的参数组合。

## 6. 一次完整回测的日志结构示例

```
证券: 159992.SZ  周期: 5m  策略日期: 20250101~20260903  预热: 365天  策略: channel_deviation  scale=1.0  资金模式: buy=0/sell=0  device=cpu
  -- 加载 102944 根 1m bar [20240102 ~ 20260903] (含预热)     ← 数据加载进度（load_bars 打印；预热段无信号输出）
...
BUY  >>> [20250106101500] 159992.SZ | UP=.. DW=.. | low_dev(L/DW)=..% high_dev(H/UP)=..%
        >> BUY  10000股 @ 1.2345  花费 12345.00  剩余资金 187655.00 持仓 210000   ← SimulatedExecutor 成交行
...
============================================================
回测盈亏汇总 (vectorized)
期末价 / 交易次数 / 期初期末资金持仓 / 策略总资产 / 不操作基线 / 盈亏比例 / CAGR / 回撤 / 胜率 ...
```

信号行字段对照见 03 文档第 5 节（info 快照）；`UP`/`DW` 是策略在 info 里塞的
（来源是 `step(state, bar, params)` 内调 `ema_channel_step`），framework 不感知。

## 7. 与 vectorized 路径的对账

`evtrade.core.replay.reconcile` 用同一份 bar 数据同时跑两条路径并对比：

```
replay_vectorized(bars, period, warmup_until, strategy_name, strategy_params, ...)
  → run_vectorized → {"sig": np.int8[N_mark1], "trades": [...], "summary": {...}}

replay_engine(bars, period, warmup_until, ...)
  → Engine.run() → {"sig": np.array(engine.bucket_signals, dtype=np.int8),
                    "trades": records (SimulatedExecutor.record_to), "summary": None}

diff_signals(k["sig"], r["sig"]) → n_diff, first_idx, ...
trades_ok = all(kt == rt for kt, rt in zip(k["trades"], r["trades"]))   # 逐笔成交
report = {"pass": sig_pass and trades_ok, ...}
```

`bucket_diff_cap` 默认 8（lenient），`EVT_RECONCILE_STRICT=1` 时 cap=0（strict，
信号分歧必须为 0）。允许 lenient 的原因：两条路径虽然调的是**同一份 `step`**，但指标
累积顺序有微妙差异——vectorized 路径在 `_aggregate_buckets` 聚合好的桶级数组上**逐桶
调 step**，Engine 路径经 BarAggregator 在桶切换时**逐 bar 聚合后驱动 step**，EMA 增量
（`ema_step` / `ema_channel_step`）的初始化/累积次序不完全相同，少量桶信号偏移属正常漂移。
成交口径则无此问题：两条路径共用 `trade_decision`（见 07 文档），逐笔 bitwise 一致。
