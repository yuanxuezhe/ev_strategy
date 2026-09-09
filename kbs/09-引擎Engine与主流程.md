# 09 引擎 Engine 与主流程

> 相关源码：`Engine`（`evtrade/core/engine.py`）、向量化引擎
> （`evtrade/core/vectorized_engine.py`）、默认配置（`evtrade/core/config.py`）、
> `main`（`evtrade/cli.py`）。
> 本文档为 2026-09-09 统一 CPU/GPU 重构版（DSL/numba 已删; 桶 CLOSE 语义; xp 化指标）。

## 1. Engine 的装配（构造函数）

```python
engine = Engine(feed, aggregator, strategy, executor, verbose=True)
```

构造时只做两件事：

1. 持有四个组件引用（feed / aggregator / strategy / executor）
2. **接线**：`self.aggregator.on_bars = self.on_bars` —— 聚合器每根 bar 的回调改道到引擎（`main` 里构造 aggregator 时传的 `on_bars=None` 就是为了在这里被覆盖）

> framework **不再持有** `tf1` / `EMAChannel` / `_pushed` / `state_spec` 等任何策略相关字段。
> `tf1` 由策略 `params_spec` 自声明；EMA 增量状态由策略 instance 字段 (`self._up_st / self._dw_st`) 自维护。

## 2. `on_bars` —— 桶 CLOSE 时驱动策略

```
on_bars(bars):
  cur = bars[-1]

  # 桶切换: 上一桶 finalize 时, 用上一桶的 finalized OHLCV 驱动策略
  if self._last_cur is not None and self._last_cur["ts"] != cur["ts"]:
      prev = self._last_cur                  # 上一桶 finalize 后的快照
      if prev.get("mark", 1) == 1:
          price = float(prev["close"])
          executor.update_price(price)
          bar = {ts, o, h, l, c, v, mark=1} from prev
          sig = strategy.compute_signals_for_one_bar(np, bar, params)
          bucket_signals.append(sig)
          if sig != 0:
              signal = {1: "BUY", -1: "SELL"}.get(sig)
              if signal: executor.trade(signal, price, prev.ts)
          if verbose and signal:
              print(strategy.format_signal_line(prev.ts, sig, info))

  self._last_cur = dict(cur)                # 记录 cur 供下一根 bar 时检测切换
```

要点：

- **桶 CLOSE 语义**：信号触发时点 = 桶**切换**那一刻（看到当前根 `ts` ≠ 上一根 `ts`），价格 = 上一桶 finalized `close`。
  与 vectorized 路径在"桶级 finalized OHLCV"上算指标完全对齐（replay/reconcile 测试通过 `bucket_diff_cap=8` 容忍 EMA 累积漂移）。
- **mark=0（预热）不驱动**：若上一桶在预热期（mark=0），不调策略、不成交。
- **信息流**：`compute_signals_for_one_bar` 内部可维护 instance state（增量 EMA、FSM）;
  `compute_signals` 的批量路径无 instance state（FSM 是函数内局部）。
  两种入口通过共享 `_fsm_step(state, ...)` 保证信号 bitwise 一致。
- `info` dict 字段集由策略自由控制，framework 不命名也不假设。

## 3. `run()` 与收尾

```python
def run(self):
    total = 0
    try:
        for bar in self.feed.stream():
            self.aggregator.update(bar)
            total += 1
        self.aggregator.flush()
        self._flush_final_bucket()           # flush 后手动触发最后一个桶的信号
    except KeyboardInterrupt:
        print("\n已停止")
    if self.verbose:
        print(f"\n完成, 共处理 {total} 根 1m bar")
    return total
```

- `KeyboardInterrupt` 被捕获打印"已停止"。`flush()` + `_flush_final_bucket()` 均在 `try` 外（但也未捕获 Ctrl+C 后的其它异常），所以 Ctrl+C 后会走到 flush（若流已被中断则跳过）。
- `_flush_final_bucket` 用最后一个桶的 finalized OHLCV 驱动策略一次（防止漏掉最后一桶的信号）。

## 4. `print_summary` —— 回测盈亏汇总

口径（详见 07 文档）：

```
期末价   = account.last_price          （最后一根策略期 bar 的 close）
策略总资产 = cash + position × 期末价
不操作基线 = init_cash + init_position × 期末价
盈亏差额   = 策略总资产 − 基线
盈亏比例   = 盈亏差额 / 基线 × 100
```

输出项：期末价、交易次数、期初/期末资金与持仓、持仓市值、策略总资产、基线、盈亏差额与比例。
该口径消除了标的本身涨跌的影响，衡量的是**择时的贡献**；未含手续费/滑点。

> **2026-09-09 变化**: `vectorized_engine` 路径下, 盈亏汇总由 `metrics.summarize(equity_curve, baseline_curve, ...)` 给出 16 字段
> (`final_cash / final_position / final_equity / turnover / n_trades / n_buy / n_sell / cagr / sharpe / sharpe_excess / sortino / sortino_excess / calmar / max_dd_days / max_dd_recovered / x_mdd / max_drawdown`)。
> Engine 路径仅给 6 项基础汇总 (`final_cash / final_position / final_equity / n_trades / baseline / diff / pct`),
> 因为其逐 bar 路径不累积 `equity_curve`（仅 `account` 记账）。对账路径 (`reconcile`) 只比信号 + 成交, 不比 summary。

> **字段单位约定**（详见 kbs/13 §1，2026-09-09 钉死）：
> - 百分数（`cagr` / `max_drawdown` / `excess_pct` / `ann_excess_pct`）CLI 用 `:.2%`；
> - 金额元（`final_equity` / `baseline` / `excess` / `final_cash` / `turnover`）CLI 用 `:.2f`；
> - 无量纲比率（`calmar` / `sharpe_excess` / `sortino_excess`）CLI 用 `:.3f`；
> - 自然日（`max_dd_days`）CLI 用 `:.1f 天`。
> 任何"金额元"被 `:.2%` 打印（如 `15734173.45%`）即视为 broken，由 `tests/test_metrics_units.py` 锁定。

## 5. `main` 的五步装配（CLI `backtest` 子命令）

```
1. feed      = MySQLBacktestFeed(code, start, end, step_days, delay, verbose)
2. account   = Account(INIT_CASH, INIT_POSITION); executor = SimulatedExecutor(account, trade_qty)
3. aggregator= BarAggregator(PERIODS[args.period], on_bars=None, warmup_until=feed.warmup_until)
4. strategy  = get_strategy(strategy_name, params=strategy_params)   # 默认 device="cpu"
5. engine    = Engine(feed, aggregator, strategy, executor, verbose=True)
   → 打印配置头 → engine.run() → engine.print_summary()
```

启动头打印：证券/周期/策略日期/预热起点/分段/sleep 开关/四阈值+tf1（tf1 由策略 `--params` 传入后由 CLI 打印），便于复核每次回测的参数组合。

## 6. 一次完整回测的日志结构示例

```
证券: 159992.SZ  周期: 5m  策略日期: 20250101~20260903  预热起点: 20240102  ...
  -- 段 20240102~20240108 处理 8231 根, 累计 8231        ← Feed 分段进度（预热段无信号输出）
  ...
BUY  >>> [20250106101500] 159992.SZ | UP=.. DW=.. | low_dev(L/DW)=..% high_dev(H/UP)=..%
        >> BUY  10000股 @ 1.2345  花费 12345.00  剩余资金 187655.00 持仓 210000   ← SimulatedExecutor 成交行
...
============ 回测盈亏汇总 ============
期末价 / 交易次数 / 期初期末资金持仓 / 策略总资产 / 不操作基线 / 盈亏差额 / 盈亏比例
```

信号行字段对照见 03 文档第 5 节（info 快照）；`UP`/`DW` 是策略在 info 里塞的（来源是 `compute_signals_for_one_bar` 内调 `ema_channel_current`），framework 不感知。

## 7. 与 vectorized_engine 的对账

`evtrade.core.replay.reconcile` 用同一份 1m bar 流同时跑两条路径并对比：

```
replay_vectorized(bars, period, warmup_until, strategy_name, strategy_params, device)
  → run_vectorized → {"sig": np.int8[N_mark1], "trades": [...], "summary": {...}}

replay_engine(bars, period, warmup_until, tf1, strategy_name, strategy_params)
  → Engine.run() → {"sig": np.array(engine.bucket_signals, dtype=np.int8),
                    "trades": executor.records, "summary": None}

diff_signals(k["sig"], r["sig"]) → n_diff, first_idx, ...
trades_ok = all(kt == rt for kt, rt in zip(k["trades"], r["trades"]))
report = {"pass": sig_pass and trades_ok, ...}
```

`bucket_diff_cap` 默认 8（lenient），`EVT_RECONCILE_STRICT=1` 时 cap=0（strict），
是因为 EMA 通道在两条路径上 EMA 累积顺序有微妙差异（`xp_ema_channel` 一次性算 vs 增量 `ema_push` 逐 bar 推），
少量桶信号偏移属正常漂移。
