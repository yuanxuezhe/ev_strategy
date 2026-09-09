# 09 引擎 Engine 与主流程

> 相关源码：`Engine`（`evtrade/core/engine.py`）、默认配置（`evtrade/core/config.py`）、`main`（`evtrade/cli.py`）
> 历史背景：Engine 原型从 `mysql_analyze_demo.py:516-625` 迁移，已**去除 EMA 算 EMA 的历史包袱**——framework 不持有 EMA 状态，指标计算由策略 DSL body 通过 `evtrade/indicators/` 增量 API 自维护。

## 1. Engine 的装配（构造函数，`engine.py:40-53`）

```python
engine = Engine(feed, aggregator, strategy, executor, verbose=True)
```

构造时只做两件事：

1. 持有四个组件引用（feed / aggregator / strategy / executor）
2. **接线**：`self.aggregator.on_bars = self.on_bars` —— 聚合器每根 bar 的回调改道到引擎（`main` 里构造 aggregator 时传的 `on_bars=None` 就是为了在这里被覆盖）

> framework **不再持有** `tf1` / `EMAChannel` / `_pushed` 等任何指标相关字段。`tf1` 由策略 `params_spec` 自声明；EMA 增量状态由策略 `state_spec` 自声明并在 DSL body 调 `evtrade.indicators.ema_channel_push` 维护。

## 2. `on_bars` —— 每根 bar 的处理流水线（`engine.py:63-96`）

```
on_bars(bars):
  closed = bars[:-1]; cur = bars[-1]
  if cur["mark"] == 0: return             # 预热桶: 仅调 strategy.check(cur,{}) 让策略预热自己的指标状态
  price = cur["close"]
  executor.account.last_price = price     # 更新估值价
  signal, info = strategy.check(cur, {})  # 策略内部从 cur + state_spec 算指标
  if signal: executor.trade(signal, price, cur.ts)   # 当根最新 close 成交
  if verbose and signal: print(信号行)     # 只在有信号时打印明细
```

要点：

- 预热/策略期的分界只挡"成交"，**不挡**策略状态推进；策略在 mark=0 期仍可调 `strategy.check(cur, {})` 来累积自己的指标状态（具体由策略自己决定要不要在 mark=0 调 `_push` / `_current`）
- 交易时点 = 信号所在的**当前未闭合桶**，价格 = 桶内最新 close（= 最近一根 1m bar 收盘）
- verbose 下非信号 bar 完全静默，只有 Feed 的分段进度和信号行、成交行
- `info` dict 字段集由策略自由控制，framework 不命名也不假设

## 3. `run()` 与收尾（`engine.py:98-109`）

```python
for bar in feed.stream():
    aggregator.update(bar)
aggregator.flush()          # 闭合最后一个桶并回调一次
```

- `KeyboardInterrupt` 被捕获打印"已停止"——**Ctrl+C 中断也会走到 flush**（try 只包住 for 循环，flush 在 try 外……实际代码 flush 在 try/except 之后执行，中断后同样收尾），但中断时未处理的 feed 剩余段不再拉取
- 返回总处理根数（1m bar 计数）

> 精确读法：`try: for ... except KeyboardInterrupt: print` 之后 `aggregator.flush()` 与 `print` 均无条件执行，所以 Ctrl+C 后仍会闭合末桶并进入 `main` 的 `print_summary()`。

## 4. `print_summary` —— 回测盈亏汇总（`engine.py:111-132`）

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

## 5. `main` 的五步装配（`cli.py:302-334` / `_run_kernel` 类似）

```
1. feed      = MySQLBacktestFeed(code, start, end, step_days, delay, verbose)
2. account   = Account(INIT_CASH, INIT_POSITION); executor = SimulatedExecutor(account, trade_qty)
3. aggregator= BarAggregator(PERIODS[args.period], on_bars=None, warmup_until=feed.warmup_until)
4. strategy  = ChannelDeviationStrategy(tf1, low1, low2, high1, high2)  # tf1 由策略私有参数
5. engine    = Engine(feed, aggregator, strategy, executor, verbose=True)
   → 打印配置头 → engine.run() → engine.print_summary()
```

启动头打印：证券/周期/策略日期/预热起点/分段/sleep 开关/四阈值+tf1（tf1 由策略 `--params` 传入后由 CLI 打印），便于复核每次回测的参数组合。

## 6. 一次完整回测的日志结构示例

```
证券: 159992.SZ  周期: 5m  策略日期: 20250101~20260903  预热起点: 20240102  ...
  -- 段 20240102~20240108 处理 8231 根, 累计 8231        ← Feed 分段进度（预热段无信号输出）
  ...
BUY  >>> [20250106101500] 159992.SZ | O:.. H:.. L:.. C:.. | vol:.. x5 | UP=.. DW=.. | low_dev=..% ... low_hit_prev=True ...
        >> BUY  10000股 @ 1.2345  花费 12345.00  剩余资金 187655.00 持仓 210000   ← SimulatedExecutor 成交行
...
============ 回测盈亏汇总 ============
期末价 / 交易次数 / 期初期末资金持仓 / 策略总资产 / 不操作基线 / 盈亏差额 / 盈亏比例
```

信号行字段对照见 03 文档第 5 节（info 快照）；`UP`/`DW` 是策略在 info 里塞的（来源是 DSL body 内调 `ema_channel_current`），framework 不感知。