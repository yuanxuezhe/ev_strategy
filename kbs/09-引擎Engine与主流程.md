# 09 引擎 Engine 与主流程

> 相关源码：`Engine`（`mysql_analyze_demo.py:516-625`）、默认配置（`mysql_analyze_demo.py:630-632`）、`main`（`mysql_analyze_demo.py:637-680`）

## 1. Engine 的装配（构造函数，`mysql_analyze_demo.py:523-536`）

```python
engine = Engine(feed, aggregator, strategy, executor, tf1=TF1, verbose=True)
```

构造时做三件事：

1. 持有四个组件引用 + `tf1`
2. 创建自己的增量通道 `self.ema_ch = EMAChannel(tf1)`，并置 `_pushed = 0`（已 push 的闭合桶数游标）
3. **接线**：`self.aggregator.on_bars = self.on_bars` —— 聚合器每根 bar 的回调改道到引擎（`main` 里构造 aggregator 时传的 `on_bars=None` 就是为了在这里被覆盖）

## 2. `_sync_ema` —— 幂等的指标喂入（`mysql_analyze_demo.py:538-544`）

`on_bars` 每根 1m bar 都被调用，但 EMA 只在**桶闭合**时才前进。用游标 `_pushed` 记录"已把第几个闭合桶 push 进 EMA"：

```python
def _sync_ema(self, closed_bars):
    n = len(closed_bars)
    while self._pushed < n:
        b = closed_bars[self._pushed]
        self.ema_ch.push(b["high"], b["low"])
        self._pushed += 1
```

- 同一根 1m bar 触发的多次回调、同一桶内的反复回调，`_pushed == len(closed)` 时空转，不重复 push
- **预热期（mark=0）也照常 push**——这正是预热的意义：mark=1 开始时指标已就绪（`mysql_analyze_demo.py:554` 注释）

## 3. `on_bars` —— 每根 bar 的处理流水线（`mysql_analyze_demo.py:546-589`）

```
on_bars(bars):
  closed = bars[:-1]; cur = bars[-1]
  _sync_ema(closed)                       # 指标前进（含预热期）
  if cur["mark"] == 0: return             # 预热桶：到此为止
  price = cur["close"]
  executor.account.last_price = price     # 更新估值价
  up, dw = ema_ch.channel(cur.high, cur.low)   # O(1) 通道值（含当前桶）
  signal, info = strategy.check(cur, up, dw)
  if signal: executor.trade(signal, price, cur.ts)   # 当根最新 close 成交
  if verbose and signal: print(信号行)     # 只在有信号时打印明细
```

要点：

- 预热/策略期的分界只挡"策略与交易"，不挡指标累积
- 交易时点 = 信号所在的**当前未闭合桶**，价格 = 桶内最新 close（= 最近一根 1m bar 收盘）
- verbose 下非信号 bar 完全静默，只有 Feed 的分段进度和信号行、成交行

## 4. `run()` 与收尾（`mysql_analyze_demo.py:591-602`）

```python
for bar in feed.stream():
    aggregator.update(bar)
aggregator.flush()          # 闭合最后一个桶并回调一次
```

- `KeyboardInterrupt` 被捕获打印"已停止"——**Ctrl+C 中断也会走到 flush**（try 只包住 for 循环，flush 在 try 外……实际代码 flush 在 try/except 之后执行，中断后同样收尾），但中断时未处理的 feed 剩余段不再拉取
- 返回总处理根数（1m bar 计数）

> 精确读法：`try: for ... except KeyboardInterrupt: print` 之后 `aggregator.flush()` 与 `print` 均无条件执行，所以 Ctrl+C 后仍会闭合末桶并进入 `main` 的 `print_summary()`。

## 5. `print_summary` —— 回测盈亏汇总（`mysql_analyze_demo.py:604-625`）

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

## 6. `main` 的五步装配（`mysql_analyze_demo.py:637-680`）

```
1. feed      = MySQLBacktestFeed(code, start, end, step_days, delay, verbose)
2. account   = Account(INIT_CASH, INIT_POSITION); executor = SimulatedExecutor(account, trade_qty)
3. aggregator= BarAggregator(PERIODS[args.period], on_bars=None, warmup_until=feed.warmup_until)
4. strategy  = ChannelDeviationStrategy(low1, low2, high1, high2)
5. engine    = Engine(feed, aggregator, strategy, executor, tf1)
   → 打印配置头 → engine.run() → engine.print_summary()
```

启动头打印：证券/周期/策略日期/预热起点/分段/TF1/sleep 开关/四阈值，便于复核每次回测的参数组合。

## 7. 一次完整回测的日志结构示例

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

信号行字段对照见 03 文档第 5 节（info 快照）。
