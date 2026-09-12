# 09 引擎 Engine 与主流程

> 相关源码：`Engine`（`evtrade/core/engine.py`）、向量化引擎
> （`evtrade/core/vectorized_engine.py`）、`main`（`evtrade/cli.py`）。
> 本文档为 2026-09-13 重构版：framework 仅驱动 `step`；资金/持仓/撮合/PnL/收益 全部下放到策略。

## 1. Engine 的装配（构造函数）

```python
engine = Engine(feed, strategy, aggregator=None, verbose=True)
```

构造时做三件事：

1. 持有 3 个组件引用（bar 流对象 / strategy / aggregator）
2. **初始化策略 state**：`self._state = strategy.init_state(strategy.params)` —— state 由引擎持有、跨调用持续
3. **接线**：`self.aggregator.on_bars = self.on_bars` —— 聚合器桶回调改道到引擎

> framework **不再持有**任何业务字段（cash / position / trades / metrics 等）。
> 策略资金/持仓/账本/PnL 全部在策略 step 内部 + 策略 `@dataclass state` 字段内。

Engine 是唯一的装配点；aggregator 的 `on_bars` 由 Engine 覆盖。

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
  self._state, sig = strategy.step(self._state, {ts,o,h,l,c,v,mark}, strategy.params)
  bucket_signals.append(sig)
  if sig != 0 and self.verbose:
      info = getattr(strategy, "_last_info", None)
      print(strategy.format_signal_line(rec["ts"], sig, info))
```

要点：

- **桶 CLOSE 语义**：信号触发时点 = 桶**切换**那一刻（看到当前根 `ts` ≠ 上一根 `ts`），价格 = 上一桶 finalized `close`。与 vectorized 路径在"桶级 finalized OHLCV"上算指标完全对齐。
- **mark=0（预热）不驱动**：`_process_bucket` 直接 return，不调策略、不成交。
- **信息流**：`step(state, bar, params)` 内部维护 state（EMA 增量 + FSM + cash/position/trades），state 由引擎持有（`Engine._state`）跨调用持续。vectorized 路径（`_compute_signals`）同样循环调 step，state 在循环内持续。两种入口走同一份 `step` 算法。
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

- `KeyboardInterrupt` 被捕获打印"已停止"。
- 最后一桶补驱动用最后一个桶的 finalized OHLCV 驱动策略一次。
- `run()` 返回处理过的 1m bar 根数，无其它产出。
- framework 不汇总任何 PnL/收益/账本字段。

## 4. framework 不汇总

2026-09-13 重构后 framework 不再做：

- ❌ 资金/持仓/撮合/记账（策略 step 内做）
- ❌ PnL / 收益 / 风险指标（策略 state 内自维护，framework 不识别）
- ❌ `metrics.summarize` 函数（已下线）
- ❌ `reconcile` 对账（已下线）
- ❌ `replay` 子命令 + `--against-ref`

framework 唯一产出 = 桶聚合 + 调 `step` 循环 + 累计 `final_state`。

## 5. 主流程装配（CLI）

`backtest` 子命令（vectorized 引擎）：

```
1. bars    = load_bars(code, start, end)         # core/data.py
2. strategy= get_strategy(strategy_name, params=strategy_params)
3. result  = run_vectorized(bars, period, warmup_until=int(start)*1_000_000,
                            strategy, params, verbose)
4. → 打印信号行（verbose=True） → 打印 final_state 透传
```

返回 = `{sig, buckets, final_state}`，framework 仅透传策略 step 末尾 state，不读字段。

要点：

- 周期解析统一走 `resolve_period_seconds(period)`（int 秒）；aggregator 收 int period-seconds
- 预热阈值 `warmup_until` 来自 `int(start) * 1_000_000`
- 实盘接入 = 自供一个 `.stream()` 对象（见 08 文档）替换 `ListBarFeed`，其余装配不变

启动头打印：证券/周期/策略日期/策略/device，便于复核每次回测的参数组合。

## 6. 一次完整回测的日志结构示例（2026-09-13 重构）

```
证券: 159992.SZ  周期: 5m  策略日期: 20250101~20260903  策略: channel_deviation  device=cpu
  -- 加载 102944 根 1m bar [20240102 ~ 20260903] (含预热)
...
BUY  >>> [20250106101500] 159992.SZ | UP=.. DW=.. | low_dev(L/DW)=..% high_dev(H/UP)=..% cash=.. pos=..
============================================================
回测完成 (vectorized; 仅驱动 step; framework 不汇总 PnL/收益)
============================================================
信号轨迹:    498 桶 (mark=1 段)
桶数:        500 (含预热)
引擎耗时:    1234.5 ms (24000 根 1m bar, cpu)
------------------------------------------------------------
策略 final_state (framework 仅透传):
  cash = 0.0
  position = 16671.6
  trades = <list>
```

信号行字段由策略 `format_signal_line` hook 决定；framework 不假设具体键集。
`cash` / `position` 等业务字段仅在策略自己打到 info 时显示。

## 7. 与 vectorized 路径的一致性

两条入口（`Engine.on_bars` 逐 bar / `run_vectorized` 桶级批量）调的是**同一份 `step` 函数**：
- signal 序列在桶切换时点同口径（桶 CLOSE 语义）
- 同一份 `@dataclass state` 跨调用持续，跨路径 bitwise 一致（少量桶信号漂移由策略 step 实现吸收）

`reconcile` 对账已下线（2026-09-13）；要验证两条路径一致，运行两次（`backtest` vs `Engine.run`），比较 `bucket_signals` 与 `sig` 即可。