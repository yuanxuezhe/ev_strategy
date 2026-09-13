# 05 引擎与 CLI

> Engine / run_vectorized 装配与时序 + CLI 子命令(backtest / sweep / params) +
> 数据加载(load_bars / synthetic_bars / DB_URL) + 输出解读
>
> 相关源码: `Engine` (`evtrade/core/engine.py`)、`run_vectorized` (`evtrade/core/vectorized_engine.py`)、
> `main` (`evtrade/cli.py`)、`data` (`evtrade/core/data.py`)

## 1. 命令行结构

```bash
python -m evtrade backtest [参数]    # 单次回测 (framework 仅驱动 step)
python -m evtrade sweep    [参数]    # 参数并发扫描 + 鲁棒评分 (见 [08-选参与鲁棒性](08-选参与鲁棒性.md))
python -m evtrade params   [子命令]   # 默认参数管理 (落盘 evtrade/strategies/_defaults/)
```

> 2026-09-13: framework 不再持有资金/持仓/撮合/PnL/收益概念;
> CLI 不再有 `--init-cash --buy-pct --sell-pct --all-in --trade-qty --warmup-days --data-cache` flag;
> `replay` 子命令已删。

### 1.1 backtest 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--strategy` | `channel_deviation` | 策略 key(注册表名) |
| `--period` | `5m` | K 线周期, 任意 `数字+m/h/d`(`5m`/`7m`/`90m`/`2h`/`4h`/`6h`/`1d`/`3d`...) |
| `--start` | `20250101` | 策略起始日期 YYYYMMDD |
| `--end` | `20260903` | 策略结束日期 YYYYMMDD |
| `--params` | (策略默认) | 策略参数 `k1:v1;k2:v2`(按 `params_spec` 校验)。**资金/撮合参数**如 `init_cash`/`init_position`/`trade_qty`/`buy_pct`/`sell_pct` 全部走 `--params` |
| `--device` | `auto` | `cpu` / `gpu` / `auto`(优先 gpu, 不可用降级 cpu) |
| `--code` | `159992.SZ` | 证券代码 |
| `--verbose` | 关 | 逐根打印策略 `format_signal_line` 输出(sig != 0 时) |
| `--synthetic-days` | 0 | >0 时用合成数据(不连库) |
| `--signals-out` | 无 | 信号轨迹 CSV(`ts, sig`, 桶级对齐) |

### 1.2 策略参数解析优先级

每条命令解析策略参数时按以下顺序:

1. **`--params "k:v;..."`** 显式传入(类型自动推导 int / float / bool / str)
2. **`evtrade/strategies/_defaults/<strategy>.json`** 落盘默认(`params save` 写入)
3. **空 dict** → `VectorizedStrategy.params_spec` 的 `default` 值

实盘 / CI 机器用 `EVTRADE_DEFAULTS_DIR` 环境变量切换落盘目录, 不污染仓库。

### 1.3 sweep 参数

`--grid "名字=v1,v2"`(可多次;**仅接受策略 `params_spec` 内的参数名**;
引擎轴仅剩 `period`)、`--splits d1,d2,d3`(滚动 WFO 窗)、`--split d`(单窗, 兼容旧名)、
`--score-lambda 1.0`、`--workers`、`--device cpu|gpu|auto`、`--synthetic-days N`、
`--out`、`--top`、`--save-defaults`(跑完按 score 选最优行 + 落盘 +
git commit; 与 `--no-save-defaults` 互斥)。

`--device gpu` + `ma_crossover` + 网格 ≥32 组会自动走 batched 单次 kernel 路径
(详见 [07-PyTorch与性能](07-PyTorch与性能.md)); 该模式下 `--workers` 被忽略(打印 notice);
其它策略 / 小网格仍走 `--workers` CPU 多线程。batched 路径仅产 sig,
cash / position 由 sweep 串行 step 补。

### 1.4 params 子命令

```bash
# 落盘
python -m evtrade params save <strategy> --params "k1:v1;k2:v2"
python -m evtrade params save <strategy> --from-csv <results.csv> --rank 1

# 查看
python -m evtrade params show <strategy>
python -m evtrade params list
```

落盘文件 `evtrade/strategies/_defaults/<strategy>.json`(策略名 → JSON):

```json
{
  "strategy": "channel_deviation",
  "params": {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5},
  "saved_at": "2026-09-07T00:00:00+00:00",
  "source": {"kind": "from_csv", "csv": "sweep_results.csv", "rank": 1}
}
```

## 2. 典型命令

```bash
# 标准 CPU 全速回测
python -m evtrade backtest --strategy channel_deviation --device cpu \
    --period 5m --start 20250101 --end 20260903

# GPU 回测 (需 torch CUDA; 无 CUDA 会抛错, 建议 --device auto)
python -m evtrade backtest --strategy channel_deviation --device auto \
    --period 5m --start 20250101 --end 20260903

# GPU 机器一次性安装 (RTX 50 / Blackwell 需 cu128)
uv sync --extra gpu              # 锁住 torch==2.9.0+cu128
bash scripts/sync-torch-cu.sh    # 防御性 helper (lockfile 被 uv 重生时恢复)

# 小段快速验证参数 (含资金/撮合参数, 全部走 --params)
python -m evtrade backtest --strategy channel_deviation \
    --params "tf1:21;low1:1.0;init_cash:200000;trade_qty:10000" \
    --period 5m --start 20250601 --end 20250901

# 换标的 + 1h 周期 + 更苛刻偏离阈值
python -m evtrade backtest --strategy channel_deviation --code 513120.SH \
    --period 1h --params "low1:2.0;low2:1.2;high1:2.0;high2:0.8"

# 无库体验 (合成数据)
python -m evtrade backtest --strategy channel_deviation --synthetic-days 120 \
    --period 5m --start 20250601 --end 20250701

# 网格扫描
python -m evtrade sweep --strategy channel_deviation --device auto \
    --grid "low1=1.2,1.5,2.5" --grid "low2=0.5,1.0,1.5" \
    --splits 20250901 20260301 --out results.csv

# 小段回测观察行为
python -m evtrade backtest --strategy channel_deviation --period 5m \
    --start 20250601 --end 20250615 --verbose
```

## 3. Engine 装配与时序

### 3.1 Engine 装配

```python
engine = Engine(feed, strategy, aggregator=None, verbose=True)
```

构造时做三件事:

1. 持有 3 个组件引用(bar 流对象 / strategy / aggregator)
2. **初始化策略 state**: `self._state = strategy.init_state(strategy.params)` —— state 由引擎持有、跨调用持续
3. **接线**: `self.aggregator.on_bars = self.on_bars` —— 聚合器桶回调改道到引擎

> framework **不再持有**任何业务字段(cash / position / trades / metrics 等)。
> 策略资金/持仓/账本/PnL 全部在策略 step 内部 + 策略 `@dataclass state` 字段内。

### 3.2 `on_bars` —— 桶 CLOSE 时驱动策略

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
  if sig != 0 and self.verbose:
      info = getattr(strategy, "_last_info", None)
      print(strategy.format_signal_line(rec["ts"], sig, info))
```

要点:

- **桶 CLOSE 语义**: 信号触发时点 = 桶**切换**那一刻(看到当前根 `ts` ≠ 上一根 `ts`),
  价格 = 上一桶 finalized `close`。与 vectorized 路径在"桶级 finalized OHLCV"上算指标完全对齐。
- **mark=0(预热)不驱动**: `_process_bucket` 直接 return, 不调策略、不成交。
- **信息流**: `step(state, bar, params)` 内部维护 state(EMA 增量 + FSM + cash/position/trades),
  state 由引擎持有(`Engine._state`)跨调用持续。vectorized 路径同样循环调 step,
  state 在循环内持续。两种入口走同一份 `step` 算法。
- `info` dict 字段集由策略自由控制, framework 不命名也不假设。

### 3.3 `run()` 与收尾

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
- `run()` 返回处理过的 1m bar 根数, 无其它产出。
- framework 不汇总任何 PnL / 收益 / 账本字段。

## 4. vectorized 引擎主流程

```
backtest:
1. bars    = load_bars(code, start, end)         # core/data.py
2. strategy= get_strategy(strategy_name, params=strategy_params)
3. result  = run_vectorized(bars, period, warmup_until=int(start)*1_000_000,
                            strategy, params, verbose)
4. → 打印信号行 (verbose=True) → 打印 final_state 透传
```

返回 = `{sig, buckets, final_state}`, framework 仅透传策略 step 末尾 state, 不读字段。

要点:

- 周期解析统一走 `resolve_period_seconds(period)`(int 秒); aggregator 收 int period-seconds
- 预热阈值 `warmup_until` 来自 `int(start) * 1_000_000`
- 实盘接入 = 自供一个 `.stream()` 对象(见 [02-数据与桶 §bar 流契约](02-数据与桶.md))
  替换 `ListBarFeed`, 其余装配不变

启动头打印: 证券 / 周期 / 策略日期 / 策略 / device, 便于复核每次回测的参数组合。

## 5. 数据加载

### 5.1 `load_bars` —— MySQL 历史行情

```python
load_bars(code, start_ymd, end_ymd, warmup_days=365,
          cache_dir=None, use_cache=True, db_url=None, verbose=True) -> dict
```

返回 numpy 数组字典 `{stime:int64, open/high/low/close/volume:float64}`, `stime` 为
14 位 `YYYYMMDDHHmmss` 整数、升序。

要点:

- **单次全量拉取**: `_fetch` 用 `pd.read_sql` 一次 fetchall 取回 `[start_ymd - warmup_days, end_ymd]`
  全部 1m bar, 再在本地排序/转数组。
- **npz 缓存**: 传 `cache_dir` 时, 缓存键 = `(code, 预热起点, end)`, 命中则零数据库开销。
- **连接保活**: `create_engine(db_url or DB_URL, pool_pre_ping=True, pool_recycle=3600)`
- 无数据时抛 `RuntimeError` 并提示检查 `EVTRADE_DB_URL` / 网络, 或改用 `--synthetic-days`。

SQL(`TABLE = "minute_bars"`, 表名 `EVTRADE_TABLE` 可覆写):

```sql
SELECT stock_code, stime, open, high, low, close, volume
FROM minute_bars
WHERE stime >= '{warmup_start}000000' AND stime <= '{end_ymd}235959'
  AND stock_code = '{code}'
ORDER BY stime ASC
```

### 5.2 `synthetic_bars` —— 合成数据(无库体验/测试)

```python
synthetic_bars(days=30, start_ymd="20250101", seed=42,
               s0=10.0, jump_prob=0.02, odd_seconds_prob=0.1) -> list[Bar]
```

确定性合成 1m `Bar` 列表(seed 固定可复现): 均值回复随机游走 + 偶发跳空(制造通道偏离),
A 股时段 09:30-11:29 / 13:00-14:59 每天 240 根, 含周末(保证跨日/跨月边界覆盖),
少量非整分秒(练习桶边界逻辑)。CLI `--synthetic-days N` 走此路径, 转成与 `load_bars`
同构的数组字典后喂给 vectorized 引擎。

### 5.3 数据库连接串(`evtrade/core/data.py::DB_URL` 默认值)

默认连接串 `mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4`,
即 `data.DB_URL` 的当前实现值(与 spec `R: Market data DB connection has a sane default` 同源)。
可用环境变量 `EVTRADE_DB_URL` 覆盖 —— 临时切库 / 离线 / 测时 escape 入口。

- 主机 `192.168.10.2:33066`, 库 `evtrade`, 用户 `EvTrade`, 密码 `p@ssw0rd`(URL 编码为 `p%40ssw0rd`)
- 表 `minute_bars` 字段: `stock_code, stime, open, high, low, close, volume`
- `stime` 为 14 位字符串 `YYYYMMDDHHmmss`, 字符串比较即时间序, SQL 范围过滤因此有效
- 建议确认 `stime` 上有索引(按 stime 范围过滤 + 排序)
- 默认 DB 不可达时 `cli.py::_run_backtest` 会接住 SQLAlchemy `OperationalError` 并打印一行中文提示
  (含 `192.168.10.2:33066` 与 `EVTRADE_DB_URL`), 再以 `SystemExit(2)` 退出

### 5.4 bar 流契约

Engine **不依赖任何具体源类**, 只要求传入对象实现 `stream() -> Iterator[Bar]`
(按时间从旧到新 yield 统一 `Bar` 结构)。

参考实现 `core/_harness.ListBarFeed`:

```python
class ListBarFeed:
    """Bar 列表 -> Bar 流 (passthrough)"""
    def __init__(self, bars: list): self.bars = bars
    def stream(self) -> Iterator: yield from self.bars
```

实盘接入 = 自己写一个 `stream()` 对象, 把实时行情源(推送/轮询)转成 `Bar` 逐根 yield。
"先历史后实时"的组合(先播含预热的历史、再接实时源)也只需在自己的 `stream()` 里
`yield from` 两段数据即可完成, framework 不再提供组合类。

## 6. 输出解读

### 6.1 数据加载进度(`load_bars` 打印)

```
  -- 加载 614400 根 1m bar [20240102 ~ 20260903] (含预热)
```

数据由 `core/data.load_bars` 单次全量拉取(一次 `fetchall` 后本地处理)。
长时间卡在加载 = SQL 慢(查 `stime` 索引)。

### 6.2 信号行(verbose 引擎打印, 仅有信号时)

```
BUY  >>> [20250106101500] | UP=1.2345 DW=1.2340 | low_dev(L/DW)=2.31% high_dev(H/UP)=0.45%
```

- 时间戳 = **桶 CLOSE 时的桶右端点 ts**(与 vectorized 路径同口径)。
- 数值按 4 位小数显示(`fmt` 函数)。
- 字段由策略 `format_signal_line` hook 决定; framework 不假设具体键集。

### 6.3 框架输出(2026-09-13 重构后)

framework 不再汇总 PnL/收益/回撤/胜率等任何业务字段。
`run_vectorized` 返回 = `{sig, buckets, final_state}`, CLI 仅打印:

```
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

要看 PnL / 收益 / 绩效请在策略 step / state 内自加字段, framework 不输出。

## 7. 运行前检查清单

1. 网络可达 `192.168.10.2:33066`, 账号 `EvTrade` 可读 `evtrade` 库
   (`DB_URL` 在 `core/data.py` 默认, 可环境变量 `EVTRADE_DB_URL` 覆盖)
2. `minute_bars` 表中存在目标 `stock_code`, 且覆盖 `[start - prewarm, end]` 的数据
3. `stime` 列建议有索引(`_fetch` 单次全量 SQL 按 `stime` 范围 + `stock_code` + 排序)
4. `low1 > low2`、`high1 > high2`(否则锁存失效); 参数经 `_resolve_params` 按 `params_spec` 校验
5. 数据量估算: 1 年 1m ≈ 24 万根/标的(交易时段), 乘以回测年数; 首次运行留意磁盘与内存
6. GPU 路径需 `torch`(CUDA 版 wheel); 缺失时 `--device gpu` 抛错(提示改用 `auto`/`cpu`),
   `--device auto` 自动降级 cpu 并打 warning

## 8. 常见问题(FAQ)

**Q: 完全没有信号?**
确认预热充分(`--period 1d` 且 start 前历史不足一年会导致指标就绪晚);
观察偏离量分布(把 `--low1` 临时调小如 0.5 看是否开始出信号),
区分阈值过严 vs 数据 / 代码问题。

**Q: 信号时间戳为什么是 10:15 而我以为是 10:10~10:14 的行情?**
桶 ts 是前开后闭区间的右端点, `[10:15]` 桶覆盖 `(10:10, 10:15]`。
详见 [02-数据与桶 §compute_bucket_general](02-数据与桶.md)。

**Q: 能回测多个标的吗?**
单次运行单标的(`load_bars` 按 `stock_code` 过滤)。多标的需循环调用或自行扩展数据源。

**Q: 中途 Ctrl+C 会丢汇总吗?**
vectorized 路径为一次性批量计算, Ctrl+C 直接中断(未跑完无 final_state)。
Engine 路径 `run()` 捕获 KeyboardInterrupt 后仍执行 `aggregator.flush()` + 末桶收尾,
基于已处理数据结束。

**Q: GPU 与 CPU 结果对不上?**
通常发生在 torch CUDA 不可用、auto 回退 cpu 时。显式传 `--device cpu` / `--device gpu` 排查;
两条路径调同一份 `step`, 少量桶信号漂移属正常。

## 9. framework 不汇总

- ❌ 资金 / 持仓 / 撮合 / 记账(策略 step 内做)
- ❌ PnL / 收益 / 风险指标(策略 state 内自维护, framework 不识别)
- ❌ `metrics.summarize` 函数(已下线)
- ❌ `reconcile` 对账(已下线)
- ❌ `replay` 子命令 + `--against-ref`

framework 唯一产出 = 桶聚合 + 调 `step` 循环 + 累计 `final_state`。
