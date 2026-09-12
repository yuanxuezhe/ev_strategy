# 08 行情数据加载与 bar 流

> **2026-09-13 重要更新**: framework 不再持有资金/持仓/撮合/PnL/收益概念; 策略 `step` 内部自管。`Account` / `Executor` / `SimulatedExecutor` / `trade_decision` / `metrics.summarize` 已下线; `core/metrics` / `core/replay` / `core/permutation` / `core/config` 已删; `replay` 子命令 + `--against-ref` 已下线; `--init-cash --buy-pct --sell-pct --all-in --trade-qty --warmup-days --data-cache` CLI flag 已删 (走 `--params`).

> 相关源码：`load_bars` / `synthetic_bars` / `_fetch`（`evtrade/core/data.py`）、
> bar 流契约与 `ListBarFeed`（`evtrade/core/_harness.py`）、`Bar`（`evtrade/primitives.py`）
>
> 2026-09 重构（consolidate-simplify-core）后，旧的行情源子包 `evtrade.feeds`
> （含分段查询源、组合源、注册表等）已整体删除；数据加载唯一入口 =
> `core/data.py`，Engine 消费任意鸭子类型 bar 流对象。

## 1. 数据源 —— `core/data.py::load_bars` / `synthetic_bars`

### 1.1 `load_bars` —— MySQL 历史行情（回测当前实现）

```python
load_bars(code, start_ymd, end_ymd, warmup_days=365,
          cache_dir=None, use_cache=True, db_url=None, verbose=True) -> dict
```

返回 numpy 数组字典 `{stime:int64, open/high/low/close/volume:float64}`，`stime` 为
14 位 `YYYYMMDDHHmmss` 整数、升序。

要点：

- **单次全量拉取**：`_fetch` 用 `pd.read_sql` 一次 fetchall 取回 `[start_ymd - warmup_days, end_ymd]`
  全部 1m bar，再在本地排序/转数组——不再分段流式查询（旧的分段区间生成器已删）
- **npz 缓存**：传 `cache_dir` 时，缓存键 = `(code, 预热起点, end)`，命中则零数据库开销
  （CLI `--data-cache` 指定目录）
- **连接保活**：`create_engine(db_url or DB_URL, pool_pre_ping=True, pool_recycle=3600)`
- 无数据时抛 `RuntimeError` 并提示检查 `EVTRADE_DB_URL`/网络，或改用 `--synthetic-days`

SQL（`TABLE = "minute_bars"`）：

```sql
SELECT stock_code, stime, open, high, low, close, volume
FROM minute_bars
WHERE stime >= '{warmup_start}000000' AND stime <= '{end_ymd}235959'
  AND stock_code = '{code}'
ORDER BY stime ASC
```

### 1.2 `synthetic_bars` —— 合成数据（无库体验/测试）

```python
synthetic_bars(days=30, start_ymd="20250101", seed=42,
               s0=10.0, jump_prob=0.02, odd_seconds_prob=0.1) -> list[Bar]
```

确定性合成 1m `Bar` 列表（seed 固定可复现）：均值回复随机游走 + 偶发跳空（制造通道偏离），
A股时段 09:30-11:29 / 13:00-14:59 每天 240 根，含周末（保证跨日/跨月边界覆盖），
少量非整分秒（练习桶边界逻辑）。CLI `--synthetic-days N` 走此路径，转成与 `load_bars`
同构的数组字典（`metrics.bars_to_arrays`）后喂给 vectorized 引擎。

### 1.3 数据库连接串（`evtrade/core/data.py::DB_URL` 默认值）

默认连接串 `mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4`，
即 `data.DB_URL` 的当前实现值（与 spec `R: Market data DB connection has a sane default` 同源）。
可用环境变量 `EVTRADE_DB_URL` 覆盖——临时切库 / 离线 / 测时 escape 入口。

- 主机 `192.168.10.2:33066`，库 `evtrade`，用户 `EvTrade`，密码 `p@ssw0rd`（URL 编码为 `p%40ssw0rd`）
- 表 `minute_bars` 字段：`stock_code, stime, open, high, low, close, volume`；表名可用 `EVTRADE_TABLE` 覆写
- `stime` 为 14 位字符串 `YYYYMMDDHHmmss`，字符串比较即时间序，SQL 的范围过滤因此有效
- 建议确认 `stime` 上有索引（按 stime 范围过滤 + 排序）
- 默认 DB 不可达时 `cli.py::_run_backtest` 会接住 SQLAlchemy `OperationalError` 并打印一行中文提示
  （含 `192.168.10.2:33066` 与 `EVTRADE_DB_URL`），再以 `SystemExit(2)` 退出

## 2. bar 流契约 —— 鸭子类型 `.stream() -> Iterator[Bar]`

Engine **不依赖任何具体源类**，只要求传入对象实现 `stream() -> Iterator[Bar]`（按时间从旧到新
yield 统一 `Bar` 结构，`evtrade/primitives.py::Bar`：`stime/code/open/high/low/close/volume`）。
实现者只需把任意数据（数组、文件、实时推送）转成 `Bar` 序列。

参考实现 `core/_harness.ListBarFeed`：

```python
class ListBarFeed:
    """Bar 列表 -> Bar 流 (passthrough)"""
    def __init__(self, bars: list): self.bars = bars
    def stream(self) -> Iterator: yield from self.bars
```

被 `replay.replay_engine`（Engine 全链路回放/对账）、tests、examples 使用。
实盘接入 = 自己写一个 `stream()` 对象，把实时行情源（推送/轮询）转成 `Bar` 逐根 yield
（需自行处理断线重连、去重、乱序丢弃）；对 Engine 完全透明，它只看到一个连续的 Bar 流。
"先历史后实时"的组合（先播含预热的历史、再接实时源）也只需在自己的 `stream()` 里
`yield from` 两段数据即可完成，framework 不再提供组合类。

## 3. 预热窗口（warmup_days → warmup_until → mark=0）

- `load_bars` 的 `warmup_days`（默认 **365**）：`warmup_start = start_ymd - 365 天`，
  实际拉取区间是 `[warmup_start, end]`
- `warmup_until` 不是源对象上的属性，而是**独立阈值**：CLI 取 `int(start_ymd) * 1_000_000`
  （即 `start_ymd + "000000"`），传给 `BarAggregator(period_seconds, on_bars, warmup_until=...)`
  与 `run_vectorized(..., warmup_until=...)`
- 聚合侧按 `stime < warmup_until` 给 bar 打 `mark=0`（预热段），`mark=1` 为策略期；
  Engine 只用预热数据喂指标累积、不驱动策略、不成交

365 天的意义：保证即使 `--period 1d`，EMA(21) 也有约一年的桶历史可完成 SMA seed。
**若把周期改得比 1d 还大，或把 tf1 调得很大，需同步调大 `--warmup-days`**。

## 4. 新数据源接入清单（详见 11 文档）

1. 在 `load_bars` 里加分支（例如 `if source == 'tushare': fetch_tushare(...)`），
   实现 `fetch_xxx()` 返回同样的 numpy 数组字典，CLI 加 `--data-source` 参数
2. 走自定义 bar 流（绕过 `load_bars`，如实时源）时：实现 `.stream()` 旧→新 yield
   `Bar(stime=..., code=..., open=..., high=..., low=..., close=..., volume=...)`
3. 保证 `stime` 是 14 位 `YYYYMMDDHHmmss` 字符串（预热 mark 与比较全依赖它）
4. 保证顺序单调不减（桶合并依赖顺序）
