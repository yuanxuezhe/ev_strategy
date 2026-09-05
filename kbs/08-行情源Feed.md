# 08 行情源 Feed

> 相关源码：`Feed`（`mysql_analyze_demo.py:423-427`）、`daterange`（`mysql_analyze_demo.py:430-445`）、`MySQLBacktestFeed`（`mysql_analyze_demo.py:448-497`）、`ChainedFeed`（`mysql_analyze_demo.py:500-511`）

## 1. `Feed` 抽象

```python
class Feed:
    def stream(self) -> Iterator[Bar]: ...   # 按时间从旧到新 yield Bar
```

唯一契约：**`stream()` 从旧到新逐根 yield 统一 `Bar` 结构**。实现者只需把任意数据源（SQL、行情 API、文件、实时推送）转成 `Bar` 序列。

## 2. `daterange` —— 分段查询的区间生成器

把 `[start, end]` 按 `step_days`（默认 7）切成**闭区间**段，从旧到新 yield（`mysql_analyze_demo.py:430-445`）：

```
start=20250101, step=7 →
  段1: [20250101000000, 20250107235959]
  段2: [20250108000000, 20250115235959]
  ...
  末段: 对齐 end 的 23:59:59，不超界
```

要点：段与段无缝衔接（下一段起点 = 上一段末日 +1 天的 00:00:00）；末段截断到 end。

**为什么分段**：避免一次性把数月甚至数年的分钟数据全部载入内存；逐段流式处理，内存占用只与一段（7 天 ≈ 数千根）相关。

## 3. `MySQLBacktestFeed` —— 历史行情源（回测当前实现）

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `code` | 必传 | 证券代码（SQL 精确过滤 `stock_code = code`） |
| `start_ymd` / `end_ymd` | 必传 | 策略起止日期 YYYYMMDD |
| `step_days` | 7 | 分段查询天数 |
| `warmup_days` | **365** | 预热天数：额外拉取 start 之前 365 天数据 |
| `delay` | 0 | 每根 bar 后 sleep 秒数（回测模拟实时节奏；`--no-sleep` 时为 0） |
| `verbose` | True | 打印每段拉取进度 |

### 预热窗口

- `warmup_start = start_ymd - 365 天`（`mysql_analyze_demo.py:465-466`）
- `warmup_until = start_ymd + "000000"`（property，`mysql_analyze_demo.py:468-471`）
- SQL 实际拉取区间是 `[warmup_start, end]`；`warmup_until` 交给 BarAggregator 打 mark=0，Engine 只用预热数据喂指标、不驱动策略

365 天的意义：保证即使 `--period 1d`，EMA(21) 也有约一年的桶历史可完成 SMA seed。**若把周期改得比 1d 还大，或把 tf1 调得很大，需同步调大 warmup_days**。

### stream() 的取数与转换（`mysql_analyze_demo.py:473-497`）

1. `create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)` —— 长时间回测保活连接（断线探测 + 1 小时回收）
2. 遍历 `daterange(warmup_start, end, step_days)`，每段执行：

```sql
SELECT stock_code, stime, open, high, low, close, volume
FROM minute_bars
WHERE stime >= '{seg_start}' AND stime <= '{seg_end}'
  AND stock_code = '{code}'
ORDER BY stime ASC
```

3. `pd.read_sql(sql, conn)` 一次性 fetchall（注释：比逐行 `conn.execute` 快数倍），再 `df.itertuples` 逐行转 `Bar` yield
4. 每段结束打印 `-- 段 ... 处理 N 根, 累计 M`，-delay>0 时每根 `time.sleep(delay)`

### 数据库连接串（敏感信息，勿外传）

```
mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4
```

- 主机 `192.168.10.2:33066`，库 `evtrade`，用户 `EvTrade`，密码 `p@ssw0rd`（URL 编码为 `p%40ssw0rd`）
- 表 `minute_bars` 字段：`stock_code, stime, open, high, low, close, volume`
- `stime` 为 14 位字符串 `YYYYMMDDHHmmss`，字符串比较即时间序，SQL 的范围过滤因此有效
- 建议确认 `stime` 上有索引（分段查询按 stime 范围过滤 + 排序）

## 4. `ChainedFeed` —— 实盘组合源

```python
feed = ChainedFeed(MySQLBacktestFeed(code, 20260101, 20260903), LiveFeed(code))
```

`stream()` 依次 yield 各子源（`yield from`，`mysql_analyze_demo.py:509-511`）。实盘标准用法：

1. 先播"今天之前"的历史（含预热 mark=0，指标就绪）
2. 再接实时源（`LiveFeed` 需自行实现，把推送转成 `Bar`）
3. 对 Engine 而言完全透明，它只看到一个连续的 Bar 流

## 5. 新数据源接入清单（详见 11 文档）

1. 继承 `Feed`，实现 `stream()`：旧→新 yield `Bar(stime=..., code=..., open=..., high=..., low=..., close=..., volume=...)`
2. 保证 `stime` 是 14 位 `YYYYMMDDHHmmss` 字符串（预热与比较全依赖它）
3. 保证顺序单调不减（桶合并依赖顺序）
4. 实时源需考虑：断线重连、去重、乱序丢弃
