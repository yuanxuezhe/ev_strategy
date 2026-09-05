# 05 指标计算：EMA 与通道轨

> 相关源码：`ema`（`mysql_analyze_demo.py:151-162`）、`IncrementalEMA`（`mysql_analyze_demo.py:165-205`）、`EMAChannel`（`mysql_analyze_demo.py:208-229`）、`ema_channel`（`mysql_analyze_demo.py:232-244`）

## 1. 指标定义：通达信蓝色通道轨

```
UP1 := EMA(H, TF1)    上轨（对周期桶最高价做 EMA）
DW1 := EMA(L, TF1)    下轨（对周期桶最低价做 EMA）
TF1 = 21（默认，--tf1 可调）
```

即上轨是"最高价的 21 周期 EMA"、下轨是"最低价的 21 周期 EMA"，构成一条随价格平滑移动的通道。

## 2. EMA 数学定义（本实现的口径）

- 平滑系数：`k = 2/(p+1)`（p=21 时 k≈0.0909）
- **种子（seed）**：前 p 个值做简单平均 `SMA = sum(前p个)/p`，作为第一个 EMA
- 之后递推：`EMA_t = price_t * k + EMA_{t-1} * (1-k)`

代码同时保留两个实现，口径一致、互为校验：

| 实现 | 用途 | 复杂度 |
|---|---|---|
| `ema(values, p)` 纯函数 | 一次性批量计算 / 单测校验 | O(n) |
| `IncrementalEMA` 状态机 | 热路径（逐桶推送） | O(1)/桶 |

## 3. `IncrementalEMA` —— 增量状态机

内部状态三个：`sum`（前 p 个值的累加）、`count`（已闭合桶数）、`ema`（闭合序列最后一个 EMA，count≥p 时有值）。

### `push(value)` —— 桶闭合时调用，O(1)

```
count < p  :  sum += value; count++
             若 count == p: ema = sum/p      ← SMA seed，首个 EMA 诞生
count >= p :  ema = value*k + ema*(1-k); count++
```

### `current(pending_value)` —— 带当前未闭合桶的"临时 EMA"，O(1) 且不改状态

用于在当前桶尚未闭合时，把当前桶最新 H/L 当作第 count+1 个值递推一次：

```
count <  p-1 :  返回 None（数据不足）
count == p-1 :  返回 (sum + pending) / p        ← pending 凑满 p 个，得 SMA seed
count >= p   :  返回 pending*k + ema*(1-k)      ← 正常递推一次
```

关键性质：`current()` 是**试探性计算**，不修改任何状态；同一桶内每根 1m bar 调一次，值随最新 H/L 变化；桶真正闭合后由 `push` 固化。这样"盘中信号"与"收盘固化"用的是同一套递推公式，不会出现口径漂移。

### 与纯函数版的一致性

`ema_channel(bars, tf1)`（纯函数）要求 `len(bars) >= tf1`，即 tf1-1 个闭合桶 + 当前桶；
`IncrementalEMA.current` 在 `count == p-1` 时用 pending 凑满 p 个——两者对"数据何时够"的判定完全一致。

## 4. `EMAChannel` —— 上/下轨的封装

```python
ch = EMAChannel(tf1=21)
ch.push(high, low)                 # 每个桶闭合时调用（内部 push 两条 IncrementalEMA）
up, dw = ch.channel(cur_h, cur_l)  # 每根 1m bar 调用，得当前 (UP, DW)，含未闭合桶
```

- `push` 接收**闭合桶**的 H/L；`channel` 接收**当前桶**的最新 H/L
- 数据不足时返回 `(None, None)`，策略层据此直接跳过（`mysql_analyze_demo.py:293-294`）
- Engine 用 `_pushed` 计数保证每个闭合桶只被 push 一次（见 09 文档 `_sync_ema`）

## 5. 数据就绪时间线（tf1=21 为例）

| 已闭合桶数 | push 后状态 | current(pending) 返回 |
|---|---|---|
| 0 ~ 19 | 累加中，无 EMA | None（count<20 时） |
| 20（count=20=p-1） | 累加中 | `(sum+pending)/21`，首个试探 EMA |
| 21（count=21=p） | `ema = sum/21`（SMA seed） | `pending*k + ema*(1-k)` |
| 之后每桶 | 正常递推 | 同上 |

因此：**策略最早能在"第 21 个周期桶"看到通道值**。回测默认预热 365 天，即使 1d 周期也有充足历史；若自定义更长的 tf1 或更大周期，需确认 `--warmup`（Feed 的 warmup_days）足够。

## 6. 复杂度对比（为什么做增量）

朴素做法：每次回调对全部闭合桶重算 EMA → O(n)/根，n 为桶数，全程 O(n²)。
增量做法：`push` O(1)/桶 + `current` O(1)/根 → 全程 O(总根数)。
单文件回测（百万级 1m bar）与实盘逐根推送都因此可行。
