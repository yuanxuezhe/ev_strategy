# 参数工作流：网格扫描 → 选参 → 落盘 → 单次回测/实盘

四步闭环, 每步一个 CLI 子命令, 最优参数落盘到仓库内
`evtrade/strategies/_defaults/<strategy>.json` (默认跟踪, 可分享),
实盘/单次回测不传参数即读默认。

---

## 1. 网格扫描 (sweep)

```bash
python -m evtrade sweep \
  --strategy channel_deviation \
  --code 159992.SZ \
  --start 20250101 --end 20260903 \
  --grid low1=1.0,1.5,2.0,2.5 \
  --grid low2=0.5,1.0,1.5 \
  --grid high1=1.0,1.5,2.0,2.5 \
  --grid high2=0.3,0.5,0.8 \
  --device auto \
  --fee-bp 5 \
  --min-trades 30 \
  --max-mdd 0.4 \
  --score-lambda 1.0 \
  --splits 20250901 20260301 \
  --workers 8 \
  --out sweep_results.csv \
  --top 20
```

要点:

- `--device auto`: GPU (torch CUDA) 可用则走 GPU, 否则 CPU 降级 + warn。
- `--grid key=v1,v2,...` 可多次, 笛卡尔积展开。
- `--splits ymd1 ymd2 ...`: 滚动 WFO, 列名带 `train_` / `test1_` / `test2_` 前缀。
- `--fee-bp / --min-trades / --max-mdd / --score-lambda`: 评分
  `score = ann_net_min / (1 + λ·S)` 的过滤与权重。
- `--out`: 结果 CSV 路径; `--top`: 控制台预览前 N 组。

---

## 2. 挑选最优参数

```bash
# 全表查看 (列太多时分页)
column -t -s, sweep_results.csv | less -S

# 按 test 段 score 排序取前 10
python -c "
import pandas as pd
df = pd.read_csv('sweep_results.csv')
print(df.sort_values('test1_score', ascending=False).head(10).to_string(index=False))
"
```

挑选原则:

- 优先看 `test1_score` / `test2_score` (实盘段), 不只看 `train_score`。
- `test1_mdd` 不超 `--max-mdd` 阈值。
- `n_trades` 不低于 `--min-trades`。
- 若 `test1_score` 与 `train_score` 差距过大 (过拟合), 选更保守的那一组。

记下四个数 (low1 / low2 / high1 / high2) 准备落盘。

---

## 3. 落盘默认参数

两种方式:

### 3a. 直接从 `--params` 字符串存

```bash
python -m evtrade params save channel_deviation \
  --params "low1:1.5;low2:1.0;high1:1.5;high2:0.5"
```

### 3b. 从 sweep 结果 CSV 取第 N 行 (1=score 最高)

```bash
python -m evtrade params save channel_deviation \
  --from-csv sweep_results.csv --rank 1
```

落盘文件: `evtrade/strategies/_defaults/channel_deviation.json`:

```json
{
  "strategy": "channel_deviation",
  "params": {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5},
  "saved_at": "2026-09-07T00:00:00+00:00",
  "source": {"kind": "from_csv", "csv": "sweep_results.csv", "rank": 1}
}
```

查看 / 列出:

```bash
python -m evtrade params show channel_deviation
python -m evtrade params list
```

---

## 4. 单次回测 (不传参数, 自动读默认)

```bash
# 不传 --params; 自动从 _defaults/channel_deviation.json 读
python -m evtrade backtest \
  --strategy channel_deviation \
  --code 159992.SZ \
  --start 20250101 --end 20260903 \
  --device auto

# 显式覆盖某个参数 (其余仍读默认)
python -m evtrade backtest \
  --strategy channel_deviation \
  --code 159992.SZ \
  --start 20260101 --end 20260903 \
  --params "low1:2.0"

# 显式全参数 (最高优先级, 完全覆盖默认)
python -m evtrade backtest \
  --strategy channel_deviation \
  --code 159992.SZ \
  --start 20260101 --end 20260903 \
  --params "low1:2.0;low2:1.5;high1:2.0;high2:1.0"
```

> 旧 `--engine kernel/ref/vectorized` 已删除（检测到时打 DeprecationWarning 后自动映射到
> `--device`）。后端统一为 torch，CPU/GPU 由 `--device {cpu,gpu,auto}` 选择。

---

## 5. 实盘接入

实盘脚本骨架 (示例, 等 live 子命令实现后填具体 broker):

```python
# scripts/run_live.py
from evtrade.strategies._defaults_loader import load

# 1. 加载默认参数 (不传参数就是实盘默认)
params = load("channel_deviation")
print(f"实盘参数: {params}")

# 2. 实例化策略
from evtrade.strategies import get_strategy
strategy = get_strategy("channel_deviation", params=params)

# 3. 喂给 Engine / BrokerExecutor (TODO: live 子命令未实现)
# engine = Engine(feed, aggregator, strategy, executor, tf1=21)
# engine.run()
```

或 CLI 方式 (live 子命令实现后):

```bash
# 实盘脚本不传 --params, 自动读默认
python -m evtrade live \
  --strategy channel_deviation \
  --code 159992.SZ \
  --broker sim   # 或真实 broker
```

实盘机特定参数覆盖 (不污染仓库):

```bash
# 在实盘机上设环境变量, 落盘路径切到 /etc/evtrade/defaults/
export EVTRADE_DEFAULTS_DIR=/etc/evtrade/defaults
python -m evtrade params save channel_deviation \
  --params "low1:1.8;low2:1.2;high1:1.8;high2:0.6"
```

---

## 参数解析优先级 (CLI)

每条命令解析策略参数时按以下顺序 (高 → 低):

1. **`--params "k:v;..."`** 显式传入 (任何策略, 类型自动推导 int/float/bool/str)
2. **`evtrade/strategies/_defaults/<strategy>.json`** 落盘默认 (任何策略)
3. **空 dict** (后续 `VectorizedStrategy` 用 `params_spec` 的 default 值)

实盘/CI 机器用 `EVTRADE_DEFAULTS_DIR` 环境变量切换落盘目录, 不污染仓库。

---

## 完整示例 (一次跑通四步)

```bash
# 1. 扫描
python -m evtrade sweep \
  --strategy channel_deviation --code 159992.SZ \
  --start 20250101 --end 20260903 \
  --grid low1=1.0,1.5,2.0 --grid low2=0.5,1.0,1.5 \
  --grid high1=1.0,1.5,2.0 --grid high2=0.3,0.5,0.8 \
  --splits 20250901 --device auto --out sweep_results.csv

# 2. (人工查看 sweep_results.csv, 假定 test1_score 最高的组合是 low1=1.5,low2=1.0,high1=1.5,high2=0.5)

# 3. 落盘
python -m evtrade params save channel_deviation \
  --params "low1:1.5;low2:1.0;high1:1.5;high2:0.5"

# 4. 单次回测 (不传参数)
python -m evtrade backtest \
  --strategy channel_deviation --code 159992.SZ \
  --start 20260101 --end 20260903 --device auto
```

第四步会打印 "策略: channel_deviation  参数: {low1=1.5, low2=1.0, ...}",
证明参数来自 `_defaults/channel_deviation.json`。
