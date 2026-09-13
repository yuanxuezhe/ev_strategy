# Proposal: 扩展绩效指标 + 修正口径

## Why

`evtrade.core.metrics.summarize` 当前输出 16 字段,但缺实盘风险管理
**必备**的几类指标:

1. **胜率 / 盈亏比** — 实盘仓位大小的核心参考
2. **最大连续亏损笔数** — 决定策略能否扛过连亏期
3. **平均/最大持仓周期** — 决定资金利用率
4. **信息比率 (IR)** — 比 Sharpe 更适合"主动管理"语境
5. **基准回撤** — 买入持有的回撤, 衡量"择时贡献的回撤"

同时现有 16 字段有几处**口径问题**:

- `x_mdd` 与 `max_drawdown` 重复 (两个名字同一值)
- `max_drawdown` 在 vectorized_engine 累计一次、metrics 又算一次, 易不一致
- `max_dd_recovered = len(eq) - trough_idx` 算法错误, 实际是"距末尾多少 bar",
  不是"trough → 恢复到前高 的 bar 数"
- `ann_excess_pct` 字段名与 `cagr` 不对称 (cagr 是复合年化, ann_excess_pct 是算术年化, 易混)

## 方案

### A. 新增字段 (9 个)

```
win_rate              胜率 = 盈利笔数 / 总笔数 (空仓时=0)
profit_factor         盈亏比 = 总盈利 / |总亏损|  (无亏损时=inf)
avg_pnl               平均每笔收益 = 总盈亏 / 总笔数
max_consecutive_wins  最大连续盈利笔数
max_consecutive_losses 最大连续亏损笔数
avg_hold_bars         平均持仓周期 (桶数)
max_hold_bars         最大持仓周期 (桶数)
ir                    信息比率 = excess 年化 / tracking error 年化 (超额序列的 std 年化)
baseline_max_dd       买入持有的最大回撤 (%)
dd_excess             策略回撤 - 基准回撤 (正值=策略比基准回撤更深)
```

合计: **16 → 25 字段**。

### B. 修正口径 (3 处)

1. **删 `x_mdd`** (与 `max_drawdown` 重复)
2. **`max_drawdown` 单一真源**: vectorized_engine 只累积 `equity_curve` /
   `baseline_curve`, `max_drawdown` 全在 metrics 里从 equity_curve 算
3. **`max_dd_recovered` 算法修正**:
   - 找 trough 之后的第一个 `eq[i] >= peak_at_trough` 的 i
   - 差值 `i - trough_idx` 即"恢复所需 bar 数"
   - **未恢复时**返回 `-1` (sentinel; 区分"恢复了 100 bar"和"从未恢复")
4. **`ann_excess_pct` → `cagr_excess`**: 字段名对齐 cagr, 口径改为复合年化
   (与 cagr 一致)

### C. 测试矩阵

新增 `tests/test_metrics_v3.py`:
- 每个新字段都有 1 个针对性用例 (构造已知输入)
- `max_dd_recovered` 三档: 已恢复 / 部分恢复 / 从未恢复 (sentinel)
- 修正口径回归: 旧 `x_mdd` 已删除、字段集不含 `x_mdd`、含 `cagr_excess`

## 影响文件

修改:
- `evtrade/core/metrics.py` — 新增 9 字段 + 修正 3 处
- `evtrade/core/vectorized_engine.py` — 删重复的 max_drawdown 累计
- `evtrade/__init__.py` — `__all__` 同步 (实际 metrics 模块本身没在 __all__ 里)
- `tests/test_metrics_v2.py` (新增)
- `tests/test_vectorized.py` — 字段集更新
- `kbs/13-绩效评估与鲁棒选参框架.md` — 字段表更新
- `kbs/10-配置参数与运行指南.md` — 输出解读更新
- `openspec/specs/evtrade-architecture/spec.md` — Requirement: Metrics 更新

## 风险

- 新字段加入会改变 `summary` dict 的 keys, 任何依赖具体字段的下游 (sweep
  CSV 列、CLI 输出) 需要同步更新 — 已盘点 sweep 用 `summary["cagr"]` /
  `summary["sharpe"]` 等已存在的字段, 不依赖新字段, 影响小。
- 修正 `max_dd_recovered` 语义会改变该字段的取值, 但当前 kbs/13 没引用,
  sweep 也没用, 影响范围 = 测试本身。

## 验收

- `uv run pytest -q` 维持 111+ passed
- `python -m evtrade backtest --synthetic-days 30 --no-sleep` 输出含
  `win_rate / profit_factor / ir / baseline_max_dd` 等新字段
- `openspec validate --specs` 通过

## What Changes

### A. 新增字段 (9 个, 16 → 25)

```
win_rate              胜率 = 盈利笔数 / 总笔数 (空仓时=0)
profit_factor         盈亏比 = 总盈利 / |总亏损|  (无亏损时=inf)
avg_pnl               平均每笔收益 = 总盈亏 / 总笔数
max_consecutive_wins  最大连续盈利笔数
max_consecutive_losses 最大连续亏损笔数
avg_hold_bars         平均持仓周期 (桶数, BUY→SELL 配对)
max_hold_bars         最大持仓周期 (桶数)
ir                    信息比率 (rf=0 时 ≡ sharpe_excess; 预留 rf 引入后区分)
baseline_max_dd       买入持有的最大回撤 (%)
dd_excess             策略回撤 - 基准回撤 (正值=策略比基准回撤更深)
```

### B. 修正口径 (3 处)

- **删 `x_mdd`** (与 `max_drawdown` 重复)
- **`max_drawdown` 单一真源**: vectorized_engine 只累积 equity/baseline 序列,
  `max_drawdown` 全在 metrics 里从 equity_curve 算
- **`max_dd_recovered` 算法修正**:
  - 找 trough 之后的第一个 `eq[i] >= peak_at_trough` 的 i
  - 差值 `i - trough_idx` 即"恢复所需 bar 数"
  - **未恢复时**返回 `-1` (sentinel)
- **`ann_excess_pct` → `cagr_excess`**: 字段名对齐 cagr, 口径改为复合年化

### C. 测试

新增 `tests/test_metrics_v3.py` (13 用例) 锁定:
- 9 字段各 1 用例
- `max_dd_recovered` 三档 (已恢复 / 部分恢复 / 从未恢复)
- 修正口径回归: x_mdd 删除、cagr_excess 替换 ann_excess_pct
- 字段集恰好 25 项
