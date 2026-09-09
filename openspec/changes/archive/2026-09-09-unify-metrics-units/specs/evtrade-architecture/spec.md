# evtrade-architecture (delta)

> 本 spec 是 2026-09-09 `unify-metrics-units` change 的 delta；落地后由 `openspec archive` 合并到 `openspec/specs/evtrade-architecture/spec.md`。

## Modified Requirements

### Requirement: metrics.summary covers full field shape

`evtrade.core.metrics.summarize` MUST 返回以下字段（缺一即视为 broken）：`final_price / n_trades / n_buy / n_sell / final_cash / final_position / final_equity / baseline / excess / excess_pct / years / ann_excess_pct / cagr / sharpe_excess / sortino_excess / calmar / max_dd_days / max_dd_recovered / x_mdd / max_drawdown / turnover`。`run_vectorized` MUST 累积 `equity_curve`（cash + position * close_to_now）与 `baseline_curve`（期初 init_position * close_to_now + init_cash），末尾调 `metrics.summarize` 输出。`sweep` 评分函数 MUST 不再依赖占位 0.0（所有 Sharpe / Sortino / Calmar / CAGR / max_dd_days 字段由 `metrics.summarize` 实算）。

字段单位 MUST 统一为下表约定：

| 字段 | 单位 | 说明 |
|---|---|---|
| `cagr` | 百分数 (%) | `(eq[-1]/eq[0])^(1/years) - 1` 后 ×100；与 KB 13 表格行 23 一致 |
| `max_drawdown` | **占当时 peak 的小数** (0.0 ~ 1.0+) | `max(0, peak_equity - trough_equity) / peak_equity`；业界惯例 (Tradestation / PT)；与 `--max-mdd` 默认 1.0 语义对齐 |
| `sharpe_excess` / `sortino_excess` | 年化（无量纲） | `mean(excess_rets) / std(excess_rets) * sqrt(bars_per_year)` |
| `calmar` | **无量纲**（比率） | `cagr(小数) / max_drawdown(小数)` = `(cagr/100) / max_drawdown` |
| `max_dd_days` | 自然日 | `n_dd_bars / bars_per_day` |
| `max_dd_recovered` | bar 数 | 触底到末尾 bar 数 |
| `x_mdd` | **累计超额回撤小数** | `max(0, peak_cumexcess - trough_cumexcess) / init_baseline` |
| `final_equity` / `baseline` / `excess` | 金额（元） | `cash + position * last_price` |
| `final_price` | 价格（元） | 最后一根 close |
| `final_cash` / `turnover` | 金额（元） | |
| `final_position` | 股数 | |
| `excess_pct` / `ann_excess_pct` | 百分数 (%) | `(equity - baseline) / baseline * 100` |
| `years` | 年 | `(last_ts - first_ts) / (365.25 * 86400)` |

CLI 打印 MUST 与字段单位一致：`max_drawdown` / `cagr` / `excess_pct` / `ann_excess_pct` MUST 用 `:.2%`；`final_equity` / `baseline` / `excess` / `final_cash` / `turnover` MUST 用 `:.2f`；`max_dd_days` MUST 用 `:.1f` + " 天" 后缀；`calmar` MUST 用 `:.3f`（无量纲）。CLI 打印行 MUST NOT 出现"金额元"字段被 `:.2%` 格式化的输出。

#### Scenario: vectorized summary 字段齐
- **WHEN** `run_vectorized(...)` 返回 `result["summary"]`
- **THEN** dict keys MUST 包含上述字段；数值 MUST 非 NaN（无数据时填 0.0）

#### Scenario: sweep 评分不再依赖占位
- **WHEN** `sweep.run_one_vectorized(...)` 返回 metrics dict
- **THEN** `sharpe_excess / sortino_excess / calmar / cagr / max_dd_days` MUST 由 equity 序列实算（非默认 0.0）

#### Scenario: max_drawdown 单位为当时 peak 的小数
- **WHEN** `run_vectorized(...)` 返回 `result["summary"]["max_drawdown"]`
- **THEN** MUST 满足 `max_drawdown ∈ [0.0, 1.5]`（正常策略 ≤ 1.0；杠杆/极端可超 1.0 但应有界）；MUST NOT 是资金元（量级 ~1e5）
- **AND** MUST 等价于 `max(0, peak_equity - trough_equity) / peak_equity`，其中 `peak_equity = max(equity_curve[:i+1])`

#### Scenario: calmar 跨单位除法已修齐
- **WHEN** 合成曲线 `cagr=10%`、`max_drawdown=0.20`
- **THEN** `summary["calmar"]` MUST ≈ `0.10 / 0.20 = 0.5`（无量纲），MUST NOT 是 `10 / 0.20 = 50.0` 或 `10 / 200000 = 5e-5`

#### Scenario: CLI 打印格式与字段单位一致
- **WHEN** `python -m evtrade backtest ...` 打印盈亏汇总
- **THEN** `最大回撤` 行 MUST 输出合理量级的百分比（如 `-12.34%` 量级），MUST NOT 输出 `1.5e7%`
- **AND** `Calmar` 行 MUST 输出无量纲数字（如 `+0.500`），MUST NOT 输出带 `%` 后缀或与 mdd 量级挂钩

## Added Requirements

### Requirement: metrics field units are normalized

`metrics.summarize` / `vectorized_engine._execute_trades` / `cli.py` / `sweep.py` MUST 在**单位约定**上保持一致：所有下游消费者（CLI 打印、sweep 过滤、回归测试） MUST 假设上述单位表。任何"金额元"与"百分比"混用 MUST 视为 broken，由 `tests/test_metrics_units.py` 锁定。`sweep --max-mdd` 默认 1.0 MUST 含义为"100% 回撤 = 不限"；实盘建议 `0.15` 现在能真正生效（默认 `1.0` 永远过；`0.15` 拒回撤 > 15% 的策略）。

#### Scenario: sweep filter_pass 跨单位对齐
- **WHEN** `sweep.run(...)` 跑出某组参数 `max_drawdown=0.18` 且 `--max-mdd=0.15`
- **THEN** 该参数 MUST NOT 出现在 `filter_pass=True` 行（MUST 被 18% > 15% 拒掉）

#### Scenario: max_drawdown 量级回归
- **WHEN** `tests/test_metrics_units.py` 跑合成数据 `eq=[200000, 180000, 120000, 150000, 200000]` (peak 锁顶在第 1 根 = 200000, trough 在第 3 根 = 120000)
- **THEN** `max_drawdown == 0.40`（即 `(peak - trough) / peak = (200000 - 120000) / 200000`）
- **AND** `calmar == cagr / 0.40`，MUST NOT 跨单位