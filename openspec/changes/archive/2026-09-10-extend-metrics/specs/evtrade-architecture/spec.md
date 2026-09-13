# Spec Delta: 扩展绩效指标 + 修正口径

## MODIFIED Requirements

### Requirement: metrics.summary covers full field shape

`evtrade.core.metrics.summarize` MUST 返回 25 字段（缺一即视为 broken）：

**终态字段**（5）：`final_price / final_cash / final_position / final_equity / baseline`

**交易字段**（5）：`n_trades / n_buy / n_sell / turnover / excess_pct`

**时间字段**（2）：`years / cagr_excess`（注：`ann_excess_pct` 已重命名为 `cagr_excess`，口径改为复合年化与 `cagr` 对齐）

**风险调整字段**（5）：`cagr / sharpe_excess / sortino_excess / calmar / ir`

**回撤字段**（3）：`max_drawdown / max_dd_days / max_dd_recovered`
- `max_drawdown` 单一真源：基于 `equity_curve` 由 metrics 算出，不再依赖 caller
  传入的 `final_state["max_drawdown"]`（删除 `x_mdd` 重复字段）
- `max_dd_recovered` 语义：**trough 到首个恢复 ≥ 前高**的 bar 数；未恢复时 MUST
  返回 `-1`（sentinel，区分"恢复用了 N bar"和"从未恢复"）

**持仓行为字段**（5）：`win_rate / profit_factor / avg_pnl / max_consecutive_wins / max_consecutive_losses / avg_hold_bars / max_hold_bars`（注：7 字段，但 trade-derived 与 hold-derived 并列）

**基准对比字段**（2）：`baseline_max_dd / dd_excess`（`dd_excess = max_drawdown - baseline_max_dd`，正值=策略比基准回撤更深）

`run_vectorized` MUST 累积 `equity_curve`（cash + position * close_to_now）与 `baseline_curve`（init_cash + init_position * close_to_now），末尾调 `metrics.summarize` 输出。`sweep` 评分函数 MUST 不再依赖占位 0.0（所有 Sharpe / Sortino / Calmar / CAGR / max_dd_days / IR 字段由 `metrics.summarize` 实算）。

#### Scenario: vectorized summary 25 字段齐全

- **WHEN** `run_vectorized(...)` 返回 `result["summary"]`
- **THEN** dict keys MUST 包含上述全部 25 字段；数值 MUST 非 NaN（无数据时填 0.0；profit_factor 无亏损时填 `inf`；max_dd_recovered 未恢复时填 `-1`）

#### Scenario: sweep 评分不再依赖占位

- **WHEN** `sweep.run_one_vectorized(...)` 返回 metrics dict
- **THEN** `sharpe_excess / sortino_excess / calmar / cagr / max_dd_days / ir / baseline_max_dd` MUST 由 equity 序列实算（非默认 0.0）

#### Scenario: max_dd_recovered 三档语义

- **WHEN** equity 序列存在最大回撤且**已恢复**（trough 后存在某 bar 使 equity ≥ 前高）
- **THEN** `max_dd_recovered` MUST 等于 (recovery_idx - trough_idx)，单位 bar
- **WHEN** 存在最大回撤但**从未恢复**（trough 后 equity 始终 < 前高）
- **THEN** `max_dd_recovered` MUST 等于 -1（sentinel，区别于"恢复用了 N bar"）

#### Scenario: x_mdd 字段已删除

- **WHEN** 调用 `metrics.summarize(...)` 返回 dict
- **THEN** MUST NOT 含 `x_mdd` key（与 `max_drawdown` 重复）

#### Scenario: 持仓周期从 BUY→SELL 配对计算

- **WHEN** trades 序列已知
- **THEN** `avg_hold_bars / max_hold_bars` MUST 基于相邻 BUY/SELL 配对的 (sell_ts - buy_ts) 桶数计算；未配对的开仓/平仓忽略
