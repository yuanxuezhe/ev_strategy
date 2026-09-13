# Tasks: 扩展绩效指标 + 修正口径

## 1. spec 落地
- [x] 1.1 改 `openspec/specs/evtrade-architecture/spec.md` Requirement: Metrics
      —— 字段集 16 → 25, 明确 max_drawdown 单一真源、max_dd_recovered sentinel
      语义、ann_excess_pct → cagr_excess 重命名
- [x] 1.2 `openspec validate --specs` 通过

## 2. metrics.py 实现
- [x] 2.1 删 `x_mdd` 字段 (与 `max_drawdown` 重复)
- [x] 2.2 `summarize` 加 9 字段: win_rate / profit_factor / avg_pnl /
      max_consecutive_wins / max_consecutive_losses / avg_hold_bars /
      max_hold_bars / ir / baseline_max_dd / dd_excess
- [x] 2.3 `max_dd_recovered` 算法改为 "trough → 首个恢复 >= 前高 bar 数",
      未恢复时 `-1`
- [x] 2.4 `ann_excess_pct` 字段名 → `cagr_excess`, 口径改复合年化
- [x] 2.5 `max_drawdown` 单一真源: 在 metrics 里基于 equity_curve 算,
      不再依赖 caller 的 `final_state["max_drawdown"]`

## 3. vectorized_engine.py 清理
- [x] 3.1 删 `_execute_trades` 中重复的 max_drawdown 累计
      (line 132-133, 142-146); `running_max` 仍保留供未来扩展, 但当前
      `_execute_trades` 不再算 max_dd

## 4. 测试
- [x] 4.1 新建 `tests/test_metrics_v3.py`:
      - test_win_rate / test_profit_factor / test_avg_pnl
      - test_max_consecutive_wins_losses
      - test_avg_max_hold_bars (构造已知 buy→sell 周期)
      - test_ir (excess return / tracking error 已知)
      - test_baseline_max_dd
      - test_dd_excess
      - test_max_dd_recovered_three_cases (已恢复 / 部分恢复 / 从未恢复 sentinel)
      - test_x_mdd_removed
      - test_cagr_excess_present
- [x] 4.2 `tests/test_strategy_unified.py` test_metrics_summary_has_16_fields →
      test_metrics_summary_has_25_fields (字段断言更新到 25 字段)
- [x] 4.3 `uv run pytest -q` 通过 (124 passed / 2 skipped)

## 5. KB 同步
- [x] 5.1 `kbs/13-绩效评估与鲁棒选参框架.md` §1 字段表 16 → 25 (分类呈现)
- [x] 5.2 `kbs/13-绩效评估与鲁棒选参框架.md` §1 加 max_dd_recovered sentinel 说明
- [x] 5.3 `kbs/10-配置参数与运行指南.md` §4.3 输出解读更新字段表

## 6. CLI 输出
- [x] 6.1 `evtrade/cli.py` backtest summary 打印含新字段
      (win_rate / profit_factor / ir / baseline_max_dd / dd_excess)

## 7. archive
- [x] 7.1 `openspec archive 2026-09-10-extend-metrics` 落地
