# 2026-09-09: unify-metrics-units — metrics 字段单位统一

## Why

`evtrade backtest` 盈亏汇总行 `最大回撤 (逐bar盯市)   : 15734173.45%` 这种数字是错的，但根因不是"打印格式错"那么简单——是 `metrics.summarize` 字段单位在 spec 里**没钉死**，导致三层同时炸：

| 层 | 文件:行 | 实际值 | 期望单位 |
|---|---|---|---|
| 字段语义 | `core/vectorized_engine.py:198` | `max_drawdown = -max_dd` 资金元 | **占初始权益小数** |
| 公式 | `core/metrics.py:131` | `calmar = cagr / max_drawdown`  cagr(%) ÷ mdd(元) | cagr(小数) ÷ mdd(小数) |
| 打印 | `cli.py:240` | `:.2%` 把"元"当小数 | 跟字段语义对齐 |
| sweep 过滤 | `core/sweep.py:351` | `mdd(元) <= max_mdd(默认 1.0 元)` 永远过 | mdd(小数) ≤ 默认 1.0(=100%=不限) |

`kbs/13` 第 39 行写" --max-mdd 默认 1.0 即不设限; 实盘建议 0.15"，是**小数语义**的明确证据，但实现里字段是元 → sweep 的 `filter_pass` **一直在默默失效**（默认 1 元几乎总过任何策略）。

回归验证脚本（`uv run python -m evtrade backtest --strategy channel_deviation --device cpu --period 5m ...`）跑出来 `Calmar +66.786 / Sharpe 66.786 / Sortino 45.358` 也是同根——跨单位 + 公式 ×100 累计误差。

## What

把 metrics 字段单位约定显式落到 spec，并修 3 处代码 + 1 处 spec + 2 处 KB：

1. **`core/vectorized_engine.py`**：`max_drawdown` 改为"占初始权益的小数"，公式 `max(0, peak-trough) / init_equity`。
2. **`core/metrics.py`**：`calmar = (cagr/100) / max_drawdown` 跨单位修齐（cagr 留 %，mdd 改小数）。
3. **`cli.py`**：`最大回撤` / `Calmar` / `最大回撤持续天数` 三行打印格式与单位显式标出；金额元字段继续用 `.2f`。
4. **新增回归 `tests/test_metrics_units.py`**：断言 `max_drawdown ∈ [0, 1.5]`、`cagr ∈ [-50, 50]`、合成曲线 `cagr=10% / mdd=20% ⇒ calmar ≈ 0.5`。

## Impact

- 受影响 capability：`evtrade-architecture`（在 `metrics.summary covers full field shape` 后新增 1 条 Requirement）
- 受影响 files：
  - 框架：`evtrade/core/vectorized_engine.py` + `evtrade/core/metrics.py` + `evtrade/cli.py`
  - spec：`openspec/specs/evtrade-architecture/spec.md`（新增 1 Requirement + 2 Scenario）
  - KB：`kbs/13-绩效评估与鲁棒选参框架.md` + `kbs/09-引擎Engine与主流程.md`
  - 测试：`tests/test_metrics_units.py`（新增）
- 受影响 public API（**语义**改，非签名改）：
  - `summary["max_drawdown"]`：资金元 → 占初始权益小数
  - `summary["calmar"]`：除法跨单位 → 除法同单位
  - 其他 14 个字段单位不变
- sweep 行为变化：`--max-mdd` 默认 1.0 现在语义=100% 回撤（实际=不限），保持向后兼容；实盘建议 `0.15` 现在能真正生效。

## Out of Scope

- 不改 equity_curve / baseline_curve 累积逻辑（仅改 max_drawdown 的归一化方式）。
- 不重写 sweep 评分公式（score / ann_net_min / S 邻域衰减）——只保证它们能拿到正确单位的字段。
- 不动指标集合（Sharpe / Sortino / Calmar 仍然存在，公式不变，仅单位换算）。
- 不改 cagr 单位（保留 %，与 KB 13 表格行 23 现状一致；只把 calmar 跨单位问题修齐）。
- 不引新策略 / 不改任何策略代码。

## Spec delta

新增 `openspec/changes/2026-09-09-unify-metrics-units/specs/evtrade-architecture/spec.md`：

- 新增 `### Requirement: metrics field units are normalized`（紧跟 `metrics.summary covers full field shape`）
- 新增 2 个 Scenario：
  - `Scenario: max_drawdown 单位为初始权益小数`
  - `Scenario: CLI 打印格式与字段单位一致`