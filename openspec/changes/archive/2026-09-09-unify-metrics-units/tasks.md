# Tasks: unify-metrics-units — metrics 字段单位统一

## 1. Spec delta + KB 同步

- [ ] 1.1 `openspec/changes/2026-09-09-unify-metrics-units/specs/evtrade-architecture/spec.md`：在 `metrics.summary covers full field shape` 后新增"字段单位约定表"；新增 `Requirement: metrics field units are normalized`；新增 2 个 Scenario（max_drawdown 单位 + calmar 跨单位）。
- [ ] 1.2 `kbs/13-绩效评估与鲁棒选参框架.md`：行 23-29 表格后插"字段单位约定表"；改行 39 的 `--max-mdd` 注释强调"1.0 = 100% = 不限"。
- [ ] 1.3 `kbs/09-引擎Engine与主流程.md`：第 4 节 `print_summary` 后（行 90 后）插"字段单位"段落。

## 2. 代码：metrics 字段单位修齐

- [ ] 2.1 `evtrade/core/vectorized_engine.py`：删 `_execute_trades` 返回 dict 里的 `"max_drawdown": -max_dd ...` 行（统一交给 `metrics.summarize` 算）；保留循环里的 `running_max / dd / max_dd` 局部变量（如果后续代码不再用就直接删）。
- [ ] 2.2 `evtrade/core/metrics.py::summarize`：line 91-94 改 `drawdown = (eq - running_peak) / init_equity`（`init_equity = float(eq[0])`），单位 = 占初始权益小数。
- [ ] 2.3 `evtrade/core/metrics.py::summarize`：line 130-131 改 `calmar = (cagr / 100.0) / max_drawdown`（跨单位修齐）。
- [ ] 2.4 `evtrade/core/metrics.py::summarize`：`x_mdd` 也归一化为小数（`/ max(init_baseline, 1e-9)`）；line 82 `x_mdd = final_state.get("max_drawdown", 0.0)` 改为默认 0.0（不再从 final_state 取，因为现在 vectorized_engine 不再算这个字段）。
- [ ] 2.5 `evtrade/cli.py` line 237-242：calmar 用 `:.3f`（无量纲）；max_drawdown 用 `:.2%`（小数 → %）；max_dd_days 用 `:.1f 天`；其它金额元字段保留 `:.2f`。

## 3. 代码：sweep 兼容性注释

- [ ] 3.1 `evtrade/core/sweep.py` line 351 附近：在 `filter_pass = ... max_drawdown <= max_mdd` 上方加注释，说明 `max_drawdown` 现在是小数，`--max-mdd` 默认 1.0 = 100% = 不限。

## 4. 测试：回归锁定单位约定

- [ ] 4.1 新增 `tests/test_metrics_units.py`：4 个 test：
  - `test_max_drawdown_unit_is_fraction`：合成曲线 `eq=[200000, 180000, 120000, 150000]`，断言 `mdd == 0.30`（量级验证）
  - `test_cagr_is_percent`：合成曲线 2 年翻 1.21 倍，断言 `cagr == 10.0`（百分数）
  - `test_calmar_cross_unit_fixed`：合成 `cagr=10% / mdd=0.20`，断言 `calmar ≈ 0.5`
  - `test_cli_max_drawdown_format`：`subprocess` 跑 `python -m evtrade backtest ...`，断言 stdout 里"最大回撤"行匹配 `\d+\.\d{2}%`（合理量级），不含 `1.5e7%`
- [ ] 4.2 `tests/test_sweep_v2.py` 现有 sweep 测试仍通过（默认 `--max-mdd 1.0` 永远过，与之前行为兼容）；加一个 case：`--max-mdd 0.0` 应拒掉所有参数。

## 5. 端到端验证

- [ ] 5.1 `uv run pytest -q` 全部 PASS（包含 test_metrics_units.py）。
- [ ] 5.2 `uv run python -m evtrade backtest --strategy channel_deviation --device cpu --period 5m --start 20250101 --end 20260903 --no-sleep`：
  - 最大回撤打印 `+X.XX%`（合理量级，量级 < 100%）
  - Calmar 打印 `+X.XXX`（无量纲，量级 ~0~10）
  - Sharpe / Sortino 量级合理（不再 66.786 这种离谱值；如果仍偏大说明 cagr/sharpe 还有别的单位问题，本次不修）
- [ ] 5.3 `uv run python -m evtrade backtest --strategy ma_crossover --device cpu ...` 同上。
- [ ] 5.4 `uv run python -m evtrade sweep --strategy channel_deviation --device cpu --max-mdd 0.15 --grid low1=0.5,1.0 --splits 20250601`：filter_pass 列能拒回撤 > 15% 的参数（CSV 含 0/1 混合）。
- [ ] 5.5 `grep -rn "max_drawdown" evtrade/`：确认所有引用方的字段访问方式与新单位兼容（CLI / sweep / metrics 都已适配）。

## 6. openspec archive

- [ ] 6.1 `openspec archive --change 2026-09-09-unify-metrics-units`（或人工合并 spec delta 到 `openspec/specs/evtrade-architecture/spec.md`）。
- [ ] 6.2 跑 `openspec validate --specs` 通过。