## 1. 删除 framework 业务概念（execution / metrics / replay / permutation / config）

- [ ] 1.1 删 `evtrade/execution/` 整目录（`__init__.py` / `account.py` / `base.py`）
- [ ] 1.2 删 `evtrade/core/metrics.py`
- [ ] 1.3 删 `evtrade/core/permutation.py`
- [ ] 1.4 删 `evtrade/core/replay.py`
- [ ] 1.5 删 `evtrade/core/config.py`（`INIT_CASH / INIT_POSITION / TRADE_QTY` 下放到策略）

## 2. 改 framework 核心：vectorized_engine / engine / sweep / batched_sweep

- [ ] 2.1 `evtrade/core/vectorized_engine.py`：
  - 删 `_execute_trades` 与 `trade_decision` 调用
  - `run_vectorized` 仅做桶聚合 + step 循环，返回 `{sig, buckets, final_state}`
- [ ] 2.2 `evtrade/core/engine.py`：
  - 删 `exec_state` + `Account` + `Executor` 调用
  - `on_bars` 仅做 step 驱动 + 累计 sig/state
- [ ] 2.3 `evtrade/core/sweep.py`：删 metrics 汇总；返回 step 最终 state 关键标量
- [ ] 2.4 `evtrade/core/batched_sweep.py`：同 sweep（batched 仅产 sig）

## 3. CLI 精简

- [ ] 3.1 `evtrade/cli.py`：删 `--init-cash --init-position --buy-pct --sell-pct --all-in --trade-qty --warmup-days --data-cache`
- [ ] 3.2 删 11 行 PnL/收益打印（cagr / sharpe / drawdown / win_rate / baseline / final_equity / ...）
- [ ] 3.3 删 `replay_main` + `build_replay_parser` + `backends` 相关
- [ ] 3.4 `backtest` 输出仅打印信号轨迹 + 策略 `format_final_state` hook 输出
- [ ] 3.5 `sweep` 保留 `--grid / --split / --splits / --fee-bp / --score-lambda / --min-trades / --max-mdd / --mc / --mc-top / --workers / --out / --top / --save-defaults`（funding 由策略 params 承担）

## 4. 重写 3 个策略：自负责撮合 + 记账（PnL 可选）

- [ ] 4.1 `evtrade/strategies/channel_deviation.py`：
  - state @dataclass 加 `cash / position / trades` 等资金/持仓/账本字段
  - step 内 inline `trade_decision` 数学（旧实现等价）
  - PnL/收益/绩效字段**可选**——用户后续要在策略内自己加；本次 apply 不写 PnL
- [ ] 4.2 `evtrade/strategies/ma_crossover.py`：同上
- [ ] 4.3 `evtrade/strategies/filtered_mr.py`：同上
- [ ] 4.4 `evtrade/strategies/vectorized_base.py`：基类**不**加 `format_final_state` /
  `sweep_export_columns` 等导出 hook（策略自由）

## 5. 测试重写

- [ ] 5.1 删 `tests/test_metrics_*` / `tests/test_assembly_coherence.py`（依赖 metrics/Account）
- [ ] 5.2 删 `tests/test_account.py` / `tests/test_executor.py` / `tests/test_trade_unified.py`（依赖 execution）
- [ ] 5.3 新增 `tests/test_framework_isolated.py`：构造 strategy.step 返 sig，验证 run_vectorized 不持 cash/position/Account/Executor；framework 返回仅 `{sig, buckets, final_state}`
- [ ] 5.4 新增 `tests/test_strategy_owns_execution.py`：验证每个策略 step state 自持 cash/position/trades
- [ ] 5.5 保留 `tests/test_strategy_unified.py` / `tests/test_strategy_template.py` 等"策略契约"测试

## 6. KB / CLAUDE.md 同步

- [ ] 6.1 `kbs/01~15` 全文重审：framework 资金/撮合/PnL 章节要么改成"策略侧"要么删
- [ ] 6.2 `kbs/使用说明.md` 同步精简（删除 CLI 资金参数说明）
- [ ] 6.3 `CLAUDE.md §3 / §5 / §6` 同步精简

## 7. 验证清单（合入前必跑）

- [ ] 7.1 hygiene grep：`grep -rn "trade_decision\|Account\|Executor\|SimulatedExecutor\|cash\|position\|equity\|cagr\|sharpe\|drawdown\|win_rate\|profit_factor\|sortino\|calmar\|metrics\|max_dd\|turnover\|baseline\|format_final_state\|sweep_export_columns" evtrade/core/ evtrade/cli.py evtrade/__init__.py evtrade/strategies/vectorized_base.py` 0 命中
- [ ] 7.2 `openspec validate --specs` 通过
- [ ] 7.3 `uv run pytest -q` 全绿（重写后应 60+ passed）
- [ ] 7.4 人工烟测：
  - `python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device cpu` 应输出仅信号轨迹（无 PnL/收益/回撤字段）
  - `python -m evtrade sweep --strategy channel_deviation --grid low1=0.05,0.1 --synthetic-days 30 --out /tmp/sweep.csv --top 5` 应输出 CSV（每行 params + final_state 字段）
  - 策略 state dump：`final_state.cash / position / trades` 应已计算正确

## 8. archive

- [ ] 8.1 合入 main 后跑 `openspec archive 2026-09-13-drop-engine-finance`，确认 spec delta 已合并 + change 进入 archive/

## 风险与缓解（提请 review 时关注）

- **BREAKING CLI**：删 `--init-cash --init-position --buy-pct --sell-pct --all-in --trade-qty --warmup-days --data-cache`、删 `replay` 子命令
- **BREAKING API**：`from evtrade.execution.*` / `from evtrade.core.metrics` / `from evtrade.core.replay` / `from evtrade.core.permutation` / `from evtrade.core.config` 全部失效
- **BREAKING 测试**：196 个测试大量失败需重写（预估剩 60~80）
- **策略代码膨胀**：3 个策略各加 100~200 行（撮合数学 + PnL 公式内联）
- **batched_step GPU sweep**：cash/position 不并行；batched 仅产 sig，cash/position 由 sweep 串行层补