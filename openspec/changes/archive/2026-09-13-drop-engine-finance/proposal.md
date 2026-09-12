## Why

framework（`evtrade/` 顶层 + `core/` + `execution/`）当前持有与策略无关的业务概念：

- 资金 / 持仓状态（`vectorized_engine._execute_trades` 内部维护 `cash/position`；`Account` 记账）
- 撮合规则（`trade_decision`、`Executor`、`SimulatedExecutor`）
- 收益 / 绩效 / 风险指标（`core/metrics.summarize` 30 字段 KPI、`equity_curve` / `baseline_curve` / `cagr` / `sharpe` / `drawdown` / `win_rate` / `profit_factor` / `...`）
- 撮合参数（`--init-cash` / `--init-position` / `--buy-pct` / `--sell-pct` / `--all-in` / `--trade-qty`）
- 对账与回放（`core/replay.py` + CLI `replay` + `--against-ref`）
- 置换检验（`core/permutation.py` 调 metrics）

framework 的本职是"桶预计算 + `step` 驱动循环"。要绩效/资金/持仓管理就该由策略自己在 `step` 内做。

## What Changes

- **删除 framework 的执行 / 撮合 / 记账 / PnL 概念**
  - `evtrade/execution/` 整目录删除（`account.py` / `base.py` / `__init__.py`）
  - `evtrade/core/metrics.py` 删除
  - `evtrade/core/permutation.py` 删除
  - `evtrade/core/replay.py` 删除
  - `evtrade/core/config.py` 删除（`INIT_CASH / INIT_POSITION / TRADE_QTY`）
- **改 framework 核心**
  - `evtrade/core/vectorized_engine.py`：删 `_execute_trades`；`run_vectorized` 只做桶聚合 + step 循环 + 返回 `{sig, buckets, final_state}`
  - `evtrade/core/engine.py`：`on_bars` 只做 step 驱动 + 累计 sig/state（不再持 cash/position/Account）
  - `evtrade/core/sweep.py` / `batched_sweep.py`：不再调 metrics；返回 step 最终 state 关键标量
- **改 CLI**
  - 删 `--init-cash / --init-position / --buy-pct / --sell-pct / --all-in / --trade-qty / --warmup-days / --data-cache`
  - 删 11 行 PnL/收益打印（cagr / sharpe / drawdown / win_rate / ...）
  - 删 `replay` 子命令 + `--against-ref`
  - `backtest` 只剩 `--strategy / --params / --period / --code / --start / --end / --synthetic-days / --device / --verbose / --signals-out`
  - `sweep` 加 `--grid / --split / --splits / --fee-bp / --score-lambda / --min-trades / --max-mdd / --mc / --mc-top / --workers / --out / --top / --save-defaults`
- **重写三个策略**（`channel_deviation` / `ma_crossover` / `filtered_mr`）
  - state 自持 cash/position/trades/equity_history 等
  - step 内**自写撮合数学**（3 份各自，不抽取共享）
  - step 内**自算 PnL**（equity curve / cagr / sharpe / drawdown 等）
- **重写测试** + **KB 全文重审** + **CLAUDE.md §3/§5/§6 同步精简** + **主 spec R 重写**

## step 接口约束

`VectorizedStrategy.step(state, bar, params) -> (state, int)` 接口签名**不变**；仅 state 内部字段扩展。`batched_step` 接口不变（GPU 批量 sweep 由策略自负责）。

## Capabilities

### New Capabilities
- `evtrade-architecture`：5 条新 R（framework 无资金概念 / 策略自负责撮合+记账+PnL / engine 仅 step 驱动 / sweep 仅跑策略 / CLI 仅 step-driver surface）

### Modified Capabilities
- `evtrade-architecture`：重写 2 条 R（CLI options / sweep grid），保留 R `Strategy contract = step(state, bar, params)` / `PyTorch 统一后端` / `Indicators are private to strategies` / `Strategy display hooks are framework-agnostic`

### Removed Capabilities
- `evtrade-architecture`：删 5 条 R（`Single trade-execution implementation` / `Account is a pure trade ledger` / `reconcile legs receive identical effective funding` / 资金相关的 engine R / permutation R）

## Impact

- **BREAKING**：CLI `--init-cash --init-position --buy-pct --sell-pct --all-in --trade-qty` 等不再存在；CLI 报 unrecognized arguments；`replay` 子命令消失；调用方需自行迁移到 `--params` 或策略默认值。
- **BREAKING**：`evtrade/execution/` 子包、`evtrade.core.metrics`、`evtrade.core.permutation`、`evtrade.core.replay`、`evtrade.core.config` 被删；任何 `from evtrade.execution.base import trade_decision` / `from evtrade.core.metrics import summarize` 等 import 失效；旧测试大量失败需重写。
- **行为**：策略侧行为与旧 framework 行为 bitwise 一致（3 个策略的撮合数学 = `trade_decision` 旧实现；PnL = 旧 `metrics.summarize` 公式）。
- **KB**：15 份全部扫一遍；用法/CLI 与策略章节要重写。

## 关键约束

- archive/ 下历史 change 不动
- `step` / `batched_step` 接口签名不变
- 3 个策略的撮合数学 = 旧 `trade_decision` 内联实现（bitwise 一致）
- 策略 PnL 公式 = 旧 `metrics.summarize` 公式内联实现