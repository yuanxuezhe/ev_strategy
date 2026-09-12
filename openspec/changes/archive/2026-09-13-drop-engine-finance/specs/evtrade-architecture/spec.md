## REMOVED Requirements

### Requirement: Single trade-execution implementation

> 2026-09-13 删除。`trade_decision` 已下放到策略（3 个策略各自实现撮合数学），不再属于 framework 业务概念。

#### Scenario: 成交公式仅一处

> 2026-09-13 删除（无 trade_decision 概念，Scenario 无意义）。

#### Scenario: 双路径逐笔一致

> 2026-09-13 删除（framework 不再有"两条引擎路径"概念——只剩 `step` 驱动一条路径；vectorized 与 Engine on_bars 都只调 `strategy.step`，不再有 trade_decision 路径分歧）。

### Requirement: Account is a pure trade ledger

> 2026-09-13 删除。`Account` 是 framework 业务概念（资金/持仓记账），下放到策略 state 自管理。

#### Scenario: 静态扫描无 equity 估值残留

> 2026-09-13 删除（Account 已下放）。

#### Scenario: 权益口径不受影响

> 2026-09-13 删除（replay --against-ref 已下放，权益口径由策略自管）。

### Requirement: metrics.summary covers full field shape

> 2026-09-13 删除。30 字段 KPI 是 framework 业务概念，由策略自算（state 字段或单独导出）。

#### Scenario: vectorized summary 30 字段齐全

> 2026-09-13 删除。

#### Scenario: sweep 评分不再依赖占位

> 2026-09-13 删除（sweep 不再依赖 metrics.summarize）。

#### Scenario: max_drawdown 单位为当时 peak 的小数

> 2026-09-13 删除（无 max_drawdown 字段）。

#### Scenario: calmar 跨单位除法已修齐

> 2026-09-13 删除（无 calmar 字段）。

#### Scenario: CLI 打印格式与字段单位一致

> 2026-09-13 删除（CLI 不再打印 PnL/收益）。

#### Scenario: max_dd_recovered sentinel

> 2026-09-13 删除（无 max_dd_recovered 字段）。

#### Scenario: 持仓周期从 BUY→SELL 配对计算

> 2026-09-13 删除。

### Requirement: metrics field units are normalized

> 2026-09-13 删除。无 metrics 模块，无字段单位约定。

#### Scenario: sweep filter_pass 跨单位对齐

> 2026-09-13 删除。

#### Scenario: sweep 评分不再依赖占位

> 2026-09-13 删除。

#### Scenario: max_dd_recovered 三档语义

> 2026-09-13 删除。

#### Scenario: x_mdd 为超额曲线回撤小数

> 2026-09-13 删除。

#### Scenario: 持仓周期从 BUY→SELL 配对计算

> 2026-09-13 删除。

### Requirement: reconcile legs receive identical effective funding

> 2026-09-13 删除。`replay --against-ref` 已下线（framework 不再有逐笔对账口径）；资金/持仓/PnL 由策略自管，对账无 framework 语义。

#### Scenario: all-in 对账可 PASS

> 2026-09-13 删除。

## MODIFIED Requirements

### Requirement: CLI options declared once

**共享选项 MUST 仅声明一次**：`backtest` / `sweep` 三个子命令的共享选项（`--strategy --params --code
--start --end --period --device --synthetic-days --verbose --signals-out` 等）MUST 声明于单一共享父 parser
（argparse `parents=`），MUST NOT 各子命令重复声明同义选项。`backtest` 与 `sweep` 的输出 MUST NOT 包含 PnL /
收益 / 回撤 / 胜率 等 framework 业务概念——这些由策略自管。CLI 汇总打印仅输出信号轨迹（`format_signal_line`）
与策略 `final_state` 关键标量（由策略 `format_final_state` hook 自定义）。

#### Scenario: 共享选项单点声明

- **WHEN** 静态扫描 `evtrade/cli.py` 的 `add_argument` 调用
- **THEN** 上述共享选项名 MUST 仅出现于共享父 parser 定义处一次

#### Scenario: 汇总打印实际值

- **WHEN** `python -m evtrade backtest --strategy channel_deviation ...`
- **THEN** 输出 MUST NOT 含"期初资金/期初持仓/期末资金/期末持仓/期末持仓市值/策略总资产/不操作基线/盈亏比例/年化/CAGR/Sharpe/Calmar/最大回撤/胜率/成交额"等行；仅打印信号轨迹（verbose=True）+ 策略 `final_state` 透传（framework 不读具体字段；2026-09-13 后 framework 不再汇总 PnL/收益/回撤/胜率等业务概念）

#### Scenario: 共享选项无资金/撮合参数

- **WHEN** 静态扫描 `evtrade/cli.py` 的 `add_argument` 调用
- **THEN** `--init-cash --init-position --buy-pct --sell-pct --all-in --trade-qty --warmup-days --data-cache` MUST 0 命中（2026-09-13 删除；funding/撮合 由策略 state 自管）

### Requirement: Sweep grid accepts only effective axes

sweep 的 `--grid` 网格轴 MUST 仅接受**策略 `params_spec` 声明的参数名**（引擎无业务参数轴，funding/撮合 由策略
params 承担）。MUST NOT 接受"下游从不读取"的轴。`--grid period=...` / `--grid code=...` /
`--grid buy_pct=...` 等引擎轴 MUST 报 `ValueError: 不支持的网格参数 '<key>'; 可用: [...]`。
backtest 子命令的 `--period` / `--code` flag 不受影响（CLI 入口，不入 grid）。

#### Scenario: 引擎轴不可网格化

- **WHEN** 执行 `python -m evtrade sweep --grid period=5m,15m --grid buy_pct=0.5,1.0 --grid trade_qty=10000 ...`
- **THEN** MUST 报 `ValueError`，提示不支持的网格参数及可用列表（来自策略 `params_spec`），MUST NOT 静默跑完

#### Scenario: 策略参数轴生效

- **WHEN** 执行 `python -m evtrade sweep --grid low1=0.05,0.1 --grid low2=0.5,1.0 ...`（命中 channel_deviation params_spec）
- **THEN** 正常展开笛卡尔积并逐组回测

#### Scenario: all_in 网格轴被拒

- **WHEN** 执行 `python -m evtrade sweep --grid all_in=true,false ...`
- **THEN** 报 `ValueError`（提示不支持的网格参数及可用列表），MUST NOT 静默跑完

#### Scenario: 资金模式经 buy_pct/sell_pct 网格化

- **WHEN** 执行 `python -m evtrade sweep --grid buy_pct=0.5,1.0 --grid sell_pct=0.5,1.0 ...`
- **THEN** 正常展开笛卡尔积并逐组回测（`all_in` 语义 = buy_pct=sell_pct=1.0 的组合）

## ADDED Requirements

### Requirement: Framework has no funding or PnL concept

framework 的所有接口 MUST NOT 包含资金 / 持仓 / 撮合 / PnL / 收益 / 风险 / 绩效 等业务概念：

- `VectorizedStrategy.step(state, bar, params) -> (state, int)` 接口签名不变；`bar` dict 仅含
  `{ts, o, h, l, c, v, mark}`；`buckets` dict 仅含 `{ts, o, h, l, c, v, mark, n_bars}`。
- `run_vectorized(...)` 返回 = `{sig, buckets, final_state}`（`final_state` = 策略 step 末尾
  state 透传，**MUST NOT** 含 framework 业务字段键如 `cash / position / equity / final_cash /
  final_position / final_equity / n_trades / n_buy / n_sell / turnover / cagr / sharpe / drawdown /
  win_rate / profit_factor / 等` —— 这些键由策略自己维护在 dataclass state field，framework 不读不写）。
- framework MUST NOT 提供 `format_final_state` / `sweep_export_columns` 等"摘要导出 hook"；
  策略若要看 PnL / 收益 / 绩效，自己加 state 字段 + 自己写打印/导出代码。
- `evtrade/execution/` 子包 MUST NOT 存在；`Account` / `Executor` / `SimulatedExecutor` /
  `trade_decision` / `apply(side, qty, price, ts)` MUST NOT 出现在 framework 代码。
- `evtrade/core/metrics.py` MUST NOT 存在；`summarize` / `metrics.*` MUST NOT 出现在 framework 代码。
- `evtrade/core/permutation.py` MUST NOT 存在（依赖 metrics）。
- `evtrade/core/replay.py` MUST NOT 存在（依赖 Account + metrics）。
- `evtrade/core/config.py` MUST NOT 含 `INIT_CASH / INIT_POSITION / TRADE_QTY` 等常量
  （CLI 不再传；策略 init_state 自定）。

#### Scenario: framework grep hygiene

- **WHEN** 执行 `grep -rn "trade_decision\|Account\|Executor\|SimulatedExecutor\|cash\|position\|equity\|cagr\|sharpe\|drawdown\|win_rate\|profit_factor\|sortino\|calmar\|metrics\|max_dd\|turnover\|baseline\|format_final_state\|sweep_export_columns" evtrade/core/ evtrade/cli.py evtrade/__init__.py evtrade/strategies/vectorized_base.py`
- **THEN** MUST 0 命中（仅桶聚合、step 驱动、numpy/tsbucket 路由相关代码；funding/PnL/绩效 概念全部下放到具体策略文件内）

#### Scenario: execution 子包已删除

- **WHEN** 静态检查仓库
- **THEN** `evtrade/execution/` 目录 MUST 不存在；`from evtrade.execution.* import` MUST ImportError

#### Scenario: run_vectorized 返回无业务字段

- **WHEN** `run_vectorized(...)` 返回 dict
- **THEN** MUST 仅含 `sig / buckets / final_state` 三个 key；`final_state` 是策略 step 末尾
  state（dataclass 或 dict），framework 仅透传其内容，不识别具体字段

### Requirement: Strategy owns trade execution and PnL accounting

策略 MUST 自负责以下业务概念（在 `step(state, bar, params)` 内部 / state 字段内 / `init_state` 内）：

- **资金 / 持仓状态**：state 字段（推荐 `@dataclass`，含 `cash / position / init_cash / init_position`
  等），或外部 dict
- **撮合数学**（buy/sell 决策 + 现金/持仓约束 + 成交价 = `bar["c"]`）：策略 `step` 内部 inline 实现
  （`trade_decision` 旧实现内联）；3 个策略各写一份，**不抽取共享**（避免 framework 隐藏业务概念）
- **记账**（trade ledger）：state 字段 `trades: list[dict]`
- **PnL 与风险**（equity curve / cagr / sharpe / drawdown / win_rate / 等）：**可选**——
  策略需要时自行加 state 字段与计算（旧 `metrics.summarize` 30 字段公式内联到策略或策略的
  helper 文件中）；framework **不强制**、**不提供**、**不识别**

#### Scenario: 策略 step state 含 cash/position/trades 字段

- **WHEN** 静态扫描 `evtrade/strategies/*.py`（排除 `vectorized_base.py`）
- **THEN** 至少一个策略的 `@dataclass` state MUST 含 `cash / position / trades` 字段（策略自管资金/持仓/账本）

#### Scenario: 策略 step 内自写撮合数学

- **WHEN** 静态扫描 `evtrade/strategies/*.py`
- **THEN** MUST 出现 `cash / price`、`buy_pct *`、`sell_pct *`、`min(cur_qty, max_by_cash)` 等
  撮合数学表达式（内联在 step 内或策略内部 helper，**不**走 framework `from evtrade.execution.* import`）

#### Scenario: 策略 PnL 字段可选

- **WHEN** 静态扫描 `evtrade/strategies/*.py`
- **THEN** PnL / 收益 / 绩效相关字段（`equity_curve / cagr / sharpe / drawdown / win_rate /
  profit_factor / ...`）**MUST NOT** 出现在 framework 代码（`evtrade/core/` / `evtrade/cli.py` /
  `evtrade/strategies/vectorized_base.py`）；允许出现在具体策略文件内（由策略自维护）

### Requirement: Engine drives step only

framework 唯一职责 = 桶预计算 + `step` 驱动循环 + 累计 sig/state。

- `VectorizedEngine.run_vectorized` MUST 仅做：
  1. 桶聚合（numpy 向量化；调用 `core/tsbucket.py::precompute_ts_mark`）
  2. 循环调 `strategy.step(state, bar, params)`（`_compute_signals`），state 由 framework 持有
  3. 返回 `{sig, buckets, final_state}`（`final_state` = 策略 step 最终 state 透传）
- `Engine.on_bars` MUST 仅做逐 bar 循环调 `strategy.step` + 累计 sig/state；不持 cash/position/
  Account/Executor；不调 metrics
- framework MUST NOT 调 `trade_decision` / `Account.apply` / `Executor.trade` 等业务接口
  （这些在策略侧 step 内完成）

#### Scenario: vectorized_engine 不含业务概念

- **WHEN** 静态扫描 `evtrade/core/vectorized_engine.py`
- **THEN** MUST NOT 出现 `trade_decision` / `Account` / `cash` / `position` / `equity` /
  `turnover` / `metrics` / `summarize` 等业务概念符号

#### Scenario: engine.py 不含业务概念

- **WHEN** 静态扫描 `evtrade/core/engine.py`
- **THEN** MUST NOT 出现 `Account` / `Executor` / `SimulatedExecutor` / `trade_decision` /
  `cash` / `position` / `equity` / `apply(` 等业务概念符号

### Requirement: Sweep runs strategy over a parameter grid

`sweep --grid` MUST 仅扫描**策略 `params_spec` 声明的参数名**；引擎无业务参数轴（funding/撮合 由策略
params 承担）。每组参数组合 MUST 调一次 `run_vectorized` 跑完桶，返回 `final_state`；CSV 写策略
`final_state`（由 sweep 序列化 dataclass/dict 字段）。

#### Scenario: sweep 跑策略参数组合

- **WHEN** 执行 `python -m evtrade sweep --strategy channel_deviation --grid low1=0.05,0.1 --grid low2=0.5,1.0 ...`
- **THEN** MUST 展开 4 组笛卡尔积，每组跑一次 `run_vectorized`，CSV 每行 = 策略 params + final_state 字段

### Requirement: CLI is step-driver surface

CLI = `python -m evtrade {backtest, sweep, params}` 三个子命令。

- `backtest`：仅 `--strategy / --params / --period / --code / --start / --end / --synthetic-days /
  --device / --verbose / --signals-out`。**删除** `--init-cash / --init-position / --buy-pct /
  --sell-pct / --all-in / --trade-qty / --warmup-days / --data-cache`（2026-09-13）
- `sweep`：保留 `--grid / --split / --splits / --fee-bp / --score-lambda / --min-trades / --max-mdd /
  --mc / --mc-top / --workers / --out / --top / --save-defaults`（funding 相关参数由策略 params 承担）
- `replay` 子命令**删除**（framework 不再有逐笔对账口径）
- CLI 汇总打印 MUST NOT 包含 PnL / 收益 / 回撤 / 胜率 / 资金 / 持仓 / 期末市值 / 成交额 / 等 framework
  业务概念；`backtest` 输出仅打印信号轨迹（`format_signal_line`）。策略若要看自己 state，由策略内部
  提供打印 hook（由策略代码决定，非 framework 职责）

#### Scenario: 旧资金/撮合 flag 已删除

- **WHEN** 执行 `python -m evtrade backtest --init-cash 100000 ...` 或 `--buy-pct 1.0 ...` 等
- **THEN** argparse 报 `unrecognized arguments: --init-cash` / `--buy-pct` 退出（2026-09-13 删除）

#### Scenario: replay 子命令已删除

- **WHEN** 执行 `python -m evtrade replay ...`
- **THEN** argparse 报 `error: argument {backtest,sweep,params}: invalid choice: 'replay'` 退出

#### Scenario: backtest 仅打印信号轨迹

- **WHEN** `python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device cpu`
- **THEN** 输出 MUST NOT 含 `盈亏比例 / CAGR / Sharpe / Calmar / 最大回撤 / 胜率 / 期末资金 / 期末持仓 /
  期末持仓市值 / 策略总资产 / 不操作基线 / 成交额` 等行；仅打印信号轨迹（`format_signal_line`）

### Requirement: replay & permutation removed

`evtrade/core/replay.py` 与 `evtrade/core/permutation.py` MUST NOT 存在（2026-09-13 删除）。framework
不再提供逐笔对账与置换检验口径。CLI 无 `replay` 子命令；`--against-ref` 不存在；`--mc` flag
MUST NOT 触发置换检验（保留参数解析但调 `NotImplementedError`，明示 framework 不再支持）。

#### Scenario: replay / permutation 模块已删除

- **WHEN** 静态检查仓库
- **THEN** `evtrade/core/replay.py` / `evtrade/core/permutation.py` MUST 不存在；`from
  evtrade.core.replay import *` / `from evtrade.core.permutation import *` MUST ImportError