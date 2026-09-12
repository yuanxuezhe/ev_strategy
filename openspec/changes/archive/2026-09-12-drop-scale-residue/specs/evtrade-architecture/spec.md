## MODIFIED Requirements

### Requirement: Single trade-execution implementation

成交决策（取量 + 资金/持仓约束 + 现金/持仓更新）MUST 有唯一实现
`evtrade.execution.base.trade_decision(side, price, cash, position, cur_qty, buy_pct,
sell_pct) -> (new_cash, new_position, qty, filled)`。`SimulatedExecutor.trade` 与
`run_vectorized` 的成交循环（`vectorized_engine._execute_trades`）MUST 均调用它，
MUST NOT 各自复制取量/约束公式。`replay --against-ref` MUST 仍逐笔
（ts/side/qty/price）对账两条引擎路径的成交结果。

#### Scenario: 成交公式仅一处
- **WHEN** 静态扫描 `evtrade/`（排除 tests/）中的取量表达式（`cash / price`、
  `buy_pct *`、`sell_pct *` 组合）
- **THEN** MUST 仅出现于 `execution/base.py::trade_decision` 一处

#### Scenario: 双路径逐笔一致
- **WHEN** `replay --log <log> --strategy channel_deviation --device cpu --against-ref`
- **THEN** 输出 `对账 [PASS]`，两条路径 trades 逐笔（ts/side/qty/price）一致且终态
  cash/position 一致

### Requirement: CLI options declared once

**共享选项 MUST 仅声明一次**：`backtest` / `sweep` / `replay` 三个子命令的共享选项（`--strategy --params --code
--start --end --period --trade-qty --all-in --buy-pct --sell-pct --warmup-days
--device --data-cache --verbose` 等）MUST 声明于单一共享父 parser（argparse `parents=`），
MUST NOT 各子命令重复声明同义选项。CLI 汇总打印 MUST 使用实际 `args` 值
（期初资金/持仓打印 `args.init_cash` / `args.init_position`，MUST NOT 打印模块常量
`INIT_CASH` / `INIT_POSITION`）。

#### Scenario: 共享选项单点声明
- **WHEN** 静态扫描 `evtrade/cli.py` 的 `add_argument` 调用
- **THEN** 上述共享选项名 MUST 仅出现于共享父 parser 定义处一次

#### Scenario: 汇总打印实际值
- **WHEN** `python -m evtrade backtest --init-cash 99999 ...`
- **THEN** 汇总头部"期初资金"行 MUST 打印 99999（非默认常量 100000）

### Requirement: Sweep grid accepts only effective axes

sweep 的 `--grid` 网格轴 MUST 仅接受两类键：引擎轴（`period / trade_qty /
buy_pct / sell_pct`）与策略 `params_spec` 声明的参数名。MUST NOT 接受"下游从不读取"
的轴（2026-09-10 `all_in` 曾是此类——被解析为 bool 后无任何消费者，静默空转；已删）。
资金模式网格化 MUST 经 `--grid buy_pct=...` / `sell_pct=...` 表达。`--grid all_in=...`
MUST 报 `ValueError: 不支持的网格参数 'all_in'; 可用: [...]`。backtest 子命令的
`--all-in` flag 不受影响（等价 `--buy-pct 1.0 --sell-pct 1.0`）。

#### Scenario: all_in 网格轴被拒
- **WHEN** 执行 `python -m evtrade sweep --grid all_in=true,false ...`
- **THEN** 报 `ValueError`（提示不支持的网格参数及可用列表），MUST NOT 静默跑完

#### Scenario: 资金模式经 buy_pct/sell_pct 网格化
- **WHEN** 执行 `python -m evtrade sweep --grid buy_pct=0.5,1.0 --grid sell_pct=0.5,1.0 ...`
- **THEN** 正常展开笛卡尔积并逐组回测（`all_in` 语义 = buy_pct=sell_pct=1.0 的组合）