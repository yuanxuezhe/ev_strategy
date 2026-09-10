# evtrade-architecture Spec Delta

## ADDED Requirements

### Requirement: Account is a pure trade ledger

`Account` MUST 只做资金/持仓记账：字段限 `init_cash / init_position / cash /
position / trades`，方法限 `apply(side, qty, price, ts)`。MUST NOT 暴露权益估值
方法（`equity` / `baseline_equity` 等），MUST NOT 持有 `last_price` 类最新价状态
——权益 / 基线曲线与期末估值唯一真源是 `metrics.summarize`（输入 `final_state`
= `{cash, position, last_price, ...}`，由 `vectorized_engine` 自维护）与
`Engine` 腿的 `exec_state` 同口径字段。`Executor` 基类 MUST NOT 提供
`update_price` 钩子（无消费者）；Engine 桶 CLOSE 驱动序列为
`strategy.step → (sig != 0 时) executor.trade`，不含价格预更新步骤。

#### Scenario: 静态扫描无 equity 估值残留
- **WHEN** 静态扫描 `evtrade/`（排除 tests/）
- **THEN** `equity(` / `baseline_equity` / `last_price = price`（Account 侧赋值）/
  `update_price` MUST 均 0 命中；`Account` 实例化后无 `last_price` 属性

#### Scenario: 权益口径不受影响
- **WHEN** `python -m evtrade replay --log <log> --strategy channel_deviation
  --device cpu --against-ref`
- **THEN** 对账 [PASS]，两腿 trades 逐笔一致，`final_equity` / `baseline` /
  `excess_pct` 与删除前完全一致（真源在 `metrics.summarize`）
