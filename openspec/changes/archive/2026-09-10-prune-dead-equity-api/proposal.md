# Change: prune-dead-equity-api

## Why

`Account.equity()` / `Account.baseline_equity()` 零调用（kbs/07 文档的"预留 API"，
实际权益/基线估值全部由 `metrics.summarize` 用 final_state 实算）。为它们服务的
调用链也整条死掉：`Executor.update_price()` 只被 `Engine._process_bucket` 调用，
`account.last_price` 只被这两个 equity 方法读。保留 = 每个实盘 Executor 子类被迫
实现一个没人用的钩子，且文档要维护一条不存在消费者的心智模型。

## What Changes

- 删 `Account.equity()` / `Account.baseline_equity()` / `Account.last_price`
- 删 `Executor.update_price()` 钩子
- 删 `Engine._process_bucket` 里的 `self.executor.update_price(price)` 调用
- spec delta: Account 字段/方法集收窄（MUST NOT 暴露 equity 估值方法）
- kbs 同步: 07（方法表 + §5 账务口径 + Executor 骨架）、03（last_price 行）、
  02 / 09（`_process_bucket` 伪代码去掉 update_price 行）
- 无行为变化：equity / baseline 的唯一真源一直是 `metrics.summarize`
  （`final_state.last_price` 来自 `vectorized_engine` 自维护变量，不经 Account）

## Impact

- Affected specs: evtrade-architecture（Account/Executor 接口收窄，新增 1 条
  MUST NOT 要求）
- Affected code: `evtrade/execution/account.py`、`evtrade/execution/base.py`、
  `evtrade/core/engine.py`
- 破坏性: 对外 API 收窄（`update_price` 钩子从 Executor 基类移除）；库内零调用，
  tests/ examples/ 无引用
