# Design: prune-dead-equity-api

## 1. 死代码链的认定

```
Account.equity() / baseline_equity()      # 零调用 (kbs/07 "预留 API")
  └─ 唯一读取 account.last_price
        └─ 唯一写入点: Executor.update_price()
              └─ 唯一调用点: Engine._process_bucket
```

删顶层即整链可删。`vectorized_engine` 的 `last_price` 是自维护局部变量
（`vectorized_engine.py:81/96/119`），与 `Account.last_price` 无关——权益曲线 /
期末估值真源一直是 `metrics.summarize(final_state)`，故行为零变化。

## 2. 为什么连 update_price 钩子一起删

- 它是 `Executor` 基类方法：实盘 Executor 子类被迫继承一个无人调用的契约。
- kbs/07 给它的唯一理由是"权益估值用，避免引擎穿透执行器内部结构"——消费者
  删除后该理由不成立；`trade(signal, price, ts)` 本身已自带价格。
- 保留 = spec 要锁一条"Engine 在 step 前必须 update_price"的无意义时序约束。

## 3. 取舍

- 不引入 Engine 直接写 account 字段的替代——`trade()` 已够用。
- `Account` 的 docstring 只留记账定位，去估值暗示。
- spec 用一条 MUST NOT 锁死"Account 纯记账"，防止下次又有人往 Account 挂估值
  方法（历史教训：预留 API 无消费者 → 文档与代码漂移）。

## 4. 验证

- `uv run pytest -q`（147 全过；无测试引用被删成员）
- `replay --against-ref` 默认腿 + all-in 腿 PASS（final_equity/baseline 不变）
- grep: `evtrade/` 内 `update_price` / `baseline_equity` / `equity(` 0 命中
