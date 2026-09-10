# Tasks: prune-dead-equity-api

## 1. 代码

- [x] 1.1 `evtrade/execution/account.py`: 删 `equity()` / `baseline_equity()` /
      `last_price` 字段; 模块/类 docstring 去估值暗示
- [x] 1.2 `evtrade/execution/base.py`: 删 `Executor.update_price()`
- [x] 1.3 `evtrade/core/engine.py`: `_process_bucket` 删
      `self.executor.update_price(price)` 行

## 2. 文档同步 (spec 权威)

- [x] 2.1 `kbs/07-账户与执行器.md`: 字段表删 last_price 行; 方法段删 equity /
      baseline_equity 两条; §2 Executor 骨架删 update_price 行 + 其说明段;
      §5 账务口径第一条改写 (去 last_price 更新时机, 权益真源指到 metrics.summarize)
- [x] 2.2 `kbs/03-核心数据结构.md`: 账户结构段删 last_price 标量
- [x] 2.3 `kbs/02-系统架构.md` / `kbs/09-引擎Engine与主流程.md`: 伪代码删
      `executor.update_price(price)` 行
- [x] 2.4 `kbs/使用说明.md` 历史表加 2026-09-10 行 (本 change)

## 3. 验证 + archive

- [x] 3.1 `uv run pytest -q` 全过
- [x] 3.2 `replay --against-ref` 默认 + all-in 均 PASS, final_equity/baseline 不变
- [x] 3.3 grep 门禁: `evtrade/` 内 update_price / baseline_equity / equity( 0 命中
- [x] 3.4 `npx openspec validate <change> --strict` 过
- [x] 3.5 手动 merge delta 进主 spec + 对应关系表 → `openspec archive` →
      commit (message 带 change 名) → push
