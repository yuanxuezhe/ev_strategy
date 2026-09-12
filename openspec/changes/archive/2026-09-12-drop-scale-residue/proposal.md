## Why

倍投系数 `scale`（连续同方向信号数量累乘）已经在最近一次重构中**从代码移除**：
`evtrade/cli.py` 不再注册 `--scale` flag，`SimulatedExecutor` 不再维护
`scale` / `last_side` 状态机，`vectorized_engine._execute_trades` 不再做
`cur_qty *= scale`，sweep 的 `GRID_KEYS` 也不再包含 `scale`。CLI 行为与
`replay --against-ref` 对账完全不受影响。

但文字层面仍有残留：

- `kbs/06-交易策略详解.md` 保留完整 §8（35 行：历史变更 + 下线观察表 + 重新启用步骤）
- `kbs/07-账户与执行器.md` L49、`kbs/12-重构与性能内核.md` L183、`kbs/使用说明.md`
  L272 各有零星提及
- 主 `openspec/specs/evtrade-architecture/spec.md` 第 408 / 447 / 479 行仍有
  `scale` 字眼（R-Executor 调用方说明、CLI flag 表、GRID_KEYS 枚举）
- `evtrade/execution/base.py` file docstring 与 `SimulatedExecutor` docstring
  仍保留"已下线"注释

代码行为与文档不一致 = CLAUDE.md §0 单一事实源原则违例。本次把所有残留清干净，
让 **代码 / 主 spec / 主 KB 三处对齐到"从未存在"**。

## What Changes

- `openspec/specs/evtrade-architecture/spec.md`
  - `R: Single trade-execution implementation` (L408)：删除 `scale（加仓翻倍）与
    last_side 状态逻辑属于调用方状态，MAY 留在各自调用处。`
  - `R: CLI options declared once` (L447)：CLI flag 表去掉 `--scale`
  - `R: Sweep grid accepts only effective axes` (L479)：引擎轴枚举去掉 `scale /`
- `kbs/06-交易策略详解.md`：删除整段 §8（L206–240），后续 §9 顺次重编为 §8
- `kbs/07-账户与执行器.md` L49：删除 `scale` 历史段
- `kbs/12-重构与性能内核.md` L183：网格键列表去掉 `/scale`
- `kbs/使用说明.md` L272：删除 `--scale` 行
- `evtrade/execution/base.py` L11–13（file docstring）+ L71（SimulatedExecutor
  docstring）：删除 `scale` 提及与"已下线"注释

## Capabilities

### New Capabilities
（无）

### Modified Capabilities
- `evtrade-architecture`：三处 R 修订（见上）

## Impact

- **行为**：零影响（代码早已移除倍投，本次仅清文字）
- **API / CLI**：零变化（`--scale` / GRID `scale` 已不存在）
- **测试**：无新增 / 无删除（清理不动行为）
- **archive/ 下历史 change**（`2026-09-09-unify-strategy-contract` /
  `2026-09-10-strategy-step-only` / `2026-09-10-consolidate-simplify-core` /
  `2026-09-10-simplify-assembly-and-coherence`）**刻意不动**——它们是历史快照，
  `scale` 在当时确实存在过，归档里保留是事实