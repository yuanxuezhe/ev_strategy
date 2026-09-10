# 2026-09-10: consolidate-simplify-core — 整理简化：合并重复实现、删死代码、统一 device 管线

## Why

torch 统一后端（`pytorch-unified-strategy`）落地后，探索确认代码库存在大量收敛空间：

1. **成交执行 3 份近逐字复制**：`SimulatedExecutor.trade`（execution/base.py）、
   `vectorized_engine._execute_trades`（保留 "与 SimulatedExecutor 同语义" 注释靠人工同步）、
   `Account.apply`（记账）。两份取量公式（buy_pct/sell_pct/scale/资金约束）必须永远 bitwise 一致
   才能通过 `replay --against-ref` 逐笔对账——人工同步不可靠。
2. **死代码堆积**（grep 验证零调用方）：`feeds/` 子包全量（CLI 数据加载走
   `core/data.py` 内联 SQL，与 `MySQLBacktestFeed` 重复）、`BrokerExecutor` 占位、
   `atr/rsi/boll` 指标模块（~300 行，无策略使用）、`xp_ema*`/`xp_*_torch` 变体、
   `gpu_info`、`capability.can_run`（常量返回）、`build_engine`、`Engine.print_summary`、
   `trades_to_list`、`compute_bucket`/`PERIODS`（legacy 测试锚）、`daterange`、
   CLI dead flags（`--no-sleep/--step-days/--show-bars/--bars-out`，解析后从不读）。
3. **device 死管线**：`run_vectorized(device=...)` 参数最终是 `_ = get_xp(device)`
   空操作；`core/gpu.py` 名为 gpu 实为纯 numpy 桶预计算且内嵌一份 timeutils 已有的
   Hinnant 历法；`capability.py` 仅 `select_device` 存活。
4. **CLI 重复**：backtest/sweep/replay 各自重复声明 ~15 个共用选项；`--engine` 兼容层有
   **两条独立映射路径**（argv 重写 + args 重映射，后者条件同义反复、警告可能双发）；
   `_auto_cast` 在 cli 与 _defaults_loader 各一份。
5. **行为 bug**：`channel_deviation.step()` 内遗留 `print(cur_ts, cur_high, cur_low)`
   （每个桶都打印，污染所有 verbose/对账输出）；`_defaults_loader.pick_best_row` 取
   "第一个非选中行"而非真 rank-2；`permutation.py` 读已改名的 `ann_excess_pct`
   （p 值恒对 0.0 计算）。

用户决策（2026-09-10）：
- 指标只留 EMA（step + numpy 批量参考两形态），删 atr/rsi/boll 与 xp_/torch 变体
- `feeds/` 子包 + `BrokerExecutor` 全删，数据加载留在 `core/data.py`
- `--device {cpu,gpu,auto}` CLI 参数**保留**（外部脚本兼容），删引擎层 device 死管线

## What

| 层 | 改动 |
|---|---|
| 成交执行 | 新增 `execution.base.trade_decision` 单一成交决策实现；`SimulatedExecutor.trade` 与 `vectorized_engine._execute_trades` 均改薄封装 |
| indicators | 删 `atr.py/rsi.py/boll.py`；`ema.py` 删 `xp_ema*`/`xp_*_torch` 变体，仅留 step 增量版 + numpy 批量参考版 |
| feeds | `git rm -r evtrade/feeds/`；删 `BrokerExecutor`、`daterange`、`NumpyDictFeed` |
| core/gpu.py | 改名 `core/tsbucket.py`（纯 numpy 桶预计算）；删 `gpu_info`；历法向量版并入 `timeutils.py` 与标量版共享 |
| core/capability.py | 删除；`select_device` → `backends.resolve_device`；`run_vectorized`/sweep/replay 全链 device 参数删除，CLI 入口统一 resolve（`--device gpu` 无 CUDA 改抛 ValueError） |
| CLI | 删 4 个 dead flags + `--engine` 兼容层（**接受的破坏性变更**）；共享 `_common_parent()`；`_auto_cast` 收编；修期初资金打印常量 bug |
| 其余 | 删 `PERIODS`/`compute_bucket`/`trades_to_list`/`build_engine`/`Engine.print_summary`；删 channel_deviation debug print + `Deviations.ready`；`pick_best_row` 真 rank-2；`permutation` 改 `cagr_excess`；`__init__.py` shim 收敛 7 项、删 kernel stub |
| 不合并 | `engine.py` 与 `vectorized_engine.py` **保持独立**——`replay --against-ref` 对账依赖双路径独立性，合并后对账自比失效 |

## Impact

- **specs**: `evtrade-architecture` — Indicators/Engine-assembly/CLI-surface/PyTorch-后端/Code-hygiene 5 条 MODIFIED，Single-trade-execution 与 CLI-options-declared-once 2 条 NEW
- **kbs**: 05/07/08(重写)/02/12/15/10/09/01/11/13/README/使用说明 + CLAUDE.md §3/§5
- **代码**: 预计净减 ~1200 行（含 feeds 140 + indicators 300 + capability 51 + 测试同步）
- **测试**: 147 → 预计 ~134（删 17 个死代码锚定测试，增 ~4 个新行为测试）
- **破坏性**: `--engine` flag 删除（传则 argparse unknown option）；`--device gpu` 无 CUDA 从静默回退改为抛 ValueError（三子命令统一）；`evtrade.feeds` / `evtrade.kernel` / `evtrade.config` 等旧 import 路径失效
