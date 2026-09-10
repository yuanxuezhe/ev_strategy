# Design: consolidate-simplify-core — 合并重复实现、删死代码、统一 device 管线

## 1. 核心决策

### 1.1 不合并两个引擎

`core/engine.py::Engine`（逐 bar，aggregator 驱动）与 `core/vectorized_engine.py::run_vectorized`
（numpy reduceat 桶级）保持独立。`replay --against-ref` 的价值 = 两条**独立**实现逐笔对账；
合并后对账退化为自比。收敛点改在**共享原子**（trade_decision）而非合并引擎。

### 1.2 trade_decision 设计

位置：`evtrade/execution/base.py` 模块级函数（execution 是成交语义的家；不新建文件）。

```python
def trade_decision(side, price, cash, position, cur_qty, buy_pct, sell_pct)
    -> tuple[new_cash, new_position, qty, filled]:
```

- side ∈ {1, -1}（调用方保证非 0）；返回 filled=False 时现金/持仓原样返回
- BUY：`max_by_cash = cash/price`；`buy_pct>0` 时 `q = min(buy_pct*max_by_cash, max_by_cash)`，
  否则 `q = min(cur_qty, max_by_cash)`；`q<=0 → unfilled`；否则
  `cash - q*price, position + q`
- SELL：`sell_pct>0` 时 `q = min(sell_pct*position, position)`，否则 `q = min(cur_qty, position)`
- **bitwise 等价性**：现有两份实现的 `x if x < y else y` ≡ `min(x,y)`（浮点全序，无 NaN 输入）；
  现金更新顺序 `cash - q*price` 保持不变 → reconcile 逐笔一致不受影响
- scale（`cur_qty *= scale` / last_side 切换重置）留在调用方：Executor 是 instance attr，
  vectorized 是循环局部变量——是调用方状态，不是成交决策
- `SimulatedExecutor.trade` 剩余职责：cur_qty/last_side 更新 + 调 trade_decision +
  `acc.apply` + record + verbose print
- `_execute_trades` 剩余职责：cur_qty 循环变量 + 调 trade_decision + trades.append
  （`cash_after` 字段由调用方用返回的 new_cash 填充）
- 无循环依赖：execution/base.py 只 import .account

### 1.3 tsbucket（原 core/gpu.py）

`git mv core/gpu.py core/tsbucket.py`。内容：`precompute_ts_mark` + LRU 缓存
（键/容量/失效 API 名不变，仅 import 路径变）；删 `gpu_info`（零调用方）。
历法去重：`_encoded_to_epoch_np` / `_epoch_to_encoded_np`（Hinnant 向量版）移入
`timeutils.py`，与标量版共享 `_days_from_civil` / `_civil_from_days`
（纯整数运算，int64 数组 numpy 广播逐元素等价；用 3000 次 roundtrip 测试扩展到数组验证）。
tsbucket 只保留桶算术 + mark 生成 + 缓存。

### 1.4 capability 收编

删 `core/capability.py`。存活函数 `select_device` → `backends.resolve_device(requested,
gpu_available=None)`（去掉无用的 `strategy_name` 参数，语义逐字复制：
cpu→cpu；gpu→无 CUDA 抛 ValueError；auto→gpu 可用则 gpu 否则 cpu + warning）。
`gpu_available` 唯一实现在 `backends`（capability 的委托副本消失）。
`can_run`（常量 `(True, "")`）与 `TARGET_CAPS`（仅测试引用）直接删除。
sweep.py 能力探测块改调 `resolve_device`。
**CLI 入口统一 resolve**：`_run_backtest` / `replay_main` / `_run_sweep` 入口
`args.device = resolve_device(args.device)`——行为变化：backtest/replay 的
`--device gpu` 无 CUDA 从静默回退（旧 get_xp RuntimeWarning 路径）改为抛 ValueError，
与 sweep 一致（spec 化的行为收敛，非回归）。

### 1.5 device 参数链删除

`run_vectorized` 删 `device` 参数（现在只是 `_ = get_xp(device)` 空操作）；
`run_one_vectorized` / `run_one_from_dict` / `replay_vectorized` 同步删除；
全部 `device=` 调用点（cli/replay/sweep/tests）清除。replay banner 打印 resolved device。

### 1.6 CLI 结构

- 删 dead flags：`--no-sleep --step-days --show-bars --bars-out`（解析后从不读）
- 删 `--engine` 兼容层全部 4 处（`_LEGACY_ENGINE_MAP`、`_emit_legacy_engine_warning`、
  `_coalesce_legacy_engine` argv 重写、隐藏 `--engine` arg + `_run_backtest` 内重映射
  ——后者条件 `args.device == "auto" or args.device != mapped` 是同义反复）。
  **硬删而非 deprecate**：flag 对应的映射目标已一年，仓库/文档内零使用；保留 = 保留
  第二条代码路径，与"能去掉去掉"目标相悖。破坏性变更记入 proposal。
- 共享 `_common_parent()`（argparse `add_help=False`）声明 ~15 个共用选项，
  三个子命令 `parents=[...]`；backtest 的富 help 文案进 parent。
  `build_root_parser` 的 `sub_p._add_action(action)` 拷贝循环随 parent 化一并重写
  （拷贝循环与 parents 机制不兼容，此步必需非可选）
- `_auto_cast` 收编：`_defaults_loader._auto_cast` → 公开 `auto_cast`，删 cli 副本
- cli.py:223-224 期初资金/持仓打印 `args.init_cash / args.init_position`

### 1.7 __init__.py shim 最小集

保留 7 项（仓库内仍有 import 方）：`evtrade.data / metrics / account / aggregator /
replay / execution / primitives`。删 kernel ModuleType stub（前提：test_timeutils.py 的
`from evtrade.kernel import ...` 迁移到 `evtrade.core.*`）。`__all__` 同步收敛。

## 2. 各包死代码清单（grep 验证零调用方后删）

| 符号 | 位置 | 处理 |
|---|---|---|
| `gpu_info` | core/gpu.py:18 | 删（随改名） |
| `xp_ema*` / `xp_*_torch` | indicators/ema.py | 删 + 删 torch import |
| atr/rsi/boll 全模块 | indicators/ | git rm |
| `build_engine` / `Engine.print_summary` | core/engine.py | 删 |
| `can_run` / `TARGET_CAPS` | core/capability.py | 删（模块整体删） |
| `compute_bucket` / `_bucket_plus_period` | core/timeutils.py | 删 + 删 legacy 等价测试 |
| `daterange` | core/timeutils.py | 删（唯一调用方是已删的 MySQLBacktestFeed） |
| `PERIODS` | core/config.py | 删 + 删测试锚 |
| `INTERVAL` | core/config.py | 删（cli import 后未用） |
| `trades_to_list` | core/metrics.py | 删 |
| `BrokerExecutor` | execution/base.py | 删 |
| `NumpyDictFeed` | core/_harness.py | 删（留 ListBarFeed：replay + demo 在用） |
| feeds/ 全子包 | evtrade/feeds/ | git rm |
| `--no-sleep` 等 4 flag + `--engine` 层 | cli.py | 删 |
| `_aggregate_bars_gpu` | tests/test_strategy_unified.py | 删（与 _aggregate_bars 重复） |
| `Deviations.ready` | strategies/channel_deviation.py | 删 |
| `step()` 内 debug print | strategies/channel_deviation.py:148 | 删（每桶打印，最大噪声源） |
| to_tensor/to_host | backends.py | **保留**（torch 后端 API，测试锚定） |

## 3. 行为 bug 修复

- `pick_best_row` runner-up：现取"第一个非选中行"，改真 rank-2
  （`others = index.difference([chosen]); runner = others[argmax(score[others])]`），
  加 4 行 frame 测试（chosen 非 row 0 时 runner_up_score = 次高 score）
- `permutation.py:38,63`：`m.get("ann_excess_pct", 0.0)` → `cagr_excess`
  （字段已改名，现在 p 值恒对 0.0 算）；sweep `--mc` 冒烟验证

## 4. 测试影响预估

147 passed → 预计 ~134：
- 删：test_capability 中 can_run/TARGET_CAPS 6 测、NumpyDictFeed 3 测、xp_ema_torch 3 测、
  to_tensor/to_host 保留、timeutils legacy 等价 1 测、harness sweep import 1 测 ≈ -14
- 增：trade_decision 对照表 + 双路径逐笔 2 测、pick_best_row rank-2 1 测、
  feeds 目录 guard 1 测 ≈ +4
- cpu-vs-gpu bitwise 测试（本机无 CUDA，skip）改为纯 CPU 断言或删除 gpu 腿

## 5. 验证门禁（每包过一遍）

```
npx openspec validate --specs
uv run pytest -q
uv run python -m evtrade replay --log cache/reconcile_smoke.csv \
    --strategy channel_deviation --device cpu --against-ref   # PASS 且无逐桶噪声
```

grep hygiene（P8 终验）：
```
grep -rn "feeds\|gpu_info\|capability\|BrokerExecutor\|xp_ema\|atr_step\|rsi_step\|boll_step" \
    evtrade/ kbs/ CLAUDE.md openspec/specs/        # 0 命中
grep -rn "compute_bucket\b\|daterange\|PERIODS\|trades_to_list\|build_engine\|print_summary" \
    evtrade/ kbs/ CLAUDE.md openspec/specs/        # 0 命中
grep -rn -- "--no-sleep\|--step-days\|--show-bars\|--bars-out\|ann_excess_pct" \
    evtrade/ kbs/ CLAUDE.md                        # 0 命中
```
