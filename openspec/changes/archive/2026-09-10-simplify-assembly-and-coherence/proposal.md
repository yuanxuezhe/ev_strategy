# Change: simplify-assembly-and-coherence

## Why

全量代码扫描发现 ~110 行可安全收敛的重复与陈旧引用（不影响任何回测/对账结果），
外加两处行为诚实性问题：

1. **重复装配**：`sweep.run_one_from_dict` / `sweep.run_one_vectorized` /
   `replay.replay_vectorized` 三处各自复制「get_strategy + run_vectorized + summary」；
   cli.py backtest / replay 各自复制 `--signals-out` CSV 写入。
2. **死代码**：cli.py 四个 `build_*_parser` 的 `ap is None:` 分支零调用方；
   `_f4` 零调用方。
3. **latent bug（实施中发现）**：`--signals-out` 在 backtest 与 replay 两条路径都以
   1m bar 数组数索引**桶级** sig 数组（桶数 < bar 数）——warmup 存在时越界崩溃，
   无 warmup 时写出大量 sig=0 错位行。修法：`replay_vectorized` 返回与 sig 对齐的
   桶级 `"ts"`，两路径统一 zip(ts, sig) 写出桶级行。
4. **常量漂移**：`INIT_CASH` / `INIT_POSITION` / `TRADE_QTY` 在 config.py 声明后，
   又在 5 个引擎/回放函数签名里以字面量重复——改 config 常量不会传导（config.py
   自己警告过这一点）。
5. **静默空转的网格轴**：`--grid all_in=...` 被 `GRID_KEYS` 接受但下游从不读取
   （看起来在扫资金模式，实际什么都不做）。
6. **对账不对称 bug**：`replay --against-ref --all-in` 时 vectorized 腿用
   `--buy-pct/--sell-pct`，Engine 腿 all-in，两腿有效成交参数不同 → 对账必然 FAIL。
7. **陈旧注释**：permutation.py 声称被不存在的 `tests/test_sweep.py` 锁定；
   metrics.py / vectorized_engine.py docstring 说「25 字段」（实际 30）。

## What Changes

### 行为变化（spec delta 覆盖）

- `--grid all_in=...` 从"接受但静默空转"改为 `ValueError: 不支持的网格参数 'all_in'`
  （资金模式网格化请用 `--grid buy_pct=.../sell_pct=...`）。backtest 的 `--all-in`
  flag 不变。
- `replay --against-ref --all-in` 修复：`all_in` 在 CLI 边界解析为
  `buy_pct=sell_pct=1.0` 后**两腿统一传入**（与 backtest 同模式），对账不再必然 FAIL。
- `--signals-out` 修复（backtest + replay）：从"按 1m bar 数索引桶级 sig"（越界/错位）
  改为桶级 (ts, sig) 逐对写出；`replay_vectorized` 返回值新增 `"ts"` 键。

### 纯重构（行为逐位一致，无 spec delta）

- `run_one_from_dict` 改为委托 `run_one_vectorized`（保留函数名与签名：
  permutation.py 与测试引用）。
- `--signals-out` 写入抽为 `_write_signals_csv` helper（backtest / replay 共用）；
  删 `_f4`。
- 删四个 `build_*_parser` 的死 `ap is None:` 分支（ap 变必填参数）。
- `vectorized_engine._summarize` 内联进 `run_vectorized`（29 行单调用方透传）。
- `vectorized_engine` / `sweep` / `replay` 的 `init_cash / init_position / trade_qty`
  签名默认值改引用 `core.config` 常量（当前值相同，无结果变化）。
- `_defaults_loader._commit_defaults_file` 三次 git subprocess 样板抽 `_git()` helper，
  `import subprocess` 移到文件顶。
- 注释修正：permutation.py docstring（删除对不存在测试文件的引用，改为"方法见
  kbs/13 第 3 层"）；metrics.py / vectorized_engine.py / core/__init__.py docstring
  「25 字段」→「30 字段」。

## Impact

- specs: `evtrade-architecture`（MODIFIED CLI surface + ADDED sweep 网格轴白名单）
- code: `evtrade/cli.py`, `evtrade/core/sweep.py`, `evtrade/core/replay.py`,
  `evtrade/core/vectorized_engine.py`, `evtrade/strategies/_defaults_loader.py`,
  `evtrade/core/permutation.py`, `evtrade/core/metrics.py`, `evtrade/core/__init__.py`
- kbs: 10（grid 行）、12（字段数/结构）、13（置换检验引用）、09（_summarize 提及）、使用说明
- tests: 无删改；`test_sweep_window` monkeypatch 的 `run_one_vectorized` 签名保持不变；
  `test_metrics_units` 的 `inspect.getsource(_run_backtest)` 正则约束的 summary 打印块
  **不动**（只把 signals-out 块移出到 helper）
- 明确不做：engine/vectorized_engine 双实现保持独立（reconcile 依赖）；
  Account.equity/baseline_equity 保留（kbs/07 实盘预留 API）
