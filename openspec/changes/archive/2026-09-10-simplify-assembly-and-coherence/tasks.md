# Tasks: simplify-assembly-and-coherence

## 1. 行为修复 (spec delta)

- [x] 1.1 删 `--grid all_in` 空转轴: sweep.py `GRID_KEYS` 去 `all_in`、`parse_grid` 去 bool 分支、`run_one_from_dict` docstring 去 all_in、cli.py `base["all_in"]` 删
- [x] 1.2 `replay --against-ref --all-in` 两腿对称: cli.py replay 侧 all_in→buy/sell_pct 解析后统一传入 replay_vectorized + reconcile

## 2. 纯重构 (行为逐位一致)

- [x] 2.1 `run_one_from_dict` 委托 `run_one_vectorized` (保留签名)
- [x] 2.2 cli.py 删四个 `build_*_parser` 死分支 (ap 必填)
- [x] 2.3 抽 `_write_signals_csv` helper (backtest/replay 共用) + 删 `_f4` (summary 打印块不动)
- [x] 2.4 `vectorized_engine._summarize` 内联进 `run_vectorized`
- [x] 2.5 `INIT_CASH/INIT_POSITION/TRADE_QTY` 字面量默认值改 config 常量引用 (vectorized_engine/sweep/replay)
- [x] 2.6 `_defaults_loader`: `_git()` helper + `import subprocess` 移顶
- [x] 2.7 注释修正: permutation.py docstring 删不存在的测试引用; metrics.py/vectorized_engine.py 「25 字段」→「30 字段」

## 3. KB / 文档同步

- [x] 3.1 kbs/10: `--grid` 行去 all_in 暗示 (若提及); 性能/FAQ 若有 all_in 网格表述同步
- [x] 3.2 kbs/09: `_summarize` 提及改为 "run_vectorized 内联调用 metrics.summarize"
- [x] 3.3 kbs/12: 结构/字段数核对 (30 字段); 包结构若提及 _f4/_summarize 同步
- [x] 3.4 kbs/13: 置换检验测试引用核对 (勿指向不存在的 test_sweep.py)
- [x] 3.5 使用说明: sweep 网格轴说明核对 (buy_pct/sell_pct 表达资金模式)

## 4. 验证与收尾

- [x] 4.1 `uv run pytest -q` 全绿 (142 passed)
- [x] 4.2 design §3 门禁全部通过 (all_in 报错 / all-in 对账 PASS / 回归对账 PASS / grep 0 命中)
- [x] 4.3 主 spec 合入 delta + `npx openspec validate --specs`
- [x] 4.4 archive + commit (消息注明 change 名) + push origin/pytorch
