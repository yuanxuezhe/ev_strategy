# Design: simplify-assembly-and-coherence

## 1. 重构映射（逐项）

| # | 目标 | 动作 | 约束 |
|---|---|---|---|
| 1 | `sweep.run_one_from_dict` (sweep.py:51-77) | 改为 ~10 行委托 `run_one_vectorized`（保留函数名/签名：`permutation.py:37,62` 与 docstring 引用；`p` 键集语义不变） | `test_metrics_units.py:177,193` 直接调 `run_one_vectorized`（签名不动） |
| 2 | `replay.replay_vectorized` (replay.py:76-104) | 保留独立函数（返回 sig/trades/summary 三键，与 summary-only 形态不同构），仅把 `init_cash/init_position/trade_qty` 默认值改引用 config 常量 | `reconcile` 调用它 |
| 3 | cli.py 四个 `build_*_parser` 的 `ap is None:` 分支 (106-112, 233-239, 345-351, 414-421) | 删分支，`ap` 必填；重复的 prog/description 字符串只留 `*_main` 一处 | 零调用方（grep 验证） |
| 4 | `--signals-out` 写入 (cli.py:211-218 与 396-401) | 抽 `_write_signals_csv(path, rows)`；backtest 传 `(int(stime[i]), int(sig[i]))`，replay 传 `(b.stime, int(k['sig'][i]))`；删零调用方 `_f4` (122-129) | **`test_metrics_units.py:126,142,155` 用 `inspect.getsource(_run_backtest)` 正则 summary 打印块——该块 (167-209) 必须原样留在 `_run_backtest` 内** |
| 5 | `vectorized_engine._summarize` (173-201, 单调用方) | 内联进 `run_vectorized`（`init_cash/init_position/first_ts/last_ts` 均在作用域）；import 改 `from .metrics import summarize` | `run_vectorized` 返回 dict 形状不变（test_vectorized / 对账锁定） |
| 6 | `INIT_CASH/INIT_POSITION/TRADE_QTY` 字面量默认值 | `vectorized_engine.run_vectorized` (208-209)、`sweep.run_one_vectorized` (83-85)、`replay.replay_vectorized` (78-79) / `replay_engine` (109-110) / `reconcile` (160-161) 改 `from .config import ...` 引用（值相同） | 不改 config.py 值 |
| 7 | `_defaults_loader._commit_defaults_file` (236-288) | 抽 `_git(*args)` helper（subprocess.run 同 kwargs）；`import subprocess` 移到文件顶（删 138 行 noqa import） | `test_sweep_save_defaults.py:184` patch `subprocess.run`（cwd monkeypatch 后行为不变——helper 内部仍走 `subprocess.run`） |
| 8 | `--grid all_in` 空转轴 | 删 `GRID_KEYS` 中的 `"all_in"`、`parse_grid` 的 bool 分支 (43-44)、`run_one_from_dict` docstring 提及 (57)、cli.py `base["all_in"]` (288) | 行为收紧：`--grid all_in=...` 改报 ValueError（spec ADDED） |
| 9 | `replay --against-ref --all-in` 不对称 | cli.py replay 侧（`replay_main`）在调用 `replay_vectorized` 与 `reconcile` 前做与 backtest 相同的解析：`buy_pct = max(args.buy_pct, 1.0) if args.all_in else args.buy_pct`（sell 同理）；`reconcile` 的 `all_in=` 实参保留给 Engine 腿（SimulatedExecutor 内部语义），两腿有效参数一致 | 非 all-in 路径逐位不变 |
| 10 | 陈旧注释 | `permutation.py:4-6` docstring 删除 `tests/test_sweep.py::test_permutation_sanity`（文件不存在）→ 改"方法固化见 kbs/13 第 3 层"；`metrics.py:6,136` 与 `vectorized_engine.py:8` 「25 字段」→「30 字段」；`core/__init__.py:12` 已是 30 无需改（实测漂移方向：spec/kbs 说 30 为对，代码 docstring 25 为错） | 纯注释 |
| 11 | `replay --signals-out` latent bug (实施中发现) | 原代码以 1m bar 数组枚举索引桶级 sig 数组（桶数 < bar 数）：warmup 存在时越界崩溃、无 warmup 时写出 sig=0 错位行。修法：`replay_vectorized` 返回值增加 `"ts"`（桶级 ts，与 trades 同源），cli 改 zip(ts, sig) 写出 | spec ADDED（第 3 条 Requirement）；`tests/test_assembly_coherence.py` 锁定 |

## 2. 明确不做

- 不合并 `engine.py` 与 `vectorized_engine.py`（reconcile 的逐笔对账价值依赖两条独立实现）。
- 不删 `Account.equity / baseline_equity`（kbs/07 文档化的实盘预留 API；用户未表态前保留）。
- 不动 `run_vectorized` 的成交循环 / `trade_decision`（spec「Single trade-execution implementation」锁定）。
- 不动 metrics 字段集与 CLI summary 打印块（test_metrics_units inspect-source 正则）。
- **backtest 路径 `--signals-out` 同修**（发现与 replay 同款 latent bug：`result["sig"]`
  是桶级数组，原代码按 1m bar 数索引 → 同样越界/错位）。改与 replay 一致：
  `ts = result["buckets"]["ts"][mark==1]`，zip(ts, sig) 写出桶级行。两路径现统一为桶级粒度。

## 3. 验证门禁

```
uv run pytest -q                       # 142 passed 不变
npx openspec validate --specs          # 通过 (delta 合入后)
python -m evtrade sweep --synthetic-days 30 --grid all_in=true,false  # ValueError
python -m evtrade replay --log bars.csv --strategy channel_deviation \
    --device cpu --against-ref --all-in                              # PASS (两腿对称)
python -m evtrade replay --log bars.csv --strategy channel_deviation \
    --device cpu --against-ref                                       # PASS (回归)
grep -rn "_f4\|ap is None\|ap: argparse.ArgumentParser | None" evtrade/   # 0 命中
grep -n "25 字段" evtrade/core/metrics.py evtrade/core/vectorized_engine.py  # 0 命中
grep -n "test_sweep.py" evtrade/core/permutation.py                     # 0 命中
```
