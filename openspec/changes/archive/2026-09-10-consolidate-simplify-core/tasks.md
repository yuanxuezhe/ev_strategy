# Tasks: consolidate-simplify-core

## 0. openspec 脚手架
- [x] 0.1 proposal.md / design.md / specs delta / tasks.md 建好
- [x] 0.2 `npx openspec validate --specs` + `validate --changes` 通过

## 1. 成交执行单一实现 (P1)
- [x] 1.1 `execution/base.py` 加 `trade_decision(side, price, cash, position, cur_qty,
      buy_pct, sell_pct) -> (new_cash, new_position, qty, filled)`；`SimulatedExecutor.trade`
      改薄封装（scale/last_side 逻辑保留）；`execution/__init__.py` 导出
- [x] 1.2 `core/vectorized_engine.py::_execute_trades` BUY/SELL 分支改调 `trade_decision`
      （`cash_after` 由调用方用返回的 new_cash 填充）
- [x] 1.3 新增 `tests/test_trade_unified.py`：trade_decision 对照表 + Executor vs
      _execute_trades 逐笔一致
- [x] 1.4 `uv run pytest tests/test_trade_unified.py tests/test_strategy_unified.py -q`
      + reconcile 冒烟 PASS

## 2. 指标收敛到 EMA (P2)
- [x] 2.1 `git rm indicators/{atr,rsi,boll}.py`；`ema.py` 删 `xp_ema`/`xp_ema_channel`/
      `xp_ema_torch`/`xp_ema_channel_torch`/`_resolve_period_torch` + `import torch`，
      留 `EMAState/EMAChannelState/ema_step/ema_channel_step/ema/ema_channel`，重写 docstring
- [x] 2.2 `indicators/__init__.py` 重写（仅 6 符号）；`evtrade/__init__.py` 去相应 re-export
- [x] 2.3 `tests/test_torch_backend.py` 删 xp_ema_torch 3 测
- [x] 2.4 pytest 全绿 + `grep -rn "atr\|rsi\|boll\|xp_ema" evtrade/ tests/` 0 命中（注释除外）

## 3. 删 feeds/ + BrokerExecutor (P3)
- [x] 3.1 `git rm -r evtrade/feeds/`；`timeutils.daterange` 删除
- [x] 3.2 `_harness.py` 删 NumpyDictFeed（留 ListBarFeed）；`execution/base.py` 删
      BrokerExecutor；`evtrade/__init__.py` 去 feeds/BrokerExecutor 导出
- [x] 3.3 `test_harness.py` 删 NumpyDictFeed/sweep-import 3 测；`test_dead_code_removed.py`
      加 feeds 目录不存在 guard
- [x] 3.4 pytest + `python examples/demo_format_signal_line.py` + reconcile 冒烟

## 4. tsbucket + capability 收编 + device plumbing 删除 (P4)
- [x] 4.1 `git mv core/gpu.py core/tsbucket.py`；删 `gpu_info`；`_encoded_to_epoch_np`/
      `_epoch_to_encoded_np` 移入 `timeutils.py` 与标量版共享 Hinnant 函数（numpy 广播），
      roundtrip 测试扩展到 int64 数组
- [x] 4.2 删 `core/capability.py`；`select_device` → `backends.resolve_device(requested,
      gpu_available=None)`（去 strategy_name 参数）
- [x] 4.3 删 `run_vectorized`/`run_one_vectorized`/`run_one_from_dict`/`replay_vectorized`
      的 device 参数及全部 `device=` 调用点（cli/replay/sweep/tests）
- [x] 4.4 `cli._run_backtest` / `replay_main` / `_run_sweep` 入口 `args.device =
      resolve_device(args.device)`；replay banner 打印 resolved device
- [x] 4.5 `test_gpu_precompute_cache.py` import 改 tsbucket；`test_capability.py` 收敛为
      resolve_device + gpu_available 测试
- [x] 4.6 pytest + `backtest --device gpu` 无 CUDA 抛 ValueError +
      `sweep --synthetic-days 15 --device auto` 正常降级

## 5. CLI 整理 (P5)
- [x] 5.1 删 dead flags `--no-sleep/--step-days/--show-bars/--bars-out`；删 `--engine`
      兼容层全部 4 处
- [x] 5.2 `_auto_cast` → `_defaults_loader.auto_cast`（公开），删 cli 副本，
      `test_defaults_loader.py` 同步
- [x] 5.3 共享 `_common_parent()`；backtest/sweep/replay `parents=` 化；`build_root_parser`
      拷贝循环改同 builder 注册
- [x] 5.4 cli 期初资金/持仓打印 `args.init_cash/init_position`；删 `INTERVAL`（cli import +
      config 定义）
- [x] 5.5 pytest test_cli/test_cli_params/test_backtest_defaults + 三个子命令 `--help`

## 6. 其余死代码 + bug (P6)
- [x] 6.1 删 `config.PERIODS` / `timeutils.compute_bucket+_bucket_plus_period` /
      `metrics.trades_to_list` / `engine.build_engine` / `Engine.print_summary`；删
      test_timeutils legacy 等价测
- [x] 6.2 `channel_deviation.py`：删 step() 内 debug print + `Deviations.ready`
- [x] 6.3 `_defaults_loader.pick_best_row` runner-up 改真 rank-2，加测试
- [x] 6.4 `permutation.py`：`ann_excess_pct` → `cagr_excess`；`sweep --mc 5
      --synthetic-days 15` 冒烟
- [x] 6.5 pytest 全绿

## 7. __init__.py 瘦身 (P7)
- [x] 7.1 `test_timeutils.py` 改从 `evtrade.core.*` import → 删 `evtrade.kernel` stub
- [x] 7.2 shim 收敛 7 项（data/metrics/account/aggregator/replay/execution/primitives）；
      `__all__` 同步
- [x] 7.3 pytest 全绿（import 级即验证）

## 8. 文档同步 (P8)
- [x] 8.1 主 spec 合入 delta；`npx openspec validate --specs`
- [x] 8.2 kbs：05(仅EMA)/07(trade_decision, 删Broker)/08(重写"行情数据加载")/02/12(包结构表)/
      15/10(参数表删旧flag)/09(删print_summary/build_engine)/01/11/13/README/使用说明；
      root README；CLAUDE.md §3/§5/§6
- [x] 8.3 grep hygiene：stale 命中 0（design §5 三条）；残留命中仅为 spec 自身的
      MUST-NOT 删除 scenario、CLAUDE.md §6 校验命令文本、kbs/README 的"已删除"变更日志——非陈旧

## 9. 端到端 + archive (P9)
- [x] 9.1 `uv run pytest -q`（~134 passed）
- [x] 9.2 reconcile 冒烟 PASS 且无逐桶 debug 噪声
- [x] 9.3 backtest e2e cpu：cagr/calmar/n_trades 与重构前一致（P1 不改信号）
- [x] 9.4 `openspec archive consolidate-simplify-core`；commit message 带 change 名
