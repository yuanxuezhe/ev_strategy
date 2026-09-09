# Tasks: dedupe-channel-deviation-fsm-helpers

## 1. 抽 helper 到 channel_deviation.py 顶部

- [ ] 1.1 在 `evtrade/strategies/channel_deviation.py` 顶部（`_fsm_step` 之后或之前）加 3 个 module-level helper:
  - `_parse_thresholds(params) -> tuple[float, float, float, float]`
  - `_compute_devs_scalar(up, dw, h, l) -> tuple[float, float, float, float]` (标量版, 逐 bar 用; 含 up==0 or dw==0 短路)
  - `_compute_devs_xp(xp, up, dw, h, l) -> tuple` (xp 数组算子版, 批量用; 含 NaN→0 与 0→safe 替换 + ready mask)
  - `_run_fsm(state, ts, low_dev_h, low_dev, high_dev_l, high_dev, low1, low2, high1, high2) -> int` (薄包装 _fsm_step)

## 2. compute_signals 精简

- [ ] 2.1 `evtrade/strategies/channel_deviation.py` `compute_signals` (L104-139) 改:
  - 参数解析改 `low1, low2, high1, high2 = _parse_thresholds(params)`
  - 4 行偏离公式 (L116-119) 替换为 `low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs_xp(xp, up_v, dw_v, bars["h"], bars["l"])`
  - `_fsm_step` 调用 (L134-137) 改 `sig[i] = _run_fsm(state, int(ts[i]), float(...), float(...), float(...), float(...), low1, low2, high1, high2)`

## 3. compute_signals_for_one_bar 精简

- [ ] 3.1 `evtrade/strategies/channel_deviation.py` `compute_signals_for_one_bar` (L143-180) 改:
  - 末尾 4 行偏离公式 (L172-175) 替换为 `low_dev, high_dev, low_dev_h, high_dev_l = _compute_devs_scalar(up, dw, cur_high, cur_low)`
  - 末尾 `_fsm_step(self._fsm, ...)` (L177-180) 改 `return _run_fsm(self._fsm, cur_ts, low_dev_h, low_dev, high_dev_l, high_dev, *(_parse_thresholds(params)))`
  - 文件末段 `float(params["low1"])` 等 4 行参数解析删掉 (已在 _run_fsm 解包时取)

## 4. KB 同步

- [ ] 4.1 `kbs/06-交易策略详解.md` §5 (L80-118) 重写:
  - 删原 §5.1 / §5.2 完整代码模板 (~50 行)
  - 新 §5: 3 个 helper 位置 + 职责列表
  - 新 §5.1: compute_signals 5 行骨架 (注释 "完整看 git blame")
  - 新 §5.2: _for_one_bar 5 行骨架
  - §5.3 (基类默认实现) 不变
- [ ] 4.2 `kbs/11-扩展指南.md` 行 47-78 同步:
  - 删两个完整方法模板
  - 新增 "复用 helper" 段: 当多策略共享偏离/参数解析时, 提到模块级 `_` 函数
  - 引用 kbs/06 作详细模板

## 5. 测试 + 端到端验证

- [ ] 5.1 `uv run pytest tests/test_strategy_unified.py -v`:
  - test_vectorized_vs_engine_on_bars_reconcile PASS (重构前后 bitwise 一致)
  - test_cpu_vs_gpu_* PASS
  - test_metrics_summary_has_16_fields PASS
- [ ] 5.2 `uv run pytest -q` 122+ 全 PASS (含 test_metrics_units 9 个)
- [ ] 5.3 跑一次 channel_deviation e2e backtest (cpu), 记录 cagr/calmar/max_drawdown/n_trades:
  ```
  - `uv run python -m evtrade backtest --strategy channel_deviation --device cpu --period 5m --start 20250101 --end 20260903 --no-sleep`
  ```
- [ ] 5.4 跑一次 channel_deviation e2e backtest (gpu), 同上记录:
  ```
  - `uv run python -m evtrade backtest --strategy channel_deviation --device gpu --period 5m --start 20250101 --end 20260903 --no-sleep`
  ```
- [ ] 5.5 对比 5.3 vs 5.4 vs 重构前 (unify-metrics-units 落地后那次跑的数据):
  - cagr 应 bitwise 一致 (除 _compute_devs_xp 的 safe 改动可能影响极端 bar)
  - n_trades 应一致或差 ±1
  - max_drawdown 应一致或差 ±0.01
  - 若差异 > 上述, 回滚并分析

## 6. openspec archive

- [ ] 6.1 把 change 目录挪到 archive (无 openspec CLI, 手工 mv):
  ```
  - `mv openspec/changes/2026-09-10-dedupe-channel-deviation-fsm-helpers openspec/changes/archive/`
  ```
- [ ] 6.2 spec 不需合并 (本次无 spec 改动, delta 仅声明 "No Requirements Changed")
- [ ] 6.3 git commit:
  ```
  - feat(strategies): dedupe channel_deviation FSM helpers (compute_signals 38->22 行, _one_bar 40->22 行, 净减 ~42 行)
  ```