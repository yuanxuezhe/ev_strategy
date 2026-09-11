# Tasks — Filtered Mean-Reversion Strategy

## 1. spec / KB 同步
- [ ] 1.1 spec.md R1 加 brief mention "filtered_mr 是 VectorizedStrategy 子类, 4 重过滤策略"
- [ ] 1.2 kbs/06-交易策略详解.md 新增 §filtered_mr 章节
- [ ] 1.3 kbs/12-重构与性能内核.md perf note

## 2. evtrade/strategies/filtered_mr.py
- [ ] 2.1 FilteredMRState dataclass (大周期桶 + EMA/ATR/ADX 增量 + FSM)
- [ ] 2.2 FilteredMRStrategy 类 + params_spec (9 个参数) + register_strategy("filtered_mr")
- [ ] 2.3 init_state 返回 dataclass 实例
- [ ] 2.4 step() 完整流程: 大周期桶切换 → EMA/ATR/ADX 累积 → mark=0 早返回 → 4 重过滤 → close 确认 FSM → sig
- [ ] 2.5 format_signal_line 默认实现 (framework 默认就够)

## 3. tests/test_filtered_mr.py
- [ ] 3.1 test_init_state_returns_dataclass
- [ ] 3.2 test_step_basic_returns_int (mark=0 早返回 + sig 范围)
- [ ] 3.3 test_strong_trend_filter_blocks_all_signals
- [ ] 3.4 test_high_vol_filter_blocks_all_signals
- [ ] 3.5 test_higher_period_filter_blocks_counter_trend
- [ ] 3.6 test_close_confirmation_requires_two_bars
- [ ] 3.7 test_no_instance_state_in_strategy (正则扫描)

## 4. 验证
- [ ] 4.1 `uv run pytest tests/test_filtered_mr.py -v` 全过
- [ ] 4.2 `uv run pytest -q` 全过 (含其他已有测试)
- [ ] 4.3 hygiene grep 三条仍 0 命中
- [ ] 4.4 manual smoke: `python -m evtrade backtest --strategy filtered_mr --synthetic-days 30 --device cpu`

## 5. archive
- [ ] 5.1 手工 mv change folder 到 archive/2026-09-11-filtered-mr-strategy
