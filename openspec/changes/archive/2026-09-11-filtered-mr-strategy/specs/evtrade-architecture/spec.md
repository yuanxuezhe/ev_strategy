# Spec Delta — Filtered Mean-Reversion Strategy

## ADDED Scenarios

### 新增策略 (在 R1 "Strategy interface contract" 末尾追加 brief mention)

#### Scenario: filtered_mr 是 VectorizedStrategy 子类
- **WHEN** 策略类 `FilteredMRStrategy` 继承 `VectorizedStrategy` 并实现 `step(self, state, bar, params)`
- **THEN** MUST 是 `VectorizedStrategy` 子类; `step` MUST 返回 `(new_state, signal)`, `signal ∈ {-1, 0, 1}`
- **AND** `filtered_mr` 实现 4 重过滤 (大周期顺势 / ADX 趋势强度 / ATR 波动率 / close 确认 FSM), 不实现 `batched_step` (FSM 难向量化, sweep 自动走 ThreadPool 路径)
