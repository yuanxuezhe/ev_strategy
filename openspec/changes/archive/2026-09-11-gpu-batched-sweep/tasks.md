# Tasks — GPU-Batched Sweep

## 1. spec / KB 同步（CLAUDE.md §0 step 1-2）

- [ ] 1.1 `openspec/specs/evtrade-architecture/spec.md` 末尾追加 Requirement "Optional GPU-batched sweep hook (`batched_step`)" + 4 Scenario
- [ ] 1.2 `kbs/15-PyTorch统一策略.md` 新增 §7.4 "Batched sweep hook (可选)"
- [ ] 1.3 `kbs/12-重构与性能内核.md` 加 perf note 一段
- [ ] 1.4 `kbs/13-绩效评估与鲁棒选参框架.md` §3 用法加一句
- [ ] 1.5 `kbs/10-配置参数与运行指南.md` --device 描述追加

## 2. indicators/ema.py — torch_ema

- [ ] 2.1 新增 `torch_ema(values: Tensor, p: int) -> Tensor`
  - 输入 `[T]`，输出 `[T]`
  - dtype/device 保留；自动转 float64
  - 前 p-1 个 0.0（跟 `ema_step` 一致）
  - 算法：`cumsum` seed（`cumsum[p-1]/p`）+ 递推尾段 `out[i] = values[i]*k + out[i-1]*(1-k)`
  - k = 2/(p+1)
  - docstring 注明：跟 numpy `ema()` float64 bit-equal
- [ ] 2.2 `__init__.py` 导出 `torch_ema`

## 3. vectorized_base.py — 默认 hook

- [ ] 3.1 新增 `@classmethod batched_step(cls, state, bars, params, *, n_combos, n_bars)` raise NotImplementedError
- [ ] 3.2 docstring 注明 "opt-in GPU 批量 hook; 默认未实现, sweep 自动走 ThreadPool"
- [ ] 3.3 不动 `step` / `init_state` / `_resolve_params` / `register_strategy`

## 4. ma_crossover.py — batched_step 实现

- [ ] 4.1 新增 `@dataclass class MABatchedState`（Tensor[N] fields）
- [ ] 4.2 新增 `@classmethod batched_step(cls, state, bars, params, *, n_combos, n_bars)`：
  - c = bars["c"] Tensor[T] → float64
  - fast/slow 提取 param Tensor[N] int64
  - 调 torch_ema 出 fast_ema[T,N] / slow_ema[T,N]（按 tf1 值分组）
  - mark=0 段 fast/slow 都累积 count（跟原 step 同语义）
  - mask: 双线任一 count<period → 该 bar 该 combo 产 sig=0
  - has_prev=False → 记 prev_diff 不产 sig
  - cross 检测同原 step 公式
  - 返回 (new_state, sig [N, T] int8)
- [ ] 4.3 不动原 step / MACrossoverState / params_spec

## 5. core/batched_sweep.py — 编排

- [ ] 5.1 新增 `run_batched(bars, base, combos, *, strategy_cls, device, wins, params_list, split_ymd, fee_bp, lam, min_trades, max_mdd) -> list[list[dict]]`
- [ ] 5.2 流程：
  - 对每个 win: window_bars 切片
  - `_aggregate_buckets`（共享）
  - numpy → to_tensor(device)
  - 构建 batched_state (init MABatchedState)
  - 调 `cls.batched_step` 一次出 sig[N, T]
  - sig 转 numpy，per-combo 切片给 `_execute_trades`（现有）
  - metrics 计算（per-combo 调 summarize）
  - 填 metrics[win_idx][combo_idx]
- [ ] 5.3 异常透传：batched_step 抛 → run_batched 直接抛，不吞
- [ ] 5.4 输出跟 ThreadPool 路径一致：metrics[win][combo] = summary dict

## 6. core/sweep.py — routing 分支

- [ ] 6.1 在 `sweep()` 设备解析后插入 `use_batched` 检查（按 design.md 规则）
- [ ] 6.2 抽 `_finalize_sweep_df(metrics, combos, wins)` 共享
- [ ] 6.3 把现有 ThreadPool 主体重命名 `_threadpool_sweep`，复用同一 metrics 收集逻辑
- [ ] 6.4 `if use_batched: try run_batched except OOM: warn + ThreadPool`
- [ ] 6.5 `--workers` 在 batched 模式下打印忽略 notice（不改 CLI 解析）
- [ ] 6.6 WFO / permutation / save_defaults / score / S / pareto 不动

## 7. tests/test_batched_sweep.py — 测试

- [ ] 7.1 `test_batched_step_default_raises` — VectorizedStrategy.batched_step raise NotImplementedError；channel_deviation 不覆写
- [ ] 7.2 `test_ma_crossover_batched_step_equivalence` — 同一 bars+params，batched_step vs per-combo step 循环 sig bit-equal（int8 np.array_equal）
- [ ] 7.3 `test_torch_ema_matches_numpy_reference` — p ∈ {2,5,21,60}，200 随机值，bit-equal
- [ ] 7.4 `test_ma_crossover_batched_mark_zero_handling` — mark=0 段双线 EMA 都累积，sig 全 0
- [ ] 7.5 `test_batched_sweep_no_instance_state` — MABatchedState 不挂 self._xxx；batched_step 是 classmethod
- [ ] 7.6 `test_sweep_routing_to_batched_when_available` — monkeypatch device / n_combos 断言路由
- [ ] 7.7 `test_sweep_batched_output_schema_matches_cpu` — 同 grid 双路径 DataFrame 列集 + score 数值相等
- [ ] 7.8 `test_sweep_batched_score_ranking_unchanged` — 双路径 score 排序 top-N 一致
- [ ] 7.9 `test_sweep_batched_propagates_exceptions_immediately` — batched_step 抛 → sweep 同步抛
- [ ] 7.10 `test_sweep_batched_fallback_to_cpu_when_no_cuda` — 无 CUDA 走 ThreadPool
- [ ] 7.11 `test_sweep_batched_cuda_oom_falls_back` — monkeypatch OOM 回退 ThreadPool + 提示

## 8. 验证（CLAUDE.md §6）

- [ ] 8.1 `openspec validate --specs` 通过
- [ ] 8.2 `uv run pytest -q` 全过（含 7.x 新增）
- [ ] 8.3 `uv run pytest tests/test_strategy_unified.py tests/test_sweep_save_defaults.py tests/test_sweep_window.py tests/test_vectorized.py tests/test_assembly_coherence.py -q` 全过
- [ ] 8.4 hygiene grep 三条（CLAUDE.md §6）仍 0 命中
- [ ] 8.5 manual smoke：4 个 sweep 命令（cpu/cd / cpu/ma / auto/ma / gpu/ma）

## 9. archive

- [ ] 9.1 `openspec archive gpu-batched-sweep` 落地
- [ ] 9.2 确认 spec / KB / 代码三者对齐
