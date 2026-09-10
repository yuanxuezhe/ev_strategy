# 14 · 旧策略 DSL 与三端转译 (已下线, 2026-09 删除)

> **本文档为存档说明。** 旧策略 DSL 与三端转译 (docstring DSL body + numba
> `@njit` 流式内核 + NVRTC CUDA `__device__` 渲染器 + `state_spec` AST 白名单三端投影)
> 已于 2026-09 整体下线, 对应代码 (`evtrade/strategies/dsl.py`、`core/kernel.py`、
> `core/gpu.py::cuda_sweep_window_generic`) 已删除。
>
> **当前唯一策略契约**: `VectorizedStrategy.step(state, bar, params) -> (state, sig)`
> + `init_state(params)`, 持久状态 = `@dataclass` (engine 持有, 策略无 instance attr),
> 展示 hook = `format_signal_line(ts, sig, info)`。批量 (vectorized) 路径与逐 bar
> (Engine.on_bars) 路径调用**同一个 `step`**。
>
> 写新策略请读 11 号文档 §3; 契约细节见 `evtrade/strategies/vectorized_base.py`。

## 1. 旧形态回顾 (2026-09-08 及之前)

重构前的形态 (仅供理解历史提交, 代码已不存在):

- 策略主逻辑写在 `compute_signal` 的 **docstring** (DSL body), 由 AST 白名单
  (`_CALL_WHITELIST`) + 三端渲染器 (Python exec / numba `@njit` / CUDA C99) 自动转译。
- 同一份 AST 三端共用, 浮点路径 bitwise 一致 (依赖 `--fmad=false`)。
- 持久状态由 `state_spec` AST 白名单声明, 投影到 Python 字段 / numba jitclass /
  CUDA `__device__` 结构体三处, 新增一个字段要同步改三端。
- 痛点: AST 白名单维护成本高; 三端语义需手动对齐; numba 编译开销影响冷启动;
  CUDA 通用 kernel 只支持 ≤8 参数; 策略复杂度受 DSL 表达能力限制。

## 2. 下线过程

| 时间 | change | 动作 |
|---|---|---|
| 2026-09-09 | `unify-strategy-contract` | DSL 渲染管线整体下线: 删除 DSL body / numba 流式内核 / NVRTC CUDA 编译; 临时改走 `compute_signals(xp, bars, params)` 抽象入口 + 策略 instance attr 状态 |
| 2026-09-10 | `strategy-step-only` | 唯一抽象方法收敛到 `step(state, bar, params) -> (state, sig)`; state 由 engine 持有 (`@dataclass`), 策略无 instance attr; 批量/逐 bar 两路径循环调同一份 `step` |
| 2026-09 (consolidate-simplify-core) | — | 过渡期入口 `compute_signals` 也删除; 三端投影 / `state_spec` 无残留 |

## 3. 旧 → 新 对照表

| 维度 | 旧 (已删) | 新 (现行) |
|---|---|---|
| 策略写法 | docstring DSL body, 三端转译 | Python 覆写 `step(self, state, bar, params) -> (state, sig)` |
| 批量入口 | 旧: DSL 渲染的 `compute_signals(xp, bars, params)` 路径 | 无; vectorized 路径 (`core/vectorized_engine` 内部逐桶信号循环) 调同一份 `step` |
| 持久状态 | `state_spec` AST 白名单 → Python / numba jitclass / CUDA 三端投影 | `@dataclass` state, 引擎持有跨调用; 无 instance attr |
| 渲染层 | Python exec + numba `@njit` + NVRTC CUDA `__device__` | 无; 一份 Python 代码, torch CPU / CUDA 同跑 |
| 指标调用 | DSL 白名单内的三端 stub (旧 `xp_ema` 等批量算子, 已删) | `evtrade.indicators.ema` 内 `ema_step` / `ema_channel_step` (增量 @dataclass state) + numpy 版 `ema` / `ema_channel` |
| 信号展示 | 旧 `get_extra_bucket_columns` / `get_extra_signal_columns` | `format_signal_line(ts, sig, info)` 单 hook |
| 参数上限 | CUDA 通用 kernel 限 8 字段 | 无上限 (`params_spec` 校验) |
| 一致性保证 | 三端 bitwise (`--fmad=false`) | 两路径同 `step`, 浮点路径一致 (reconcile 锁定) |

## 4. 迁移指引 (旧 DSL 策略 → 现行契约)

1. 把 DSL body 里的指标递推 (如 EMA) 改写为 `step` 内调用 `ema_step(state, value, p)` /
   `ema_channel_step(state, h, l, p)`, state 收敛进一个 `@dataclass`。
2. 把 FSM / 锁存逻辑写成 `step` 内的纯函数 + state 字段; **不要**用 `self._xxx`
   instance attr。
3. `init_state(self, params)` 返回 state 初值 (无状态策略返回 `None`)。
4. 展示列如需自定义, 覆写 `format_signal_line(ts, sig, info)`。
5. 验证: `tests/test_strategy_unified.py` (step 状态跨调用持续 + CPU/GPU 对账) +
   `python -m evtrade replay --log <log.csv> --strategy <key> --device cpu --against-ref`。

## 5. 跨文档索引

- kbs/06 §5 — 策略 `step` 骨架 (channel_deviation / ma_crossover 范式)
- kbs/09 §2 — `Engine.on_bars` 桶 CLOSE 语义 (桶切换时调 step)
- kbs/11 §3 — 新策略开发模板
- kbs/12 — 重构与性能内核 (旧 numba / CUDA 章节已废)
- kbs/15 — PyTorch 统一后端 (torch CPU/CUDA 单后端)
- spec.md — `Single contract = VectorizedStrategy.step`
