# Tasks: 框架不假定任何指标

## 1. Spec / KB 同步（CLAUDE.md §0）

- [ ] 1.1 改 `openspec/specs/evtrade-architecture/spec.md`：新增 Requirement "Indicators are private to strategies"；改 Requirement "Strategy interface contract"（删兼容 (cur,up,dw) 措辞）；改 Requirement "Per-bar dict contract"（删 up/dw 框架键）；改 Requirement "DSL→CUDA projection"（白名单包含 indicators 函数）
- [ ] 1.2 同步 `kbs/02-系统架构.md`：删 EMA 引用；engine.on_bars 改成不调 EMAChannel
- [ ] 1.3 同步 `kbs/05-指标计算-EMA通道.md`：从 framework 私有改成 indicators 子包；新增增量 API 段
- [ ] 1.4 同步 `kbs/06-交易策略详解.md`：channel_deviation DSL body 两段式
- [ ] 1.5 同步 `kbs/09-引擎Engine与主流程.md`：on_bars 删 EMA 段
- [ ] 1.6 同步 `kbs/11-扩展指南.md`：新增指标 = indicators + 白名单一行
- [ ] 1.7 同步 `kbs/12-重构与性能内核.md`：KernelState 字段表删 EMA；GPU 模板删 EMA
- [ ] 1.8 同步 `kbs/14-策略DSL与三端转译.md`：DSL 白名单扩展；调用 indicators 写法示例
- [ ] 1.9 跑 `openspec validate --specs` 通过

## 2. indicators 子包扩展（DSL 可调用增量 API）

- [ ] 2.1 `evtrade/indicators/ema.py`：新增 `@njit(cache=True) ema_push(s_sum, s_count, s_ema, value, p) -> (sum, count, ema)` / `ema_current(s_sum, s_count, s_ema, p, pending) -> value` / `ema_channel_push(up_s_sum, up_s_count, up_s_ema, dw_s_sum, dw_s_count, dw_s_ema, h, l, p) -> tuple` / `ema_channel_current(...) -> (up, dw)`（unready 返回 `inf`）
- [ ] 2.2 `evtrade/indicators/atr.py`：新增 `@njit atr_push / atr_current`
- [ ] 2.3 `evtrade/indicators/rsi.py`：新增 `@njit rsi_push / rsi_current`
- [ ] 2.4 `evtrade/indicators/boll.py`：新增 `@njit boll_push / boll_current / sma_push / sma_current`
- [ ] 2.5 `evtrade/indicators/__init__.py`：导出所有新增增量 API
- [ ] 2.6 验证：`uv run python -c "from evtrade.indicators.ema import ema_push, ema_current; print(ema_push(0,0,0,1.0,5))"`

## 3. DSL 白名单扩展（`strategies/dsl.py`）

- [ ] 3.1 `_CALL_WHITELIST` 加入所有 indicators 增量 API
- [ ] 3.2 `core/kernel_dsl.py::_build_dsl_kernel_impl`：渲染 numba 函数体前 `from evtrade.indicators.ema import ema_channel_push, ema_channel_current` 等
- [ ] 3.3 `strategies/dsl.py::render_cuda_device_function`：白名单函数调 CUDA 等价实现（开始仅 ema 系列，ATR/RSI/Boll 留 TODO）
- [ ] 3.4 验证：`make_python_runner` / `render_numba_body` / `render_cuda_device_function` 对包含 `ema_channel_push(...)` 的 DSL 都能产出源码

## 4. 策略改造（`strategies/channel_deviation.py`）

- [ ] 4.1 `state_spec` 加 6 个 EMA 增量状态字段（up_st_sum/count/ema, dw_st_sum/count/ema）
- [ ] 4.2 `params_spec` 加 `tf1` 参数（int, default=21, min=2, max=1000）
- [ ] 4.3 DSL body 第一段调 `ema_channel_push` / `ema_channel_current` 维护指标；第二段信号逻辑不变
- [ ] 4.4 `check(cur, indicators)` 不再用 framework 灌的 up/dw
- [ ] 4.5 `get_extra_signal_columns` 用 `indicators.ema.ema_channel(per_bar["h"], per_bar["l"], self.params["tf1"])` 一次性算 per-bar 数组（供 `--signals-out` 用）

## 5. Framework 解耦

- [ ] 5.1 `evtrade/core/incremental_indicators.py` 整文件删除
- [ ] 5.2 `evtrade/core/engine.py`：
  - `__init__` 删 tf1 / ema_ch / _pushed 字段
  - 删 `_sync_ema` 方法
  - `on_bars` 删 EMAChannel 调用；直接 `self.strategy.check(cur, {})`
  - `build_engine` 删 tf1
- [ ] 5.3 `evtrade/core/kernel.py`：
  - `KernelState` 两个版本都删 EMA 6 字段
  - 删 `_ema_push` / `_ema_current` / `_ema_channel`
  - `step()` 删 EMA 推入与计算两步
  - `make_state_general` 不再传 tf1（kernel_dsl.py 同步）
- [ ] 5.4 `evtrade/core/gpu.py`：
  - CUDA 模板删 tf1s / k_ema / 6 register / 内联 push-current
  - host 端删 tf1s 数组分配与上传
- [ ] 5.5 `evtrade/core/kernel_dsl.py`：make_state_general 删 tf1 参数；构造 KS 时不再传 tf1
- [ ] 5.6 `evtrade/core/replay.py`：replay_kernel / replay_engine / reconcile 删 tf1
- [ ] 5.7 `evtrade/core/sweep.py`：GRID_KEYS 删 tf1
- [ ] 5.8 `evtrade/core/config.py`：删 TF1 常量
- [ ] 5.9 `evtrade/__init__.py`：
  - 删 `from .core.incremental_indicators import EMAChannel, IncrementalEMA, ema, ema_channel`
  - 删 `__all__` 中对应 4 项
  - 删 backward-compat shim `sys.modules.setdefault("evtrade.incremental_indicators", ...)` 与 `_incremental_indicators`
  - 新增 `from .indicators.ema import ema_push, ema_current, ema_channel_push, ema_channel_current` 等
- [ ] 5.10 验证：`grep -rn "IncrementalEMA\|EMAChannel\|_ema_push\|_ema_current" evtrade/core/ # 必须 0 命中`

## 6. Tests 适配

- [ ] 6.1 `tests/test_differential.py`：删 up/dw 数组断言；`test_differential_tf1` 改用 `params["tf1"]`；保留 signal/trade/cash bitwise 断言
- [ ] 6.2 `tests/test_replay.py`：删 `tf1` 参数；up/dw 数组断言改由策略 `get_extra_signal_columns` 提供
- [ ] 6.3 `tests/test_kernel_unit.py`：删 `_ema_push` / `_ema_current` 直测；新增 `indicators.ema.ema_push` / `ema_channel_push` 单测
- [ ] 6.4 `tests/test_dsl_cuda.py`：保留通用 CUDA 模板测试（按策略自管指标路径）
- [ ] 6.5 `uv run pytest -q` 全部通过

## 7. 端到端验证

- [ ] 7.1 三引擎跑同一份数据，结果 bitwise 一致：
  ```
  python -m evtrade backtest --engine ref    --strategy channel_deviation --params "tf1:21;..." --code XXX --start ... --end ...
  python -m evtrade backtest --engine kernel --strategy channel_deviation --params "tf1:21;..." --code XXX --start ... --end ...
  ```
- [ ] 7.2 新增指标扩展性演示（验证 framework 0 改动）：
  - 在 `indicators/momentum.py` 加 `@njit momentum_push / momentum_current`
  - 在 `dsl.py::_CALL_WHITELIST` 加 2 个名字
  - 写一个新策略 DSL body 调它——三端自动支持，framework 0 改动

## 8. Archive

- [ ] 8.1 `openspec validate 2026-09-09-decouple-indicators-from-framework` 通过
- [ ] 8.2 `openspec archive 2026-09-09-decouple-indicators-from-framework --yes` 把 change 移到 archive/，spec delta 同步到主 spec
- [ ] 8.3 提交：`git commit -m "refactor(decouple-indicators): framework 不再预计算指标, EMA 下放到 indicators 子包 (change 2026-09-09-...)"`