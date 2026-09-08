# 14 · 策略 DSL 与三端转译 (2026-09-06; 2026-09-08 同步 state_spec 声明化)

> 一份策略主逻辑 (DSL), 三端自动可用: **Python 参考引擎 / numba 流式内核 / CUDA GPU 内核**。
> 本文说明 DSL 写法、字段契约、三端转译机制与测试锁定。写新策略前必读 11 号文档第 3 节。

## 1. 为什么做这件事

重构前 (12 号文档) 的状态: `kernel.py._strategy_check` 与 `gpu.py` 的 CUDA 段是
**手写**的 channel_deviation 策略, 新策略只能走参考引擎 (慢约 500x)。
策略逻辑因此存在三份拷贝 (参考引擎 / numba / CUDA), 每改一次策略要人工同步三处。

步骤 1~3 (2026-09-06) 的改造: 策略主逻辑写在 `compute_signal` 的 **docstring** 里
(DSL), 由 `strategies/dsl.py` 的 AST 转译器渲染成三端代码:

| 端 | 渲染入口 | 消费方 |
|---|---|---|
| Python | `make_python_runner` (直接 exec, 不重渲染) | 参考引擎 `Engine` (经 `dsl_check`) |
| numba | `render_numba_state_body` (ctx.X → st.X) | `core/kernel_dsl.py` 特化内核模块 |
| CUDA | `render_cuda_device_function` (在 `strategies/dsl.py`; ctx.X → 内核名, `__device__` 函数) | `gpu.cuda_sweep_window_generic` (在 `core/gpu.py`) |

物理保证: 同一份 AST → 同一批字面表达式 → 三端浮点路径逐位一致 (bitwise),
配合 `--fmad=false` 禁止 FMA 合并。

## 2. DSL 白名单

写法约束 (编译期 `CompileError` 拒绝, 见 `dsl._validate`):

- ✅ 算术 `+ - * /`, 比较 `< > <= >= == !=`, 布尔 `and or not`, `if / elif / else`
- ✅ 常量: 数字 / `True` / `False`; 整数字面量在三端都按 float 语义参与算术
- ✅ 标量赋值: `ctx.x = v` 与裸名 `x = v` (局部变量, 首赋值前不可读)
- ✅ `return <int>`: `1` = BUY, `-1` = SELL, `0` = 无信号; **允许提前 return**
- ❌ 字符串 / 列表 / 字典 / 元组 / 调用 (连 `min/max/abs` 也只在 Python 端可用,
  numba/CUDA 端不支持 → 不要用) / 循环 / 异常 / lambda / f-string

注意: 提前 `return` 在三端语义一致 (CUDA 端靠 `__device__` 函数的函数返回实现),
但 channel_deviation 保持 **signal 变量形式** (不提前 return), 让本桶的锁存段
在信号赋值后仍生效 —— 公式与历史实现逐位一致 (test_differential 72 项 bitwise)。

## 3. ctx 字段契约 (`dsl.build_ctx_to_kernel_map`, 动态构建)

> **2026-09-08 重构**: 旧的静态常量 `_CTX_TO_KERNEL` 已删除, 改由
> `dsl.build_ctx_to_kernel_map(strategy_class)` (dsl.py:491-502) 动态构建。
> 映射来源 = `_FRAMEWORK_CTX_FIELDS` (固定 bar info) + 策略 `state_spec` 字段 + `p0..p15`。
> **单名空间 identity 映射**: `ctx.<name>` = 内核侧 `<name>` = `state_spec[<name>]`,
> 旧的 `_bucket_ts → lock_ts` 等改名随 channel_deviation 迁移一并取消 (dsl.py:448-460)。
> 状态字段来源是策略类 `state_spec`, **框架层无任何 baked-in 字段名**。

| DSL 写法 | numba 内核 | CUDA device 函数 | 含义 |
|---|---|---|---|
| `ctx.p0` .. `ctx.p15` | `st.p0`..`st.p15` | `p0`..`p7` (寄存器) | 策略参数, **按 params_spec 声明顺序** |
| `ctx.cur_ts` | `st.cur_ts` | `cur_ts` | 当前桶时间戳 (框架字段, 非 state_spec) |
| `ctx.cur_open/high/low/close/volume` | `st.cur_*` | 同名 | 当前桶运行中 OHLCV (框架字段) |
| `ctx.up` / `ctx.dw` | 裸名 `up` / `dw` (函数参数) | 同名 | 通道上/下轨 (NaN=未就绪; 框架字段) |
| `ctx.low_hit` / `ctx.high_hit` | `st.low_hit` / `st.high_hit` | 同名 | 极端偏离锁存 (**state_spec 声明**) |
| `ctx.low_acted` / `ctx.high_acted` | `st.low_acted` / `st.high_acted` | 同名 | 本桶已操作锁 (**state_spec 声明**) |
| `ctx.lock_ts` | `st.lock_ts` | `lock_ts` | 桶切换检测 (≠cur_ts 时重置 *_acted; **state_spec 声明**) |
| 其余 `ctx.x` / 裸名 `x` | 局部变量 | 局部变量 (自动声明) | 桶内临时值, 跨桶不保留 |

> 上表中 `low_hit/high_hit/low_acted/high_acted/lock_ts` 是 channel_deviation 的
> `state_spec` 字段 (channel_deviation.py:77-83), 非框架内置; 换策略后这些字段随之改变。
> 框架字段 (`cur_*/up/dw`) 由 dsl_check / kernel step 每根 bar 注入, 不在 state_spec。

CUDA 端参数上限 8 个 (`p0..p7`), numba 端 16 个 (`p0..p15`); 超限在编译/调用期报错。
**非法引用** (如 CUDA 端用 `ctx.p8`) 在渲染期即被拒绝。

## 4. numba 端: kernel_dsl 按策略特化内核

机制 —— **一份内核源, N 个策略特化**:

1. `kernel.py` 的 `_strategy_check` 函数体位于
   `# ==== DSL-STRATEGY-BEGIN/END ====` 标记之间, 是空模板 (占位 `pass`);
2. `core/kernel_dsl.build_dsl_kernel(name)` 读 kernel.py 源码, 把标记之间的整段
   函数体替换成该策略 DSL 渲染出的同形代码, `exec` 出一个**独立内核模块**
   (jitclass / step / run_backtest / summarize 全套; numba 惰性编译, 进程内缓存);
3. `dsl_kernel(name) = build_dsl_kernel(name)`: 所有 DSL 策略 (含 channel_deviation)
   走同一渲染路径; 进程内按策略名缓存, 二次调用零额外编译。

上层 API:

- `make_state_general(name, period, warmup_until, tf1, ..., strategy_params)`:
  按 params_spec 顺序把参数填进 `p0..pN` 构造状态;
- `run_one_dsl(bars, ..., strategy_name, strategy_params)`: 单窗回测,
  指标口径 = `kernel.summarize`;
- `strategy_has_dsl(name)`: 是否有可渲染的 DSL (决定 sweep 走哪条路)。

## 5. CUDA 端: 通用 kernel + device 函数

`core/gpu.py::_CUDA_SOURCE_GENERIC_TEMPLATE` = 与旧 `sweep_kernel` 逐行等价的
桶合并/EMA/成交/绩效段 + 三个编译期占位符 (gpu.py:320-323 替换):

- `{STRATEGY_BODY}` — 注入 `render_cuda_device_function` 生成的 `__device__ __forceinline__ int strategy_check(...)`;
- `{STATE_DECLS}` — 按 `state_spec` 注入的状态字段寄存器声明 (dsl.py:542-568 `build_cuda_state_decls`);
- `{STRATEGY_STATE_ARGS}` — `strategy_check(...)` 调用处的 state arg 列表 (dsl.py:571-581 `build_cuda_strategy_check_call`)。

> **2026-09-08 迁移** (commit `dfd4785`): 上述三个投影函数 (`render_cuda_device_function` /
> `build_cuda_state_decls` / `build_cuda_strategy_check_call` / `build_cuda_device_header`)
> 已从 `core/gpu.py` 迁至 **`strategies/dsl.py`**; `gpu.py` 只保留 kernel 源码模板 + 编译 + 调度。
> 状态字段来源是策略 `state_spec`, 框架无硬编码状态字段 (见第 3 节)。

device 函数形态的关键: 状态字段 (low_hit/lock_ts/...) 按**引用**传入可写,
行情/参数为 const 引用; DSL 的 `return X` 直接成为函数返回 ——
"提前返回跳过后续语句" 的语义与 Python/numba 完全一致 (内联块做不到)。

调用: `cuda_sweep_window_generic(bars, params_list, warmup_until,
strategy_name=...)`, 参数矩阵按 params_spec 顺序填充 (一线程一组参数,
`params_arr[tid * num_params + i]`)。回撤时间戳用 1m bar 的 `stime_arr`
(与 CPU `peak_eq_ts/valley_ts` 同口径)。

## 6. sweep / CLI 路径选择 (步骤 3)

`sweep()` 对每组参数选路:

| 策略 | device=cpu | device=gpu |
|---|---|---|
| **带 DSL** 的策略 (含 channel_deviation) | **DSL 特化内核** `run_one_dsl` | **通用 CUDA kernel** `cuda_sweep_window_generic` |
| 无 DSL (如 breakout) | 参考引擎 `run_one_general` (兜底) | 同左 |

CLI:

```bash
# dev_trigger: --grid 直接用 params_spec 里的参数名 (CLI 自动放行)
python -m evtrade sweep --strategy dev_trigger --synthetic-days 40 \
    --grid entry_dev=0.5,0.8 --out sweep_dev.csv
# backtest: DSL 策略走内核路径 (无 DSL 的策略会提示用 --engine ref)
python -m evtrade backtest --strategy dev_trigger --params "entry_dev:0.5" ...
```

## 7. 新增 DSL 策略三步法 (抄 `strategies/example_dev_trigger.py`)

1. 写 `strategies/my_strategy.py`: `@register_strategy("my_strategy")` +
   `params_spec` (声明顺序即 p0..pN 顺序) + **`state_spec`** (DSL 必填;
   schema `{"name": {"type": bool/int/float, "default": 标量}}`; 无持久状态时 `state_spec = {}`;
   缺声明编译期抛 `CompileError`, 见 12 号文档 2.5 节) +
   `compute_signal` docstring (DSL) + `check()` 两行样板 (`DSLCtx` + `dsl_check`, 见示例文件);
2. 在 `strategies/__init__.py` 加一行 `from . import my_strategy`;
3. 复制 `tests/test_strategy_template.py` 改类名, 再到
   `tests/test_kernel_dsl.py` 加一条 "kernel vs 参考引擎 bitwise" 测试
   (抄 `test_dev_trigger_kernel_vs_ref_engine_bitwise`)。

写完即可: 参考引擎 / `--engine kernel` / sweep (cpu+gpu) 三端全部生效。

## 8. 测试锁定

| 测试 | 锁什么 |
|---|---|
| `tests/test_kernel_dsl.py::test_dsl_spliced_channel_deviation_vs_ref_engine_bitwise` | DSL 渲染特化内核 ≡ Python ref 引擎 (信号轨迹, 含倍投; 2026-09 重构后冻结本尊 _strategy_check 已清空, 改与 Python ref 引擎对账) |
| `tests/test_kernel_dsl.py::test_dev_trigger_kernel_vs_ref_engine_bitwise` | 新策略三端同源: numba 内核 ≡ 参考引擎 |
| `tests/test_kernel_dsl.py::test_run_one_dsl_channel_deviation_matches_run_one` | 通用入口 ≡ 冻结 `run_one` |
| `tests/test_dsl_cuda.py` | 渲染产物结构 (ctx 剥离 / 状态映射 / 局部声明 / 溢出拒绝) |
| `tests/test_dsl_cuda.py::test_generic_cuda_matches_cpu_*` | 通用 CUDA kernel ≡ CPU DSL 内核 (全指标逐位, GPU 可用时) |
| `tests/test_dsl_cuda.py::test_sweep_gpu_generic_dev_trigger` | sweep GPU 通用路径 ≡ CPU 路径 |
| `tests/test_strategy_params.py::test_sweep_dev_trigger_uses_kernel_path` | sweep DSL 策略走内核路径且与 `run_one_dsl` 对账 |

## 9. 已知边界

- 持久状态字段由策略类 `state_spec` 声明 (第 3 节); 需要新状态字段 (如自有锁存/计数器) 时,
  只需在策略类 `state_spec` 加一行 (schema `{"name": {"type": bool/int/float, "default": 标量}}`),
  框架自动投影到 numba jitclass / CUDA device 函数 / Python ctx, **无需改内核** (见 12 号文档 2.5 节);
- breakout (滚动窗口型) 需要 O(N) 窗口状态, DSL 契约装不下, 保持参考引擎路径;
- CUDA device 函数签名的局部变量类型按字面推断 (纯整数字面量 → int, 其余 →
  double); 策略里不要用与内核字段同名的局部变量名;
- GPU 一律走通用 kernel (`cuda_sweep_window_generic`), 所有 DSL 策略 (含
  channel_deviation) 同路径; 旧 `cuda_sweep_window` 顶层 shim 已删除 (channel_deviation
  旧 API 的顶层 low1..high2 现在直接装入 params dict 即可)。回撤时间戳用 1m bar stime,
  与 CPU `kernel.step` 同口径。旧冻结模板 (`_CUDA_SOURCE`, 桶 ts 口径) 已移除。
