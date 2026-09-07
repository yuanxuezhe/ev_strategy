# 14 · 策略 DSL 与三端转译 (2026-09-06)

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
| CUDA | `render_cuda_device_function` (ctx.X → 内核名, `__device__` 函数) | `gpu.cuda_sweep_window_generic` |

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
但 channel_deviation 保持 **signal 变量形式** (不提前 return), 与冻结版
`frozen/strategy.py` 逐位对齐 —— 它的信号赋值后还要执行本桶的锁存段。

## 3. ctx 字段契约 (`dsl._CTX_TO_KERNEL`, 单一事实源)

| DSL 写法 | numba 内核 | CUDA device 函数 | 含义 |
|---|---|---|---|
| `ctx.p0` .. `ctx.p15` | `st.p0`..`st.p15` | `p0`..`p7` (寄存器) | 策略参数, **按 params_spec 声明顺序** |
| `ctx.cur_ts` | `st.cur_ts` | `cur_ts` | 当前桶时间戳 |
| `ctx.cur_open/high/low/close/volume` | `st.cur_*` | 同名 | 当前桶运行中 OHLCV |
| `ctx.up` / `ctx.dw` | 裸名 `up` / `dw` (函数参数) | 同名 | 通道上/下轨 (NaN=未就绪) |
| `ctx.low_hit` / `ctx.high_hit` | `st.low_hit` / `st.high_hit` | 同名 | 极端偏离锁存 |
| `ctx._low_acted` / `ctx._high_acted` | `st.low_acted` / `st.high_acted` | 同名 | 本桶已操作锁 |
| `ctx._bucket_ts` | `st.lock_ts` | `lock_ts` | 桶切换检测 (≠cur_ts 时重置 *_acted) |
| 其余 `ctx.x` / 裸名 `x` | 局部变量 | 局部变量 (自动声明) | 桶内临时值, 跨桶不保留 |

CUDA 端参数上限 8 个 (`p0..p7`), numba 端 16 个 (`p0..p15`); 超限在编译/调用期报错。
**非法引用** (如 CUDA 端用 `ctx.p8`) 在渲染期即被拒绝。

## 4. numba 端: kernel_dsl 按策略特化内核

机制 —— **一份内核源, N 个策略特化**:

1. `kernel.py` 的 `_strategy_check` 函数体位于
   `# ==== DSL-STRATEGY-BEGIN/END ====` 标记之间, 是 channel_deviation DSL 的
   手写 st 形式 (与 `frozen/strategy.py` 逐位锁定, 72 项差分测试不受影响);
2. `core/kernel_dsl.build_dsl_kernel(name)` 读 kernel.py 源码, 把标记之间的整段
   函数体替换成该策略 DSL 渲染出的同形代码, `exec` 出一个**独立内核模块**
   (jitclass / step / run_backtest / summarize 全套; numba 惰性编译, 进程内缓存);
3. `dsl_kernel(name)`: channel_deviation 直接返回冻结 kernel 本尊 (零额外编译);
   其他 DSL 策略返回特化模块。

上层 API:

- `make_state_general(name, period, warmup_until, tf1, ..., strategy_params)`:
  按 params_spec 顺序把参数填进 `p0..pN` 构造状态;
- `run_one_dsl(bars, ..., strategy_name, strategy_params)`: 单窗回测,
  指标口径 = `kernel.summarize`;
- `strategy_has_dsl(name)`: 是否有可渲染的 DSL (决定 sweep 走哪条路)。

## 5. CUDA 端: 通用 kernel + device 函数

`gpu.py::_CUDA_SOURCE_GENERIC_TEMPLATE` = 与旧 `sweep_kernel` 逐行等价的
桶合并/EMA/成交/绩效段 + `{STRATEGY_BODY}` 占位符 (编译期注入
`render_cuda_device_function` 生成的 `__device__ __forceinline__ int
strategy_check(...)`)。

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
| channel_deviation | 冻结内核 `_run_window` | 旧 CUDA kernel |
| 其他 **带 DSL** 的策略 | **DSL 特化内核** `run_one_dsl` | **通用 CUDA kernel** |
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
   `params_spec` (声明顺序即 p0..pN 顺序) + `compute_signal` docstring (DSL)
   + `check()` 两行样板 (`DSLCtx` + `dsl_check`, 见示例文件);
2. 在 `strategies/__init__.py` 加一行 `from . import my_strategy`;
3. 复制 `tests/test_strategy_template.py` 改类名, 再到
   `tests/test_kernel_dsl.py` 加一条 "kernel vs 参考引擎 bitwise" 测试
   (抄 `test_dev_trigger_kernel_vs_ref_engine_bitwise`)。

写完即可: 参考引擎 / `--engine kernel` / sweep (cpu+gpu) 三端全部生效。

## 8. 测试锁定

| 测试 | 锁什么 |
|---|---|
| `tests/test_kernel_dsl.py::test_dsl_spliced_channel_deviation_bitwise` | DSL 渲染特化内核 ≡ 冻结 kernel (信号轨迹 + 绩效, 含倍投) |
| `tests/test_kernel_dsl.py::test_dev_trigger_kernel_vs_ref_engine_bitwise` | 新策略三端同源: numba 内核 ≡ 参考引擎 |
| `tests/test_kernel_dsl.py::test_run_one_dsl_channel_deviation_matches_run_one` | 通用入口 ≡ 冻结 `run_one` |
| `tests/test_dsl_cuda.py` | 渲染产物结构 (ctx 剥离 / 状态映射 / 局部声明 / 溢出拒绝) |
| `tests/test_dsl_cuda.py::test_generic_cuda_matches_cpu_*` | 通用 CUDA kernel ≡ CPU DSL 内核 (全指标逐位, GPU 可用时) |
| `tests/test_dsl_cuda.py::test_sweep_gpu_generic_dev_trigger` | sweep GPU 通用路径 ≡ CPU 路径 |
| `tests/test_strategy_params.py::test_sweep_dev_trigger_uses_kernel_path` | sweep DSL 策略走内核路径且与 `run_one_dsl` 对账 |

## 9. 已知边界

- DSL 状态契约固定 (第 3 节的锁存字段); 需要新状态字段 (如自有锁存/计数器) 时
  要同步扩 `KernelState._STATE_SPEC` + CUDA 模板状态块 + `_CTX_TO_KERNEL`
  —— 属于内核改造, 参照 12 号文档的同步清单;
- breakout (滚动窗口型) 需要 O(N) 窗口状态, DSL 契约装不下, 保持参考引擎路径;
- CUDA device 函数签名的局部变量类型按字面推断 (纯整数字面量 → int, 其余 →
  double); 策略里不要用与内核字段同名的局部变量名;
- GPU 一律走通用 kernel (`cuda_sweep_window_generic`), 所有 DSL 策略 (含
  channel_deviation) 同路径; `cuda_sweep_window` 保留为向后兼容 shim (顶层
  low1..high2 归一化到 params dict 后委托 generic)。回撤时间戳用 1m bar stime,
  与 CPU `kernel.step` 同口径。旧冻结模板 (`_CUDA_SOURCE`, 桶 ts 口径) 已移除。
