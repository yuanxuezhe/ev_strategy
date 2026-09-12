# evtrade-architecture

> **本文档为 evtrade 项目架构的行为权威 spec。**
> **详细中文说明见 `kbs/` 目录（特别是 02、06、11、12、14 号文档）；kbs/ 为本 spec 的中文投影，二者必须同步演进。改一处必改另一处。**
>
> 本文用 OpenSpec `spec-driven` 模式：每条 Requirement 描述可观测行为，Scenario 给定 WHEN/THEN。新增/修改需求走 `openspec/changes/` 下的 change proposal（proposal → specs → design → tasks），落地后 `archive`。

## Purpose

evtrade 是一个多周期 K 线 + 策略回测/扫参框架。本 spec 定义其行为契约：分层、数据流、`step`/`init_state` 策略契约、per-bar dict 契约、策略 hook 协议、CLI 参数协议。

---

## Requirements

### Requirement: Strategy interface contract

策略 MUST 实现 `VectorizedStrategy.step(self, state, bar: dict, params: dict) -> tuple[state, int]`
作为**唯一** framework 调用入口。`bar` 契约 MUST 为 dict
`{"ts", "o", "h", "l", "c", "v", "mark"}`（单桶 OHLCV + 策略期标记，全部标量）；
`mark == 0` 表示预热段，策略 `step` 在该段 MUST 返回 `(state, 0)`。
返回元组 MUST 长度 2：`(new_state, signal)`，`signal` ∈ {-1, 0, 1}（SELL / 无 / BUY）。

策略 MUST NOT 持有 instance-level 持久状态（不允许 `self._fsm / self._up_st / self._dw_st`
等 instance attr）——所有跨桶持久化 MUST 由 `state` 参数承载（推荐 `@dataclass`，dict 亦允许）。
策略 SHOULD 覆写 `VectorizedStrategy.init_state(self, params) -> state` 提供 state 初值
（基类默认返回 `None`，表示无状态策略）。

引擎层（`run_vectorized` 的 `_compute_signals` 与 `Engine.on_bars`）MUST 在循环外持有 state，
每桶一次调用 `strategy.step(state, bar, params)`，state 跨调用持续。策略 `step` MUST 仅做
"算法"（拿 `{ts,h,l,...}` + 上一步 state → 指标增量 → FSM → 返回 `(state, sig)`），MUST NOT 在
step 内出现批量循环（`for i in range(n)`）或后端数据迁移（`hasattr(x, "get").get()` 等）。

`params_spec` 类属性 MUST 声明所有策略参数（含 `default / type / min / max`），framework 在
`__init__` / `_resolve_params` 阶段做校验（填默认 + 类型转换 + 范围检查）；未在 `params_spec`
声明的参数名 MUST 报错（防拼写错误）。策略代码 MUST NOT `import numba` / `import cupy`。

#### Scenario: framework 不预计算指标
- **WHEN** 引擎（任一路径）调 `strategy.step(state, bar, params)`
- **THEN** 传入的 `bar` dict MUST 仅含 `ts/o/h/l/c/v/mark`；MUST NOT 含 `up` / `dw` /
  `low_dev` / `high_dev` 等任何具体指标键

#### Scenario: 策略 step 签名
- **WHEN** 一个策略类继承 `VectorizedStrategy` 并实现 `step(self, state, bar, params)`
- **THEN** framework MUST 按该签名调用；策略 MUST 返回 `(new_state, signal)` 且
  `signal ∈ {-1, 0, 1}`

#### Scenario: 策略无 instance state
- **WHEN** 静态扫描 `evtrade/strategies/*.py`（排除 `vectorized_base.py`）
- **THEN** MUST NOT 出现 `self._fsm` / `self._up_st` / `self._dw_st` / `self._has_prev` 等
  instance-level 持久状态字段；所有持久状态 MUST 由 `step` 的 state 参数承载

### Requirement: Strategy display hooks are framework-agnostic
`VectorizedStrategy` MUST 仅暴露一个可覆写展示 hook：
`format_signal_line(self, ts, sig, info=None) -> str`（hook 签名无 positional 指标；所需字段从
`info` dict 自取，默认实现仅打 `ts / sig / side`）。framework（`Engine` / `run_vectorized`
verbose 路径）SHALL NOT 假定任何指标字段名，仅 `print(strategy.format_signal_line(...))`。
`get_extra_bucket_columns` / `get_extra_signal_columns` 已删除（DSL/bundle 时代产物）。

#### Scenario: Engine 仅 print 策略返回值
- **WHEN** verbose 模式下引擎触发信号行打印
- **THEN** 引擎直接 `print(strategy.format_signal_line(ts, sig, info))`，不修改、不假设
  `info` 键集；默认实现仅显示 `ts / sig / side`

### Requirement: CLI params via `--params` validated by `params_spec`
CLI MUST 接受 `--params "k1:v1;k2:v2"` 一次传入策略参数；`_resolve_strategy_params` 按策略类
`params_spec` 填默认 + 类型/范围校验；未在 `params_spec` 中声明的参数名会报错（防拼写错误）。
策略参数（如 `tf1` / `low1` / `high2`）不再是独立 CLI flag，一律经 `--params` 传入；
`--device {cpu,gpu,auto}` 是唯一的后端选择参数。

策略类可声明**跨字段校验** `validators: list[Callable[[dict], None]]`（类属性，类级声明）。
校验时机：在 `_resolve_params` 完成单字段填默认 + 类型 + min/max 之后；
校验失败 MUST 抛 `ValueError`（带字段名与不通过原因）；校验成功 MUST 静默返回。
`validators` 的目的是表达 `low1 > low2` 这类**跨字段**硬约束（单字段 min/max 表达不了）；
未声明 `validators` 的策略与现状一致（仅单字段校验）。

#### Scenario: 未声明参数报错
- **WHEN** CLI 传入 `--params "low1:1.5;unknown_x:0.3"`
- **THEN** 启动时报 `ValueError: ChannelDeviationStrategy 收到未声明的参数 ['unknown_x']`
  （`vectorized_base.py::_resolve_params`）

#### Scenario: 跨字段 validator 拒绝违反锁存约束的参数组合
- **WHEN** CLI / sweep / `_defaults_loader` 任一入口传入 `low1=1.0, low2=1.5`（违反 `channel_deviation` 硬约束 `low1 > low2`）
- **THEN** 启动时 MUST 抛 `ValueError: channel_deviation: low1 (1.0) 必须 > low2 (1.5); 否则迟滞结构退化`
  （`ChannelDeviationStrategy._validate_latch_order` → `VectorizedStrategy._resolve_params`）
- **AND** sweep grid 中所有违反 `validators` 的组合 MUST 在 sweep 入口的预校验阶段被识别并跳过；
  跳过 MUST 打 warning（含 combo 字典 + 失败原因）；`sweep_results.csv` MUST NOT 含被跳过的 combo 行

### Requirement: Engine is the sole assembly point
Engine MUST 是唯一装配点：构造时把 `aggregator.on_bars` 覆写为自己的 `on_bars`。
Engine 消费任意 `stream() -> Iterator[Bar]` 的 bar 流对象（回测用
`core/_harness.ListBarFeed`；实盘接入时提供自己的流实现），切换回测/实盘 = 换 bar 流；
策略的撮合/账本由 `step` 内部自管（无 Executor / Account 介入）。`evtrade/feeds/` 子包已于 2026-09-10
删除（`MySQLBacktestFeed` 的 SQL 与 `core/data.py` 重复；`ChainedFeed` 无使用方）；
回测数据加载唯一入口 MUST 为 `core/data.load_bars`（MySQL 单次拉取 + npz 缓存）或
`core/data.synthetic_bars`（合成数据）。

#### Scenario: 切换回测/实盘仅换 bar 流
- **WHEN** 把 `ListBarFeed`（`evtrade/core/_harness.py`）换成自定义实盘 bar 流
  （`.stream() -> Iterator[Bar]` 鸭子类型实现）
- **THEN** Engine / Aggregator / 策略 / 数据加载代码均无需改动；策略的撮合数学
  在其 `step` 内部自管（无 Executor / Account 介入；参见 spec R "Strategy owns
  trade execution and PnL accounting"）

#### Scenario: feeds 子包已删除
- **WHEN** 静态检查仓库
- **THEN** `evtrade/feeds/` 目录 MUST 不存在；`evtrade/` 内 MUST NOT 有 `BrokerExecutor` /
  `ChainedFeed` / `get_feed` / `register_feed` 符号

### Requirement: Indicators are private to strategies (framework MUST NOT compute any indicator)

框架 MUST NOT 预计算任何指标。所有指标计算由策略在 `step(state, bar, params)` body 内通过
`evtrade/indicators/` 子包导出的 API 完成。`evtrade/indicators/` 目前 MUST 仅提供 **EMA**
一种指标、**三种形态**：

1. **step 增量版**（`@dataclass state` in/out，策略 `step` 逐桶调用）：
   `ema_step` / `ema_channel_step`（state：`EMAState` / `EMAChannelState`）
2. **numpy 批量参考版**（`ema` / `ema_channel`，reconcile 参考实现与测试用）
3. **torch 批量版**（`torch_ema`；GPU-batched sweep `batched_step` hook 专用 opt-in 形态，
   float64 与 numpy 参考版 bitwise 一致；用于 hook 内部按 period 分组调一次出整段
   `[T]` Tensor；不暴露为策略 `step` 的替代路径——`step` 仍走 `ema_step` / `ema_channel_step`）

`evtrade/indicators/` MUST NOT 包含除 EMA 之外的指标（`atr/rsi/boll` 已于
2026-09-10 删除；新增指标按上述三形态添加）。`evtrade/indicators/` MUST NOT `import cupy` /
`import numba`。`pyproject.toml` MUST NOT 声明 `cupy` / `numba` 为依赖，MUST 声明 `torch>=2.0`。

#### Scenario: indicators 提供 step 版 / numpy 参考版 / torch 批量版
- **WHEN** 策略用 `from evtrade.indicators import ema_step` 调用
- **THEN** `ema_step(state, value, p) -> (state, ema)` 为策略 step 用增量接口；
  `ema(values, p) -> ndarray` 为 numpy 批量参考接口；
  `torch_ema(values: Tensor[T], p) -> Tensor[T]` 为 GPU-batched sweep 批量形态
  （opt-in，batched_step hook 专用，不暴露给策略 `step`）

#### Scenario: torch_ema 与 numpy ema bitwise 一致
- **WHEN** 同一 `values: ndarray` 序列同 `p` 同时走 `torch_ema` 与 `ema`
- **THEN** `np.array_equal(torch_ema(Tensor(values), p).cpu().numpy(),
  ema(values, p)[p-1:])` MUST 为 True（首 `p-1` 个 step 形态返 0，numpy 返 NaN，对齐从 `p-1` 起）

#### Scenario: 策略 step 用 ema_step 维护增量
- **WHEN** 策略在 `step(state, bar, params)` 内维护 EMA 增量
- **THEN** MUST 用 `ema_step` / `ema_channel_step` 纯 Python 接口；与 numpy 批量参考版
  `ema` / `ema_channel` 的信号轨迹 MUST 在 reconcile `bucket_diff_cap` 内一致

#### Scenario: atr/rsi/boll 已删除
- **WHEN** 静态扫描 `evtrade/` 与 `tests/`
- **THEN** MUST NOT 出现 `atr_step` / `rsi_step` / `boll_step` / `sma_step` /
  `xp_ema` / `xp_ema_torch` 等已删符号的 import 或调用

### Requirement: Code hygiene — unused imports and non-underscore internal helpers MUST NOT accumulate

`evtrade/` 包内的源码文件 MUST NOT 留下未使用的 import 与未加下划线前缀的纯内部辅助函数。
"未使用"指该项目内部（含 `tests/`、`examples/`）无 import / 直接调用 / 字符串反射调用。
豁免类别：dev reload 工具（`*_cache` 类）、公共扩展 API（注册表 getter）、抽象基类的
`NotImplementedError` 占位、以及 `evtrade/__init__.py` 的向后兼容 shim 层——该 shim
MUST 收敛为最小集（`sys.modules.setdefault` 仅覆盖仓库内仍使用的旧 import 路径：
`evtrade.data` / `evtrade.metrics` / `evtrade.account` / `evtrade.aggregator` /
`evtrade.replay` / `evtrade.execution` / `evtrade.primitives` 共 7 项）；
`evtrade.kernel` ModuleType stub MUST NOT 存在（2026-09-10 删除）。

#### Scenario: 死 import 已清零
- **WHEN** 静态扫描 `evtrade/**/*.py`（排除 `__pycache__`）中的 import 语句
- **THEN** 所有非 `from __future__ import` / typing / 显式 re-export shim 的 import
  必须在文件内或项目内有引用方

#### Scenario: 内部辅助函数以下划线标注
- **WHEN** 一个公开名（非下划线前缀）的函数 / 类仅在定义文件内部被调用，且项目内无任何外部 caller
- **THEN** 该符号 SHOULD 重命名为下划线前缀，除非属于豁免类别

#### Scenario: shim 最小集
- **WHEN** 静态扫描 `evtrade/__init__.py`
- **THEN** `sys.modules.setdefault` 项 MUST ≤ 7 个且全部有仓库内 import 方；
  MUST NOT 存在 `types.ModuleType("evtrade.kernel")` 或等价 stub

---

### Requirement: Single contract = VectorizedStrategy.step

策略 MUST 是 `evtrade.strategies.VectorizedStrategy` 的子类（通过 `@register_strategy("name")` 注册），
MUST 实现 `step(self, state, bar, params)` 一个 framework 调用入口。策略 MUST NOT 拥有除
`step` / `init_state` / `format_signal_line` 之外的 framework 调用入口；MUST NOT 出现
`check(cur)` / `_strategy_check` / `state_spec` / `compute_signals` / `compute_signals_for_one_bar`
等已废形态。`params_spec` 类属性 MUST 声明所有策略参数（含 default / type / min / max），
framework 在 `__init__` / `_resolve_params` 阶段做校验（fill default + type cast + range check）。
策略代码 MUST NOT `import numba` / `import cupy`。同一份策略代码 MUST 在 `device="cpu"` 与
`device="gpu"` 下产生 bitwise 一致的信号序列（如有浮点舍入差异则在
`tests/test_strategy_unified.py::test_*_cpu_vs_gpu_*` 锁定）。

#### Scenario: 策略代码一份同时跑 CPU / GPU
- **WHEN** 同一策略分别在 `device="cpu"` 与 `device="gpu"` 下跑同一组数据
- **THEN** 返回的 `sig` 序列 MUST bitwise 一致（`np.array_equal(sig_cpu, sig_gpu)` 为 True）

#### Scenario: vectorized vs Engine 引擎 bitwise 一致
- **WHEN** 同一策略分别走 `run_vectorized`（桶级批量循环 step）与 `Engine.on_bars`（逐 bar
  循环 step）两条路径
- **THEN** 生成的 `trades` list 与终态 `cash / position` MUST 逐笔一致（同 ts / side / qty / price）

#### Scenario: 策略只写一处
- **WHEN** 静态扫描 `evtrade/strategies/*.py`
- **THEN** MUST NOT 出现 `def compute_signals(`、`def check(self,`、`state_spec`、
  `dsl_check`、`@njit`、`DSL docstring` 等已废形态

### Requirement: CLI surface = `--device {cpu, gpu, auto}`

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto}`
（默认 `auto`）。设备解析统一由 `evtrade.backends.resolve_device(requested, gpu_available)`
在三个子命令入口执行：`"gpu"` 且 CUDA 不可用 MUST 抛 `ValueError`（带可操作提示，
措辞包含"当前 torch 构建无 CUDA 支持 / 改用 --device auto 或 --device cpu / GPU 机器
请 `uv sync --extra gpu`"）；`"auto"` 优先 gpu，不可用时降级 cpu 并打 warning（不抛）。
`"cpu"` 直接返回。**`--device gpu` 报错的根本原因是 torch 包未安装 CUDA wheel；GPU
协作者 MUST 先 `uv sync --extra gpu`（或跑 `scripts/sync-torch-cu.sh`）再使用
`--device gpu`。** 引擎层（`run_vectorized` 等）MUST NOT 携带 device 参数——
后端为 torch 统一，实际计算路径设备无关（numpy 桶预计算 + 标量 step 循环）。旧
`--engine {kernel,ref,vectorized}` flag MUST NOT 存在（2026-09-10 删除，传入报 argparse
unknown option）。MUST NOT 存在"被接受但从不读取"的 CLI flag（`--no-sleep` /
`--step-days` / `--show-bars` / `--bars-out` 已删除）。

#### Scenario: --device gpu 无 CUDA 报错
- **WHEN** 未安装 GPU 版 torch 的机器执行 `python -m evtrade backtest --device gpu ...`
- **THEN** MUST 抛 `ValueError`，文案明确：当前 torch 构建无 CUDA 支持；提示改用
  `--device auto` / `--device cpu`；GPU 机器请先 `uv sync --extra gpu` 或
  `bash scripts/sync-torch-cu.sh`

#### Scenario: --device gpu 安装正确时跑通
- **WHEN** 已 `uv sync --extra gpu`（或跑过 sync helper）的机器执行
  `python -m evtrade backtest --device gpu ...`
- **THEN** MUST 跑通；`torch.cuda.is_available()` 为 True；与 `--device cpu` 路径
  产出 bitwise 一致

#### Scenario: --device auto 静默降级
- **WHEN** 未安装 GPU 版 torch 的机器执行 `python -m evtrade sweep --device auto ...`
- **THEN** 降级 cpu 并打 warning，正常运行完成

#### Scenario: --engine 已删除
- **WHEN** 执行 `python -m evtrade backtest --engine kernel ...`
- **THEN** argparse 报 `unrecognized arguments: --engine` 退出

#### Scenario: PyTorch 后端端到端跑通
- **WHEN** `python -m evtrade backtest --device auto --strategy channel_deviation ...`
- **THEN** MUST 跑通并打印 30 字段盈亏汇总（含 cagr / sharpe_excess / calmar / win_rate /
  profit_factor / baseline_max_dd / dd_excess / x_mdd 等）

### Requirement: PyTorch 统一后端

`evtrade` MUST 使用 PyTorch 作为唯一 array 后端能力来源（`pyproject.toml` 声明
`torch>=2.0`）；CPU 与 GPU 由 **同一 torch 包**的不同 wheel 提供。GPU wheel 的安装
MUST 走 **受支持的 optional extra**：`[project.optional-dependencies]` 中含 `gpu` extra，
GPU 协作者通过 `uv sync --extra gpu` 显式启用；未启用 `gpu` extra 时 MUST 拉 pypi.org
的 CPU-only wheel（与旧行为一致）。`evtrade.backends.gpu_available()` MUST 委托
`torch.cuda.is_available()`。当前热路径（桶预计算 + 策略 step）为设备无关 numpy/标量
实现，`backends.get_xp` 仅为需要 tensor 的扩展代码提供 `torch.device` 路由。
`evtrade/core/capability.py` MUST NOT 存在（能力探测收编至 `backends`）；
`evtrade/core/tsbucket.py` MUST 为纯 numpy 桶级 `ts/mark` 预计算（模块名/内容 MUST
NOT 含 gpu/cuda/torch 语义依赖，LRU 缓存行为不变）；历法运算（`encoded_to_epoch` /
`epoch_to_encoded`）MUST 单一真源于 `core/timeutils.py`（标量与向量两形态共享
Hinnant 整数日历）。`gpu_info` 已删除。

#### Scenario: cupy 不再是可选依赖
- **WHEN** 用户执行 `grep -r "import cupy" evtrade/`
- **THEN** MUST 0 命中（cupy 已完全下线）

#### Scenario: gpu 是受支持的 optional extra
- **WHEN** 用户执行 `grep -n "optional-dependencies" pyproject.toml`
- **THEN** MUST 命中 **恰好一个** `[project.optional-dependencies]` 段（含 `gpu`
  extra）；`gpu` extra 的依赖列表 MUST 含 `torch==2.9.0+cu128`（Blackwell sm_120
  支持的最低 cu128 系列）；不含 `cupy` / `numba`

#### Scenario: gpu 不再是独立安装路径
- **WHEN** 用户执行 `grep -n "optional-dependencies" pyproject.toml`
- **THEN** MUST 命中 **恰好一个** `[project.optional-dependencies]` 段；MUST NOT
  存在第三个 GPU 安装 extra（`cudnn` / `rocm` / `xpu` 等）；`--device` 是运行时参数
  （语义保留），不再是安装路径

#### Scenario: sync-torch-cu helper 在 uv sync 后恢复 cu128 wheel
- **WHEN** GPU 协作者首次 `uv sync`（CPU wheel 装上）后跑 `bash scripts/sync-torch-cu.sh`
- **THEN** 脚本 MUST 检测当前 torch 是 CPU 版，自动 `uv pip install --reinstall
  --index-strategy unsafe-best-match torch==2.9.0+cu128 --index-url
  https://download.pytorch.org/whl/cu128`；退出码 0；之后 `uv run` MUST 不再回退到
  CPU wheel；脚本 MUST 接受 `EVT_TORCH_CU_TAG` 环境变量覆写 cu tag（默认 `cu128`）

#### Scenario: tsbucket 纯 numpy
- **WHEN** 静态扫描 `evtrade/core/tsbucket.py`
- **THEN** MUST 无 `torch` / `cuda` / `cupy` 引用；`precompute_ts_mark` 输出与重构前
  bitwise 一致（`tests/test_tsbucket_cache.py` 锁定，仅 import 路径变更）

#### Scenario: torch 后端单测通过
- **WHEN** `pytest tests/test_torch_backend.py -v`
- **THEN** `get_xp` 返 `torch.device`；`gpu_available` 与 `torch.cuda.is_available()` bitwise 一致

### Requirement: CLI options declared once

**共享选项 MUST 仅声明一次**：`backtest` / `sweep` 三个子命令的共享选项（`--strategy --params --code
--start --end --period --device --synthetic-days --verbose --signals-out` 等）MUST 声明于单一共享父 parser
（argparse `parents=`），MUST NOT 各子命令重复声明同义选项。`backtest` 与 `sweep` 的输出 MUST NOT 包含 PnL /
收益 / 回撤 / 胜率 等 framework 业务概念——这些由策略自管。CLI 汇总打印仅输出信号轨迹（`format_signal_line`）
与策略 `final_state` 关键标量（由策略 `format_final_state` hook 自定义）。

#### Scenario: 共享选项单点声明

- **WHEN** 静态扫描 `evtrade/cli.py` 的 `add_argument` 调用
- **THEN** 上述共享选项名 MUST 仅出现于共享父 parser 定义处一次

#### Scenario: 汇总打印实际值

- **WHEN** `python -m evtrade backtest --strategy channel_deviation ...`
- **THEN** 输出 MUST NOT 含"期初资金/期初持仓/期末资金/期末持仓/期末持仓市值/策略总资产/不操作基线/盈亏比例/年化/CAGR/Sharpe/Calmar/最大回撤/胜率/成交额"等行；仅打印信号轨迹（verbose=True）+ 策略 `final_state` 透传（framework 不读具体字段；2026-09-13 后 framework 不再汇总 PnL/收益/回撤/胜率等业务概念）

#### Scenario: 共享选项无资金/撮合参数

- **WHEN** 静态扫描 `evtrade/cli.py` 的 `add_argument` 调用
- **THEN** `--init-cash --init-position --buy-pct --sell-pct --all-in --trade-qty --warmup-days --data-cache` MUST 0 命中（2026-09-13 删除；funding/撮合 由策略 state 自管）

### Requirement: 用户文档单一入口

用户文档 MUST 收敛为两条线：根 `README.md`（唯一入口：定位 / 快速上手 / 工作流
四步 / 命令速查 / 结构 / 指向 `kbs/`）+ `kbs/`（中文详述投影 + `使用说明.md`
操作手册）。仓库根 MUST NOT 存在 `docs/` 目录（原 `docs/quickstart.md` /
`docs/params-workflow.md` 内容已并入 `kbs/使用说明.md` 与 `kbs/10-配置参数与运行指南.md`）。
README 的"详见"清单 MUST 只指向 `kbs/` 内文档（不得指向仓库内不存在的文件）。

#### Scenario: docs/ 目录已删除
- **WHEN** 执行 `ls docs/`
- **THEN** MUST 报 "No such file or directory"

#### Scenario: README quickstart 命令现行
- **WHEN** 按 README「快速上手」的示例命令逐条执行（`python -m evtrade sweep ...` 等）
- **THEN** MUST 无 "unrecognized arguments" / "No such file" 错误

### Requirement: Sweep grid accepts only effective axes

sweep 的 `--grid` 网格轴 MUST 仅接受**策略 `params_spec` 声明的参数名**（引擎无业务参数轴，funding/撮合 由策略
params 承担）。MUST NOT 接受"下游从不读取"的轴。`--grid period=...` / `--grid code=...` /
`--grid buy_pct=...` 等引擎轴 MUST 报 `ValueError: 不支持的网格参数 '<key>'; 可用: [...]`。
backtest 子命令的 `--period` / `--code` flag 不受影响（CLI 入口，不入 grid）。

#### Scenario: 引擎轴不可网格化

- **WHEN** 执行 `python -m evtrade sweep --grid period=5m,15m --grid buy_pct=0.5,1.0 --grid trade_qty=10000 ...`
- **THEN** MUST 报 `ValueError`，提示不支持的网格参数及可用列表（来自策略 `params_spec`），MUST NOT 静默跑完

#### Scenario: 策略参数轴生效

- **WHEN** 执行 `python -m evtrade sweep --grid low1=0.05,0.1 --grid low2=0.5,1.0 ...`（命中 channel_deviation params_spec）
- **THEN** 正常展开笛卡尔积并逐组回测

#### Scenario: all_in 网格轴被拒

- **WHEN** 执行 `python -m evtrade sweep --grid all_in=true,false ...`
- **THEN** 报 `ValueError`（提示不支持的网格参数及可用列表），MUST NOT 静默跑完

#### Scenario: 资金模式经 buy_pct/sell_pct 网格化

- **WHEN** 执行 `python -m evtrade sweep --grid buy_pct=0.5,1.0 --grid sell_pct=0.5,1.0 ...`
- **THEN** 正常展开笛卡尔积并逐组回测（`all_in` 语义 = buy_pct=sell_pct=1.0 的组合）

### Requirement: signal trajectory output carries aligned timestamps

`--signals-out`（`backtest` 子命令）MUST 写出与信号**逐元素对齐**的桶级
`(ts, sig)` 行。`run_vectorized(...)` 返回的 `result["buckets"]["ts"][mark==1]`
MUST 与 `result["sig"]` 等长（与 `_aggregate_buckets` 产出的桶 ts 同源）；`backtest`
MUST 用此对子写出 CSV。MUST NOT 以 1m bar 数组数索引**桶级** sig 数组（桶数 < bar 数）——
2026-09-10 前 backtest / replay 两路径即以此方式索引（latent bug：warmup 存在时越界崩溃，
无 warmup 时写出大量 sig=0 错位行）。`replay` 子命令已于 2026-09-13 下线。

#### Scenario: run_vectorized 返回的 sig 与 ts 等长且对齐
- **WHEN** `run_vectorized(...)` 返回 `result`
- **THEN** `len(result["buckets"]["ts"][mark==1]) == len(result["sig"])`；每对 `(ts, sig)`
  中 ts 与 `_aggregate_buckets` 产出的桶 ts 同源

#### Scenario: --signals-out 写出桶级轨迹
- **WHEN** `python -m evtrade backtest --strategy <name> ... --signals-out <out.csv>`
- **THEN** 正常完成，`<out.csv>` 行数 = 桶级 `mark==1` 段的行数（非 1m bar 数），无越界异常

### Requirement: Optional GPU-batched sweep hook (`batched_step`)

`VectorizedStrategy` MUST 暴露一个可选 `@classmethod batched_step(cls, state, bars, params, *, n_combos, n_bars)` hook，作为 GPU 批量网格扫描的 opt-in 入口。基类默认实现 MUST raise `NotImplementedError`，表示策略未实现批量路径——`sweep()` 路由会自动 fallback 到 per-combo `ThreadPool` 路径。

实现该 hook 的策略应满足：
- `state` 为 dataclass，每个字段 MUST 为 `[N]` Tensor（`N = n_combos`），存储每 combo 的批量状态；
- `bars` MUST 为 dict[str, Tensor]，每个 value 形状 `[T]`（`T = n_bars`），键集 `{ts, o, h, l, c, v, mark}`；
- `params` MUST 为 dict[str, Tensor]，每个 value 形状 `[N]`，覆盖 `params_spec` 中声明的所有策略参数；
- 返回 `(new_state, sig)`：`sig` MUST 形状 `[N, T]` dtype `int8`，取值 ∈ {-1, 0, 1}；
- 浮点计算 MUST 用 `float64`，与 per-combo `step` 路径产出 bitwise 一致（或 float64 tolerance 1e-12 内）；
- `mark=0` 段处理 MUST 与原 `step` 语义一致（如 `ma_crossover` 仍推 EMA 累积、仅不产 sig）；
- 异常 MUST 立即向上抛（不延后到 sync point），保持 `test_sweep_does_not_crash_when_one_combo_fails` 语义。

`channel_deviation` MUST NOT 实现该 hook（FSM 锁存难向量化），`sweep()` 路由 MUST 自动跳过。

sweep 路由规则（`core/sweep.sweep()`）：
- `hasattr(cls, "batched_step") and cls.batched_step is not VectorizedStrategy.batched_step`（真覆写）；
- `device != "cpu"`；
- `len(combos) >= 32`（小网格 GPU 启动开销 > 收益）；
- `gpu_available()` 为 True。

任一条件不满足则走现有 per-combo ThreadPool 路径；`run_batched` 内捕获 `torch.cuda.OutOfMemoryError` 后自动 fallback 到 ThreadPool 并打 warning。`--workers` 在 batched 模式下 MUST 被忽略（不改 CLI 解析，打印 notice）。

#### Scenario: Hook default 未实现走 ThreadPool
- **WHEN** 策略类继承 `VectorizedStrategy` 但未覆写 `batched_step`
- **THEN** `hasattr(cls, "batched_step")` 为 True，调用 MUST raise `NotImplementedError`；`sweep()` 路由跳过 batched 路径，走现有 ThreadPool

#### Scenario: ma_crossover 真覆写 + 阈值满足走 batched
- **WHEN** `--strategy ma_crossover --device gpu --grid fast=3,5,10 --grid slow=20,60`（6 个 combos，< 32 个）OR `--device cpu`
- **THEN** `sweep()` 路由走 ThreadPool（device=cpu 路径；N=6 < 32 阈值）；ma_crossover 不实际跑 batched_step
- **AND WHEN** `--strategy ma_crossover --device gpu --grid fast=... --grid slow=...` 共 ≥ 32 个 combos 且 CUDA 可用
- **THEN** `sweep()` 路由走 batched 路径，调 `MACrossoverStrategy.batched_step` 一次产出 `[N, T]` sig

#### Scenario: batched_step 输出与 per-combo step 循环 bitwise 一致
- **WHEN** 同一 bars + 同一组 params 同时走 `MACrossoverStrategy.batched_step`（1 次调用）和 per-combo `MACrossoverStrategy.step` 循环
- **THEN** 产出的 sig 数组在 float64 tolerance 1e-12 内 bitwise 一致（`np.array_equal` 通过）；trades / summary 与 per-combo 路径走同一 `_execute_trades` 产出

#### Scenario: 无 CUDA / CUDA OOM / 小网格 fallback 到 ThreadPool
- **WHEN** 任一条件命中：`gpu_available() is False`、CUDA OOM at runtime、`len(combos) < 32`、`device == "cpu"`、策略未覆写 hook
- **THEN** sweep MUST 走 ThreadPool 路径，CSV 输出与纯 ThreadPool 路径 bitwise 一致；OOM 情况打印 fallback warning

### Requirement: Filtered mean-reversion strategy (`filtered_mr`)

`FilteredMRStrategy` 是 `VectorizedStrategy` 子类，实现 4 重过滤以避免单边暴涨暴跌中均值回归"接飞刀"：

1. **大周期顺势过滤**：用 `--higher-period`（如 1h）桶的 `EMA(higher_ema_period)` 判定方向；
   大周期多头时关闭上轨做空信号，大周期空头时关闭下轨做多信号。
2. **ADX 趋势强度过滤**：本周期 `ADX(adx_period)` > `adx_threshold`（默认 30）时暂停所有回归开仓。
3. **ATR 波动率过滤**：`ATR(atr_period)` > `atr_vol_mult * MA(ATR, atr_ma_period)`（默认 1.5×）时暂停所有回归开仓。
4. **Close 确认 FSM**：上一桶影线触轨 + 本桶 `close` 回到带内才出信号，避免盘中探底/冲高即开仓。

Strategy MUST 实现 `step(self, state, bar, params) -> (state, int)`，`state` MUST 为 `@dataclass`（含大周期桶跟踪字段、EMA/ADX/ATR 增量状态、触轨 FSM 标志）；`mark=0` 段 MUST 仍累积 EMA/ADX/ATR 指标，仅不产信号。`filtered_mr` MUST NOT 实现 `batched_step`（FSM 难向量化），sweep 自动走 ThreadPool 路径。

#### Scenario: filtered_mr 4 重过滤生效
- **WHEN** `python -m evtrade backtest --strategy filtered_mr ...`
- **THEN** 强趋势段（ADX > 阈值）/ 高波动段（ATR > mult × MA）/ 逆大周期段 MUST 不产信号；正常震荡段 MUST 产 close 确认后的回归信号

### Requirement: Framework has no funding or PnL concept

framework 的所有接口 MUST NOT 包含资金 / 持仓 / 撮合 / PnL / 收益 / 风险 / 绩效 等业务概念：

- `VectorizedStrategy.step(state, bar, params) -> (state, int)` 接口签名不变；`bar` dict 仅含
  `{ts, o, h, l, c, v, mark}`；`buckets` dict 仅含 `{ts, o, h, l, c, v, mark, n_bars}`。
- `run_vectorized(...)` 返回 = `{sig, buckets, final_state}`（`final_state` = 策略 step 末尾
  state 透传，**MUST NOT** 含 framework 业务字段键如 `cash / position / equity / final_cash /
  final_position / final_equity / n_trades / n_buy / n_sell / turnover / cagr / sharpe / drawdown /
  win_rate / profit_factor / 等` —— 这些键由策略自己维护在 dataclass state field，framework 不读不写）。
- framework MUST NOT 提供 `format_final_state` / `sweep_export_columns` 等"摘要导出 hook"；
  策略若要看 PnL / 收益 / 绩效，自己加 state 字段 + 自己写打印/导出代码。
- `evtrade/execution/` 子包 MUST NOT 存在；`Account` / `Executor` / `SimulatedExecutor` /
  `trade_decision` / `apply(side, qty, price, ts)` MUST NOT 出现在 framework 代码。
- `evtrade/core/metrics.py` MUST NOT 存在；`summarize` / `metrics.*` MUST NOT 出现在 framework 代码。
- `evtrade/core/permutation.py` MUST NOT 存在（依赖 metrics）。
- `evtrade/core/replay.py` MUST NOT 存在（依赖 Account + metrics）。
- `evtrade/core/config.py` MUST NOT 含 `INIT_CASH / INIT_POSITION / TRADE_QTY` 等常量
  （CLI 不再传；策略 init_state 自定）。

#### Scenario: framework grep hygiene

- **WHEN** 执行 `grep -rn "trade_decision\|Account\|Executor\|SimulatedExecutor\|cash\|position\|equity\|cagr\|sharpe\|drawdown\|win_rate\|profit_factor\|sortino\|calmar\|metrics\|max_dd\|turnover\|baseline\|format_final_state\|sweep_export_columns" evtrade/core/ evtrade/cli.py evtrade/__init__.py evtrade/strategies/vectorized_base.py`
- **THEN** MUST 0 命中（仅桶聚合、step 驱动、numpy/tsbucket 路由相关代码；funding/PnL/绩效 概念全部下放到具体策略文件内）

#### Scenario: execution 子包已删除

- **WHEN** 静态检查仓库
- **THEN** `evtrade/execution/` 目录 MUST 不存在；`from evtrade.execution.* import` MUST ImportError

#### Scenario: run_vectorized 返回无业务字段

- **WHEN** `run_vectorized(...)` 返回 dict
- **THEN** MUST 仅含 `sig / buckets / final_state` 三个 key；`final_state` 是策略 step 末尾
  state（dataclass 或 dict），framework 仅透传其内容，不识别具体字段

### Requirement: Strategy owns trade execution and PnL accounting

策略 MUST 自负责以下业务概念（在 `step(state, bar, params)` 内部 / state 字段内 / `init_state` 内）：

- **资金 / 持仓状态**：state 字段（推荐 `@dataclass`，含 `cash / position / init_cash / init_position`
  等），或外部 dict
- **撮合数学**（buy/sell 决策 + 现金/持仓约束 + 成交价 = `bar["c"]`）：策略 `step` 内部 inline 实现
  （`trade_decision` 旧实现内联）；3 个策略各写一份，**不抽取共享**（避免 framework 隐藏业务概念）
- **记账**（trade ledger）：state 字段 `trades: list[dict]`
- **PnL 与风险**（equity curve / cagr / sharpe / drawdown / win_rate / 等）：**可选**——
  策略需要时自行加 state 字段与计算（旧 `metrics.summarize` 30 字段公式内联到策略或策略的
  helper 文件中）；framework **不强制**、**不提供**、**不识别**

#### Scenario: 策略 step state 含 cash/position/trades 字段

- **WHEN** 静态扫描 `evtrade/strategies/*.py`（排除 `vectorized_base.py`）
- **THEN** 至少一个策略的 `@dataclass` state MUST 含 `cash / position / trades` 字段（策略自管资金/持仓/账本）

#### Scenario: 策略 step 内自写撮合数学

- **WHEN** 静态扫描 `evtrade/strategies/*.py`
- **THEN** MUST 出现 `cash / price`、`buy_pct *`、`sell_pct *`、`min(cur_qty, max_by_cash)` 等
  撮合数学表达式（内联在 step 内或策略内部 helper，**不**走 framework `from evtrade.execution.* import`）

#### Scenario: 策略 PnL 字段可选

- **WHEN** 静态扫描 `evtrade/strategies/*.py`
- **THEN** PnL / 收益 / 绩效相关字段（`equity_curve / cagr / sharpe / drawdown / win_rate /
  profit_factor / ...`）**MUST NOT** 出现在 framework 代码（`evtrade/core/` / `evtrade/cli.py` /
  `evtrade/strategies/vectorized_base.py`）；允许出现在具体策略文件内（由策略自维护）

### Requirement: Engine drives step only

framework 唯一职责 = 桶预计算 + `step` 驱动循环 + 累计 sig/state。

- `VectorizedEngine.run_vectorized` MUST 仅做：
  1. 桶聚合（numpy 向量化；调用 `core/tsbucket.py::precompute_ts_mark`）
  2. 循环调 `strategy.step(state, bar, params)`（`_compute_signals`），state 由 framework 持有
  3. 返回 `{sig, buckets, final_state}`（`final_state` = 策略 step 最终 state 透传）
- `Engine.on_bars` MUST 仅做逐 bar 循环调 `strategy.step` + 累计 sig/state；不持 cash/position/
  Account/Executor；不调 metrics
- framework MUST NOT 调 `trade_decision` / `Account.apply` / `Executor.trade` 等业务接口
  （这些在策略侧 step 内完成）

#### Scenario: vectorized_engine 不含业务概念

- **WHEN** 静态扫描 `evtrade/core/vectorized_engine.py`
- **THEN** MUST NOT 出现 `trade_decision` / `Account` / `cash` / `position` / `equity` /
  `turnover` / `metrics` / `summarize` 等业务概念符号

#### Scenario: engine.py 不含业务概念

- **WHEN** 静态扫描 `evtrade/core/engine.py`
- **THEN** MUST NOT 出现 `Account` / `Executor` / `SimulatedExecutor` / `trade_decision` /
  `cash` / `position` / `equity` / `apply(` 等业务概念符号

### Requirement: Sweep runs strategy over a parameter grid

`sweep --grid` MUST 仅扫描**策略 `params_spec` 声明的参数名**；引擎无业务参数轴（funding/撮合 由策略
params 承担）。每组参数组合 MUST 调一次 `run_vectorized` 跑完桶，返回 `final_state`；CSV 写策略
`final_state`（由 sweep 序列化 dataclass/dict 字段）。

#### Scenario: sweep 跑策略参数组合

- **WHEN** 执行 `python -m evtrade sweep --strategy channel_deviation --grid low1=0.05,0.1 --grid low2=0.5,1.0 ...`
- **THEN** MUST 展开 4 组笛卡尔积，每组跑一次 `run_vectorized`，CSV 每行 = 策略 params + final_state 字段

### Requirement: CLI is step-driver surface

CLI = `python -m evtrade {backtest, sweep, params}` 三个子命令。

- `backtest`：仅 `--strategy / --params / --period / --code / --start / --end / --synthetic-days /
  --device / --verbose / --signals-out`。**删除** `--init-cash / --init-position / --buy-pct /
  --sell-pct / --all-in / --trade-qty / --warmup-days / --data-cache`（2026-09-13）
- `sweep`：保留 `--grid / --split / --splits / --fee-bp / --score-lambda / --min-trades / --max-mdd /
  --mc / --mc-top / --workers / --out / --top / --save-defaults`（funding 相关参数由策略 params 承担）
- `replay` 子命令**删除**（framework 不再有逐笔对账口径）
- CLI 汇总打印 MUST NOT 包含 PnL / 收益 / 回撤 / 胜率 / 资金 / 持仓 / 期末市值 / 成交额 / 等 framework
  业务概念；`backtest` 输出仅打印信号轨迹（`format_signal_line`）。策略若要看自己 state，由策略内部
  提供打印 hook（由策略代码决定，非 framework 职责）

#### Scenario: 旧资金/撮合 flag 已删除

- **WHEN** 执行 `python -m evtrade backtest --init-cash 100000 ...` 或 `--buy-pct 1.0 ...` 等
- **THEN** argparse 报 `unrecognized arguments: --init-cash` / `--buy-pct` 退出（2026-09-13 删除）

#### Scenario: replay 子命令已删除

- **WHEN** 执行 `python -m evtrade replay ...`
- **THEN** argparse 报 `error: argument {backtest,sweep,params}: invalid choice: 'replay'` 退出

#### Scenario: backtest 仅打印信号轨迹

- **WHEN** `python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device cpu`
- **THEN** 输出 MUST NOT 含 `盈亏比例 / CAGR / Sharpe / Calmar / 最大回撤 / 胜率 / 期末资金 / 期末持仓 /
  期末持仓市值 / 策略总资产 / 不操作基线 / 成交额` 等行；仅打印信号轨迹（`format_signal_line`）

### Requirement: replay & permutation removed

`evtrade/core/replay.py` 与 `evtrade/core/permutation.py` MUST NOT 存在（2026-09-13 删除）。framework
不再提供逐笔对账与置换检验口径。CLI 无 `replay` 子命令；`--against-ref` 不存在；`--mc` flag
MUST NOT 触发置换检验（保留参数解析但调 `NotImplementedError`，明示 framework 不再支持）。

#### Scenario: replay / permutation 模块已删除

- **WHEN** 静态检查仓库
- **THEN** `evtrade/core/replay.py` / `evtrade/core/permutation.py` MUST 不存在；`from
  evtrade.core.replay import *` / `from evtrade.core.permutation import *` MUST ImportError

### Requirement: Market data DB connection has a sane default

`evtrade.core.data.load_bars` 在未指定 `db_url` 参数时 MUST 解析到一个对当前协作组开箱即用的 MySQL 连接串默认值；默认主机 = `192.168.10.2:33066`、库 = `evtrade`、用户 = `EvTrade`。密码中 `@` MUST URL-encode 为 `%40`。`TABLE` 默认名 MUST 仍为 `minute_bars`。这两个常量是 framework 行为契约的一部分，被 `load_bars` / `_fetch` / CLI `backtest` / `sweep` 间接依赖；`EVTRADE_DB_URL` 与 `EVTRADE_TABLE` 环境变量 MUST 优先于默认值（分别覆写连接串与表名），作为"临时切库 / 离线 / 测"的逃生口。

#### Scenario: 默认 DB_URL 指向 192.168.10.2
- **WHEN** 协作者在不设 `EVTRADE_DB_URL` 的情况下跑 `python -m evtrade backtest --strategy filtered_mr --code 159992.SZ --start 20260101 --end 20260903 --device gpu`
- **THEN** `load_bars` MUST 尝试连 `mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4`（不是 `127.0.0.1:3306`），即默认主机 = `192.168.10.2`、端口 = `33066`、库 = `evtrade`

#### Scenario: EVTRADE_DB_URL 优先于默认
- **WHEN** 协作者 `export EVTRADE_DB_URL=mysql+pymysql://other:other@10.0.0.1:3306/other?charset=utf8mb4` 后再跑 `backtest`
- **THEN** `load_bars` MUST 尝试连 `10.0.0.1:3306/other`，不连 `192.168.10.2`；`EVTRADE_DB_URL` 必须解析到最终 SQLAlchemy `create_engine` 调用

#### Scenario: EVTRADE_TABLE 覆写表名
- **WHEN** 协作者 `export EVTRADE_TABLE=other_table` 后跑 `backtest`
- **THEN** `_fetch` 生成的 SQL MUST FROM `other_table`，不是 `minute_bars`；`EVTRADE_TABLE` 默认值 MUST 为 `minute_bars`（与 spec 一致）

#### Scenario: 默认 DB_URL 中 `@` 已 URL-encode
- **WHEN** 静态扫描 `evtrade/core/data.py` 中的 `DB_URL` 默认值
- **THEN** MUST 含 `p%40ssw0rd`（而非裸 `p@ssw0rd`）；确保 SQLAlchemy / pymysql 不会把 `@` 当 host 分割符

#### Scenario: DB 不可达时 CLI 输出默认主机提示
- **WHEN** 协作者在不设 `EVTRADE_DB_URL`、默认 DB 也无法连接的环境下跑 `backtest`
- **THEN** CLI MUST 在 SQLAlchemy 抛 `OperationalError` 后打印包含 `192.168.10.2:33066` 与 `EVTRADE_DB_URL` 的中文提示行，并以非零退出码退出；非 `OperationalError` MUST 不被该分支吞掉

### Requirement: Strategy signal line shows trigger bar context

策略（至少 `filtered_mr` 与 `channel_deviation`）的信号行 hook MUST 在 `sig != 0` 时把
触发该信号的方向、成交价与触发 K 线 OHLCV 全部展示在一行内，方便协作现场直接读出
"在什么价位的什么 K 线上触发了 BUY/SELL"，不再需要拿 `ts` 反查行情。具体契约：

- `step` 入口处 MUST 写 `self._last_info = {"side": "", "price": float, "o": float,
  "h": float, "l": float, "c": float, "v": float|int}`，字段值取自当前 step 接收的 finalized
  bar（确保任何 early-return 路径都让 framework 拿到最新 bar 的字段）；`side` 在 step 末尾
  覆写为 `"BUY"` / `"SELL"` / `""` 之一
- `format_signal_line` hook MUST 用 `info.get(...)` 而非 `info[...]`，防止 info 缺字段时
  KeyError
- 当 `sig != 0` 时 hook 返回字符串 MUST 包含方向词（`BUY` 或 `SELL`）、价格数字、
  以及 4 个 OHLC 字段名（`o=` / `h=` / `l=` / `c=`）；否则视为该策略违反本契约
- `VectorizedStrategy` 基类默认 hook 不变（仅 `ts / sig / side`），不强制所有策略实现本契约

framework 行为（`Engine._process_bucket` / `run_vectorized` 的 `info` 透传机制）不变，
本条 Requirement 只约束策略层。

#### Scenario: filtered_mr 信号行包含方向/价格/OHLCV
- **WHEN** `filtered_mr` 在某个 finalized bar 上触发 `sig=+1`（BUY）
- **THEN** `format_signal_line(ts, sig, info)` 返回字符串 MUST 同时含 `BUY`、价格字段
  （与该 bar 的 `close` 一致）、`o=`、`h=`、`l=`、`c=` 四个 OHLC 字段名；不含这些的视为缺漏

#### Scenario: channel_deviation 信号行扩展后含触发 K 线
- **WHEN** `channel_deviation` 在某个 finalized bar 上触发 `sig=-1`（SELL）
- **THEN** `format_signal_line(ts, sig, info)` 返回字符串 MUST 含 `SELL`、价格、4 个 OHLC
  字段名；原有的 `up` / `dw` / `low_dev` / `high_dev` 字段 MUST 仍然出现（向后兼容）

#### Scenario: hook 用 info.get 不抛 KeyError
- **WHEN** 协作者直接调用 `strategy.format_signal_line(ts, sig, info={})`（info 为空 dict）
- **THEN** hook MUST NOT 抛 `KeyError`；MUST 返回包含 `ts` / `sig` / `BUY`-or-`SELL`-or-空
  的字符串（OHLCV 字段可显示为空或缺失标记，不报错）

#### Scenario: _last_info 字段集与 step 触发 bar 一致
- **WHEN** 协作者断言 `_last_info["c"]` 必须等于触发该 sig 的 bar 的 `close`
- **THEN** MUST 等值（MUST NOT 是上一桶或下一桶的 close）；同理 `o` / `h` / `l` / `v`
  与触发 bar 的同名字段一一对应

## 与 `kbs/` 的对应关系

| 本 spec 节 | `kbs/` 详述 |
|---|---|
| Strategy interface contract | kbs/14 §1, kbs/11 §3 |
| Strategy display hooks | kbs/06 §9, kbs/11 §7, kbs/02 §1 |
| Single contract = VectorizedStrategy.step | kbs/14 §1, kbs/06, kbs/09 |
| CLI params via `--params` | kbs/06 §3, kbs/10, kbs/14 §1 |
| Engine is the sole assembly point | kbs/02 §1-4 |
| Indicators are private to strategies | kbs/05, kbs/11 §5, kbs/12, kbs/14 §2 |
| metrics.summary covers full field shape | kbs/13, kbs/09 |
| Code hygiene (no unused imports, internal helpers underscored) | kbs/01 (源码地图: 本次清理 + 改名) |
| PyTorch 统一后端 (`gpu` extra 是受支持路径) | kbs/15 §7.1, kbs/10, 使用说明 §0 |
| Market data DB connection has a sane default | kbs/08 §1.3 |
| Strategy signal line shows trigger bar context | kbs/06 (策略信号行打印段) |
| 用户文档单一入口 (无 docs/ 目录) | kbs/README, kbs/使用说明 |
| Sweep grid accepts only effective axes (无空转轴) | kbs/10 (sweep 参数), kbs/13 |
| reconcile legs receive identical effective funding | kbs/09, kbs/10 (replay) |
| signal trajectory output carries aligned timestamps | kbs/10 (replay --signals-out) |
| Account is a pure trade ledger (无 equity 估值方法) | kbs/07, kbs/03, kbs/09 |
| Optional GPU-batched sweep hook (`batched_step`) | kbs/15 §7.4, kbs/12 perf note, kbs/13 §3, kbs/10 --device |
| Filtered mean-reversion strategy (`filtered_mr`) | kbs/06 §10, kbs/12 perf note |

修改本 spec 时**必须**同步更新对应 `kbs/` 文档（反之亦然），并在 commit message 中标注。
