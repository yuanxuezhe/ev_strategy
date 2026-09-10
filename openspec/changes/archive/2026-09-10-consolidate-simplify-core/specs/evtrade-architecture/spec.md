# evtrade-architecture (delta)

> 本 spec 是 2026-09-10 `consolidate-simplify-core` change 的 delta；落地后合并到
> `openspec/specs/evtrade-architecture/spec.md`。

## Modified Requirements

### Requirement: Indicators are private to strategies (framework MUST NOT compute any indicator)

框架 MUST NOT 预计算任何指标。所有指标计算由策略在 `step(state, bar, params)` body 内通过
`evtrade/indicators/` 子包导出的 API 完成。`evtrade/indicators/` 目前 MUST 仅提供 **EMA**
一种指标、两种形态：

1. **step 增量版**（`@dataclass state` in/out，策略 `step` 逐桶调用）：
   `ema_step` / `ema_channel_step`（state：`EMAState` / `EMAChannelState`）
2. **numpy 批量参考版**（`ema` / `ema_channel`，reconcile 参考实现与测试用）

`evtrade/indicators/` MUST NOT 包含 xp/torch 变体或 EMA 之外的指标（`atr/rsi/boll` 已于
2026-09-10 删除；新增指标按上述两形态添加）。`evtrade/indicators/` MUST NOT `import cupy` /
`import numba`。`pyproject.toml` MUST NOT 声明 `cupy` / `numba` 为依赖，MUST 声明 `torch>=2.0`。

#### Scenario: indicators 提供 step 版与 numpy 参考版
- **WHEN** 策略用 `from evtrade.indicators import ema_step` 调用
- **THEN** `ema_step(state, value, p) -> (state, ema)` 为策略 step 用增量接口；
  `ema(values, p) -> ndarray` 为 numpy 批量参考接口

#### Scenario: 策略 step 用 ema_step 维护增量
- **WHEN** 策略在 `step(state, bar, params)` 内维护 EMA 增量
- **THEN** MUST 用 `ema_step` / `ema_channel_step` 纯 Python 接口；与 numpy 批量参考版
  `ema` / `ema_channel` 的信号轨迹 MUST 在 reconcile `bucket_diff_cap` 内一致

#### Scenario: atr/rsi/boll 已删除
- **WHEN** 静态扫描 `evtrade/` 与 `tests/`
- **THEN** MUST NOT 出现 `atr_step` / `rsi_step` / `boll_step` / `sma_step` /
  `xp_ema` / `xp_ema_torch` 等已删符号的 import 或调用

### Requirement: Engine is the sole assembly point

Engine MUST 是唯一装配点：构造时把 `aggregator.on_bars` 覆写为自己的 `on_bars`。
Engine 消费任意 `stream() -> Iterator[Bar]` 的 bar 流对象（回测用
`core/_harness.ListBarFeed`；实盘接入时提供自己的流实现），切换回测/实盘 = 换 bar 流 +
换 Executor，Engine / Aggregator / 策略 / Account 不变。`evtrade/feeds/` 子包已于 2026-09-10
删除（`MySQLBacktestFeed` 的 SQL 与 `core/data.py` 重复；`ChainedFeed` 无使用方）；
回测数据加载唯一入口 MUST 为 `core/data.load_bars`（MySQL 单次拉取 + npz 缓存）或
`core/data.synthetic_bars`（合成数据）。

#### Scenario: 切换回测/实盘仅换 bar 流 + Executor
- **WHEN** 把 `ListBarFeed` 换成自定义实盘 bar 流，把 `SimulatedExecutor` 换成自定义 Executor
- **THEN** Engine / Aggregator / 策略 / Account 无需改动

#### Scenario: feeds 子包已删除
- **WHEN** 静态检查仓库
- **THEN** `evtrade/feeds/` 目录 MUST 不存在；`evtrade/` 内 MUST NOT 有 `BrokerExecutor` /
  `ChainedFeed` / `get_feed` / `register_feed` 符号

### Requirement: CLI surface = `--device {cpu, gpu, auto}`

`python -m evtrade backtest / sweep / replay` MUST 接受 `--device {cpu, gpu, auto}`
（默认 `auto`）。设备解析统一由 `evtrade.backends.resolve_device(requested, gpu_available)`
在三个子命令入口执行：`"gpu"` 且 CUDA 不可用 MUST 抛 `ValueError`（带可操作提示）；
`"auto"` 优先 gpu，不可用时降级 cpu 并打 warning（不抛）。`"cpu"` 直接返回。
引擎层（`run_vectorized` 等）MUST NOT 携带 device 参数——后端为 torch 统一，实际计算
路径设备无关（numpy 桶预计算 + 标量 step 循环）。旧 `--engine {kernel,ref,vectorized}`
flag MUST NOT 存在（2026-09-10 删除，传入报 argparse unknown option）。
MUST NOT 存在"被接受但从不读取"的 CLI flag（`--no-sleep` / `--step-days` /
`--show-bars` / `--bars-out` 已删除）。

#### Scenario: --device gpu 无 CUDA 报错
- **WHEN** 无 CUDA 的机器执行 `python -m evtrade backtest --device gpu ...`
- **THEN** MUST 抛 `ValueError`（提示 CUDA 不可用，改用 `--device auto` 或 `--device cpu`）

#### Scenario: --device auto 静默降级
- **WHEN** 无 CUDA 的机器执行 `python -m evtrade sweep --device auto ...`
- **THEN** 降级 cpu 并打 warning，正常运行完成

#### Scenario: --engine 已删除
- **WHEN** 执行 `python -m evtrade backtest --engine kernel ...`
- **THEN** argparse 报 `unrecognized arguments: --engine` 退出

### Requirement: PyTorch 统一后端

`evtrade` MUST 使用 PyTorch 作为唯一 array 后端能力来源（`pyproject.toml` 声明
`torch>=2.0`）；`evtrade.backends.gpu_available()` MUST 委托 `torch.cuda.is_available()`。
当前热路径（桶预计算 + 策略 step）为设备无关 numpy/标量实现，`backends.get_xp` 仅为
需要 tensor 的扩展代码提供 `torch.device` 路由。`evtrade/core/capability.py` MUST NOT
存在（能力探测收编至 `backends`）；`evtrade/core/tsbucket.py` MUST 为纯 numpy 桶级
`ts/mark` 预计算（模块名/内容 MUST NOT 含 gpu/cuda/torch 语义依赖，LRU 缓存行为不变）；
历法运算（`encoded_to_epoch` / `epoch_to_encoded`）MUST 单一真源于 `core/timeutils.py`
（标量与向量两形态共享 Hinnant 整数日历）。`gpu_info` 已删除。

#### Scenario: cupy 不再是可选依赖
- **WHEN** 用户执行 `grep -r "import cupy" evtrade/`
- **THEN** MUST 0 命中（cupy 已完全下线）

#### Scenario: tsbucket 纯 numpy
- **WHEN** 静态扫描 `evtrade/core/tsbucket.py`
- **THEN** MUST 无 `torch` / `cuda` / `cupy` 引用；`precompute_ts_mark` 输出与重构前
  bitwise 一致（`tests/test_tsbucket_cache.py` 锁定，仅 import 路径变更）

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

## New Requirements

### Requirement: Single trade-execution implementation

成交决策（取量 + 资金/持仓约束 + 现金/持仓更新）MUST 有唯一实现
`evtrade.execution.base.trade_decision(side, price, cash, position, cur_qty, buy_pct,
sell_pct) -> (new_cash, new_position, qty, filled)`。`SimulatedExecutor.trade` 与
`run_vectorized` 的成交循环（`vectorized_engine._execute_trades`）MUST 均调用它，
MUST NOT 各自复制取量/约束公式。scale（加仓翻倍）与 last_side 状态逻辑属于调用方
状态，MAY 留在各自调用处。`replay --against-ref` MUST 仍逐笔（ts/side/qty/price）
对账两条引擎路径的成交结果。

#### Scenario: 成交公式仅一处
- **WHEN** 静态扫描 `evtrade/`（排除 tests/）中的取量表达式（`cash / price`、
  `buy_pct *`、`sell_pct *` 组合）
- **THEN** MUST 仅出现于 `execution/base.py::trade_decision` 一处

#### Scenario: 双路径逐笔一致
- **WHEN** `replay --log <log> --strategy channel_deviation --device cpu --against-ref`
- **THEN** 输出 `对账 [PASS]`，两条路径 trades 逐笔（ts/side/qty/price）一致且终态
  cash/position 一致

### Requirement: CLI options declared once

**共享选项 MUST 仅声明一次**：`backtest` / `sweep` / `replay` 三个子命令的共享选项（`--strategy --params --code
--start --end --period --trade-qty --scale --all-in --buy-pct --sell-pct --warmup-days
--device --data-cache --verbose` 等）MUST 声明于单一共享父 parser（argparse `parents=`），
MUST NOT 各子命令重复声明同义选项。CLI 汇总打印 MUST 使用实际 `args` 值
（期初资金/持仓打印 `args.init_cash` / `args.init_position`，MUST NOT 打印模块常量
`INIT_CASH` / `INIT_POSITION`）。

#### Scenario: 共享选项单点声明
- **WHEN** 静态扫描 `evtrade/cli.py` 的 `add_argument` 调用
- **THEN** 上述共享选项名 MUST 仅出现于共享父 parser 定义处一次

#### Scenario: 汇总打印实际值
- **WHEN** `python -m evtrade backtest --init-cash 99999 ...`
- **THEN** 汇总头部"期初资金"行 MUST 打印 99999（非默认常量 100000）
