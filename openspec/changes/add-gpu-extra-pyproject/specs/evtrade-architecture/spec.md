## MODIFIED Requirements

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