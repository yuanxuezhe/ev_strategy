## MODIFIED Requirements

### Requirement: PyTorch 统一后端

`evtrade` MUST 使用 PyTorch 作为唯一 array 后端能力来源（`pyproject.toml` 声明
`torch>=2.0`）；CPU 与 GPU 由 **同一 torch 包**的不同 wheel 提供。**默认 `uv sync`
MUST 拉 pypi.org 的 CPU-only wheel**；`pyproject.toml` MUST NOT 声明 GPU optional
extra（`gpu = ["torch==2.9.0+cu128"]`），MUST NOT 为 torch 配置指向 cu128 索引的
`[tool.uv.sources]` / `[[tool.uv.index]]` —— 因为 `uv lock` 会把 base 依赖与所有
extras 一起锁进同一份 `uv.lock`，声明 `gpu` extra 会把 torch 按包名统一塌缩成
cu128，导致无 GPU 的 CPU 机器也被迫下载 GPU wheel。GPU 协作者的 cu128 wheel 安装
MUST 走 `scripts/sync-torch-cu.sh`（在 `uv sync` 之后把 venv 内 torch 覆盖到 cu128，
与 lockfile 无关）。`evtrade.backends.gpu_available()` MUST 委托
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

#### Scenario: 无 GPU optional extra（避免污染共享 lock）
- **WHEN** 用户执行 `grep -n "optional-dependencies\|cu128" pyproject.toml`
- **THEN** 活跃配置（非注释行）MUST NOT 含 `[project.optional-dependencies].gpu`
  段；MUST NOT 出现 `torch==2.9.0+cu128`；MUST NOT 为 torch 配 cu128 索引 source
  （`[tool.uv.sources]` / `[[tool.uv.index]]` 不引用 cu128）；`torch>=2.0` MUST 保留
  为核心依赖（pypi CPU wheel 默认）；不含 `cupy` / `numba`

#### Scenario: sync-torch-cu helper 是 GPU wheel 唯一安装路径
- **WHEN** GPU 协作者 `uv sync`（CPU wheel 装上）后跑 `bash scripts/sync-torch-cu.sh`
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
请 `uv sync` 后跑 `bash scripts/sync-torch-cu.sh`"）；`"auto"` 优先 gpu，不可用时降级
cpu 并打 warning（不抛）。`"cpu"` 直接返回。**`--device gpu` 报错的根本原因是 torch
包未安装 CUDA wheel；GPU 协作者 MUST 先 `uv sync` 再 `bash scripts/sync-torch-cu.sh`
把 torch 覆盖到 cu128 wheel 后再使用 `--device gpu`。** 引擎层（`run_vectorized` 等）
MUST NOT 携带 device 参数——后端为 torch 统一，实际计算路径设备无关（numpy 桶预计算
+ 标量 step 循环）。旧 `--engine {kernel,ref,vectorized}` flag MUST NOT 存在（2026-09-10
删除，传入报 argparse unknown option）。MUST NOT 存在"被接受但从不读取"的 CLI flag
（`--no-sleep` / `--step-days` / `--show-bars` / `--bars-out` 已删除）。

#### Scenario: --device gpu 无 CUDA 报错
- **WHEN** 未安装 GPU 版 torch 的机器执行 `python -m evtrade backtest --device gpu ...`
- **THEN** MUST 抛 `ValueError`，文案明确：当前 torch 构建无 CUDA 支持；提示改用
  `--device auto` / `--device cpu`；GPU 机器请先 `uv sync` 后 `bash scripts/sync-torch-cu.sh`

#### Scenario: --device gpu 安装正确时跑通
- **WHEN** 已 `uv sync` + `bash scripts/sync-torch-cu.sh`（torch 为 cu128）的机器执行
  `python -m evtrade backtest --device gpu ...`
- **THEN** MUST 跑通；`torch.cuda.is_available()` 为 True；与 `--device cpu` 路径
  产出 bitwise 一致

#### Scenario: --device auto 静默降级
- **WHEN** 未安装 GPU 版 torch 的机器执行 `python -m evtrade sweep --device auto ...`
- **THEN** 降级 cpu 并打 warning，正常运行完成
