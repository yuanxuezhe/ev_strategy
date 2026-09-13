## Why

`add-gpu-extra-pyproject` (2026-09-12) 在 `pyproject.toml` 声明了
`[project.optional-dependencies].gpu = ["torch==2.9.0+cu128"]` +
`[tool.uv.sources]` / `[[tool.uv.index]]` 指向 cu128 索引, 意图让 GPU 协作者
`uv sync --extra gpu` 一行锁住 cu128 wheel。

实测发现这条路径有**反向副作用**:`uv lock` 会把 base 依赖(`torch>=2.0`)与所有
extras **一起锁进同一份 `uv.lock`**。torch 同时被 base 与 gpu extra 引用, 而
`2.9.0+cu128` 是只存在于 cu128 索引的本地版本号, uv 只能把 torch **按包名统一塌缩**
成 cu128 —— 于是**无 NVIDIA 显卡的 CPU 机器, 默认 `uv sync` 也被迫下载 GPU 版 torch
wheel**(实测本机 `uv sync` 装到 `torch 2.9.0+cu128`, 几百 MB 一次性下载)。

复现与根因(本机 Windows / Python 3.12, 无 NVIDIA 卡):

- 删除 gpu extra + cu128 source 后 `uv lock` → torch 解析为 pypi `2.14.0`(CPU),
  lock 内 0 个 `cu128` 引用。
- 保留 gpu extra 时 `uv lock` → torch 统一为 `2.9.0+cu128`, 且 numpy 等 base 依赖
  一度从 cu128 索引拉(索引未设 `explicit` 时还会泄漏成所有包的 fallback)。
- 一份共享 lock **无法**同时表达"默认 CPU + 某 extra 才 cu128"两个不同版本 —— 这是
  uv 对同包名 CPU/GPU 双版本的限制, `uv 0.11.7` 亦无 `uv lock --no-extra`。

项目里其实早有一条真正独立的 GPU 安装路径:`scripts/sync-torch-cu.sh`。它在 `uv sync`
之后把 venv 内 torch **就地覆盖**到 cu128(探测 / 幂等 / `EVT_TORCH_CU_TAG` 覆写),
**与 lockfile 无关**, 天然不会污染共享 lock。`--extra gpu` 是半成品, 没能让 GPU 机器
"多得到"什么(脚本已能装 cu128), 却让所有 CPU 机器都被迫下载 cu128。

## What Changes

- **`pyproject.toml`**:删除 `[project.optional-dependencies].gpu`、`[tool.uv.sources]`
  的 torch cu128 覆盖、`[[tool.uv.index]] pytorch-cu128`(已无引用)。保留核心依赖
  `torch>=2.0`(pypi CPU wheel 默认)。加注释说明为何 GPU 不走 extra。
- **`evtrade/backends.py`**:`resolve_device` 的 `--device gpu` 无 CUDA 报错文案,
  从"请先 `uv sync --extra gpu` 或 `bash scripts/sync-torch-cu.sh`"改为
  "请先 `uv sync` 再 `bash scripts/sync-torch-cu.sh`"。
- **`uv.lock`**(本地, 不入库):重新 `uv lock`, torch 回到 pypi CPU 默认。
- **spec delta**:`R: PyTorch 统一后端` 反转安装约束 —— 默认 MUST 拉 pypi CPU wheel,
  pyproject MUST NOT 声明 gpu extra / cu128 source; GPU MUST 走 `sync-torch-cu.sh`。
  `R: CLI surface = --device` 的报错文案 Scenario 同步改为 `uv sync` + 脚本。
- **tests**:`test_pyproject.py` 三个断言(gpu extra 存在 / sources 存在)反转为
  "无 gpu extra / 无 cu128 source / torch>=2.0 保留"; `test_torch_backend.py` 断言
  文案含 `sync-torch-cu.sh` 且**不再**含 `--extra gpu`。
- **KB / CLAUDE.md**:kbs/01、05、07 §3.1、README、CLAUDE.md §5 同步为
  "默认 CPU + GPU 走脚本"。

## Capabilities

### New Capabilities
(无)

### Modified Capabilities
- `evtrade-architecture`:`R: PyTorch 统一后端` 安装约束反转(extra → 脚本);
  `R: CLI surface = --device {cpu,gpu,auto}` 的无 CUDA 报错文案 Scenario 更新。

## Impact

- **依赖 / 装包**:CPU 协作者 `uv sync` 拿 pypi CPU torch(`+cpu`, `cuda=None`, 零
  nvidia 包); GPU 协作者 `uv sync` + `bash scripts/sync-torch-cu.sh` 覆盖到 cu128。
  `--device {cpu,gpu,auto}` 运行时行为不变。
- **lockfile**:`uv.lock` 不入库(各机器本地生成), 故本次改动不影响他人 lock;
  本机重 lock 后 torch = pypi 2.14.0。
- **测试**:GPU 单测跳过条件不变(需 `pytest -m gpu` / `--gpu`)。
- **BREAKING**:依赖 `uv sync --extra gpu` 的既有脚本 / 文档失效, 需改走脚本
  (已全量 sweep 更新 kbs / CLAUDE.md / tests)。
