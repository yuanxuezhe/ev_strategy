## Why

`pyproject.toml` 当前只声明 `torch>=2.0`，uv.lock 解析到 pypi.org 的最新 wheel `torch==2.14.0+cpu`（CPU-only）。在这台 RTX 5090 (Blackwell sm_120) 机器上，`--device gpu` 必然踩中 spec 强制的 `--device gpu 无 CUDA 报错` Scenario —— `torch.cuda.is_available()` 永远是 False，因为 torch 本身没 CUDA。

现状：
- 没有 GPU 的机器：默认就够用
- 有 GPU 的机器：必须绕开 uv 的 lockfile，手动 `uv pip install torch==2.9.0+cu128`，但任何后续 `uv run` 又会被覆盖回 CPU 版（已实测）

需要把"GPU 机器装 cu128 wheel"做成**显式声明的可选 extra**，让 `uv sync --extra gpu` 一行命令锁住，同时更新 spec/KB/CLAUDE.md 让这条安装路径成为受支持的能力。

## What Changes

- **`pyproject.toml`**：新增 `[project.optional-dependencies]` 段，含 `gpu = ["torch==2.9.0+cu128"]`（带 `cu128` 索引的 marker 说明在注释中）；`dependencies` 中 `torch>=2.0` 保持不变。
- **`scripts/sync-torch-cu.sh`** (新增)：在 `uv sync` 之后跑（必要时用 `--reinstall` + `--index-strategy unsafe-best-match` + `--index-url https://download.pytorch.org/whl/cu128`），把 torch 重写到 cu128 wheel；CI / 个人开发用同一脚本。**BREAKING**：未来 lockfile 可能因此需要重生成（已有 lockfile 必须 re-lock 或第一次跑会临时覆盖）。
- **`uv.lock`**：需用 `uv lock --upgrade-package torch==2.9.0+cu128` 重生（`--index-strategy unsafe-best-match` + cu128 index）。这是仓库级强制更新，PR review 重点。
- **spec delta**：
  - `R: PyTorch 统一后端` 修订"无独立 GPU 安装路径"约束 → 改为承认 `[project.optional-dependencies].gpu` 是受支持的安装路径；
  - Scenario `gpu 不再是独立安装路径` 改写为正向陈述：可选 extra 存在 + 触发命令 + sync helper；
  - 新增 Scenario 描述 sync helper 行为。
- **KB delta**：
  - `kbs/15-PyTorch统一策略.md` §7.1 重写 `--device` 与安装约定的关系（CPU/GPU 两条安装路径用 extra 区分）；
  - `kbs/10-配置参数与运行指南.md` 增加"GPU 机器一次性安装步骤"段；
  - `kbs/使用说明.md` 同上。
- **CLAUDE.md §5**：删除"无独立 GPU extra"硬约束，改为引用 spec + KB 的最新约定。
- **tests**：新增 `tests/test_pyproject_gpu_extra.py`，断言 `[project.optional-dependencies].gpu` 存在 + torch 版本要求。

## Capabilities

### New Capabilities
（无）

### Modified Capabilities
- `evtrade-architecture`：R `PyTorch 统一后端` 修改安装约束；同时 `--device gpu 无 CUDA 报错` Scenario 文案需配合 `(已安装 GPU 版 torch 的) 机器` 语义精确化（安装错仍是 ValueError）。

## Impact

- **依赖**：`uv.lock` 必须重生（PR 必带）。CI 默认 `uv sync` 仍走 CPU，GPU CI 用 `uv sync --extra gpu`。
- **构建 / 装包**：所有用 `uv run` 的人没变化（默认仍是 CPU wheel）；GPU 机器装法从"手动覆盖 + 反复被覆盖"变成"装一次锁住"。
- **API / 行为**：`--device {cpu,gpu,auto}` 行为完全不变；`backends.resolve_device` 行为不变；CPU wheel 装好时 `--device gpu` 仍 ValueError（这是正确的，符合 spec）。
- **测试**：GPU 单测跳过条件不变（需 `pytest -m gpu`）。