## Context

参见 `proposal.md` Why。当前状态：`pyproject.toml` 单依赖 `torch>=2.0`；`uv.lock` 锁在 `2.14.0` (pypi.org CPU wheel)。本机 RTX 5090 已实测必须手动 `uv pip install --index-strategy unsafe-best-match torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --reinstall`，且每次 `uv run` 都会被覆盖回 CPU。

约束（CLAUDE.md §5 现行）：
- `--device {cpu,gpu,auto}` 是唯一运行时旋钮，不变；
- `backends.resolve_device` 行为不变（CPU wheel 装 `--device gpu` 仍 ValueError，符合 spec）；
- 策略代码 / framework / API 完全不动 — 纯安装侧变更。

## Goals / Non-Goals

**Goals:**
- GPU 机器一次性安装命令 `uv sync --extra gpu`（或旧 lockfile 兼容的 `bash scripts/sync-torch-cu.sh`），锁住 `torch==2.9.0+cu128`，之后 `uv run` 不再回退。
- CPU 机器 `uv sync` 默认行为零变化（仍走 pypi.org CPU wheel）。
- spec / KB / CLAUDE.md 同步演进，单一真源收敛。

**Non-Goals:**
- 不引入 `cupy` / `numba`（spec R `PyTorch 统一后端` 仍生效）。
- 不为不同 CUDA 版本（cu118 / cu126 / cu130）各起一个 extra — 只维护 cu128 一个，因 RTX 50 系列 (sm_120) 必需 cu128+，旧卡用户暂时用 CPU 版即可（已有生产 sweep 经验）。
- 不改 `--device` 解析、`backends` 模块、`VectorizedStrategy` 接口。
- 不动 GPU-batched sweep hook 逻辑（这是 GPU 计算路径优化，不是安装路径）。

## Decisions

### Decision 1: 单 extra `gpu` 锁 cu128

- **选择**：`[project.optional-dependencies] gpu = ["torch==2.9.0+cu128"]`。
- **理由**：项目方目前只有这台 RTX 5090 (sm_120) 作为 GPU 目标硬件；cu128 是首个官方支持 Blackwell 的稳定轮子（torch 2.7+ 起）。多版本维护（cu118/cu126/cu130）成本高于价值，先收口。
- **替代方案**：
  - **A. 不写死 cu128，让 `uv sync --extra gpu` 自动拉 cu128 latest**：放弃。`>=2.7+cu128` 跨 major 重生 lock 时容易拿到 `+cpu`（pypi.org 也有 cu128 镜像，但默认源会拿 CPU）。写死更可控。
  - **B. 在 `[project.optional-dependencies]` 之外用 `[tool.uv.sources]` 指定 torch 索引**：技术上更干净（PEP 508 直接 `torch @ https://download.pytorch.org/whl/cu128/torch-2.9.0+cu128-cp312-cp312-win_amd64.whl`），但每个平台要列全 wheel URL，跨平台不可移植。放弃。
  - **C. 不加 extra，文档告诉用户"自己手动 pip install"**：放弃。已在文档外挣扎了一次（本次会话现场），不可重复。

### Decision 2: 同步 helper 脚本 vs. 纯文档说明

- **选择**：新增 `scripts/sync-torch-cu.sh` (bash)，逻辑是 detect 当前 torch 后端 + 如果是 CPU 就 reinstall cu128。
- **理由**：用户群体在 Windows 上用 Git Bash 调用 `bash scripts/...` 没问题（CLAUDE.md 已明确 "Shell: bash"）。脚本里 `set -euo pipefail` + 检查 `torch.version.cuda` 字段，避免误覆盖已经正确的版本。
- **替代方案**：
  - **A. 纯文档 `docs/gpu-install.md`**：用户在 lockfile 重生后必须手工跑那段长命令；上文已证手工 `uv pip install` 后被 `uv run` 覆盖是隐形问题，文档无法阻止。
  - **B. 写到 `pyproject.toml` 的 `[tool.uv]` post-hook**：uv 不支持 post-hook，放弃。

### Decision 3: `uv.lock` 的处理方式

- **选择**：本 change 提交前 `uv lock --upgrade-package torch==2.9.0+cu128 --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128` 重生 lock，然后 commit lockfile。
- **理由**：spec 不强制 lockfile 内容，但 `uv.lock` 是仓库级共享资产，**必须**随 change 一起提交（否则别人 `uv sync` 拿不到 cu128）。
- **替代方案**：
  - **A. 不提交 lockfile，让 CI 用 `--frozen` 失败**：放弃，会破 CI。
  - **B. 加 `.gitignore uv.lock`**：放弃，违背 uv 的最佳实践。

### Decision 4: spec delta 范围

- **选择**：仅修改 `R: PyTorch 统一后端`（承认 gpu extra），对应 Scenario `gpu 不再是独立安装路径` 改写为正向陈述。新增一个 Scenario 描述 sync helper。
- **理由**：行为变化只在于"安装路径可选"这一条；`--device`、API、framework 逻辑全部不动。其它 Requirements (CLI / step / metrics / etc.) 不受影响。
- **替代方案**：新增独立 capability `install-and-sync`。放弃，会拆碎 spec 主线，且该 capability 与现有 R `PyTorch 统一后端` 高度耦合（同一主题的两个面）。

## Risks / Trade-offs

- [Risk] `uv lock --upgrade-package torch` 在没有 cu128 索引机器上会失败 → Mitigation：脚本里先调用 `uv lock` (CPU 默认)，再 reinstall torch；lockfile 中 torch 段保留 cu128 metadata。文档明示。
- [Risk] Windows 上 `bash scripts/sync-torch-cu.sh` 依赖 Git Bash（CLAUDE.md 已确认环境是 bash）→ Mitigation：脚本加 shebang `#!/usr/bin/env bash` + 顶部 `echo "[sync-torch-cu] requires bash"` 提示。
- [Risk] 未来 NVIDIA 推新 CUDA / RTX 60 系列 → Mitigation：脚本接受 `EVT_TORCH_CU_TAG` 环境变量覆写 (默认 `cu128`)，不修改脚本即可适配。
- [Risk] `uv.lock` 跨平台 (linux/macOS/windows) 重生时 torch wheel 不同 → Mitigation：lockfile 是 per-platform 多 marker，标准做法；当前仓库已这样管理 (lockfile 含 `python_full_version` / `sys_platform` 多个 torch 条目)。
- [Trade-off] 单 cu128 extra 让 Ampere (sm_80) / Ada (sm_89) 用户失去 GPU 选项 → 接受：这些卡也能跑 cu128 wheel，只是更老的卡可降级到 cu118；目前项目方无该需求。

## Migration Plan

1. 编辑 `pyproject.toml` + 加 `scripts/sync-torch-cu.sh`。
2. `uv lock --upgrade-package torch==2.9.0+cu128 --index-strategy unsafe-best-match` (在 cu128 index 下重生 lock)。
3. 提交 change + lockfile 一起 PR。
4. GPU 机器协作者拉 PR 后跑 `uv sync --extra gpu` 或 `bash scripts/sync-torch-cu.sh`。
5. CPU 机器协作者：现有 `uv sync` 不变，零动作。

**回滚**：revert 单 commit 即可；pyproject.toml / lockfile / 脚本同时还原。

## Open Questions

无。所有决策都已收敛（cu128 写死、脚本而非文档、单 extra、spec delta 范围）。