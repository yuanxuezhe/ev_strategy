## 1. pyproject.toml 加 `gpu` extra

- [x] 1.1 在 `pyproject.toml` 加 `[project.optional-dependencies]` 段，`gpu = ["torch==2.9.0+cu128"]`，并加注释说明需要从 `https://download.pytorch.org/whl/cu128` 索引装。验证：`grep -A2 "optional-dependencies" pyproject.toml` 命中。
- [x] 1.2 加 `[tool.uv.sources]` 段（uv 0.4.7+）指定 torch 在启用 `gpu` extra 时从 cu128 索引解析；保留 `dependencies` 中 `torch>=2.0` 不变。验证：`uv lock --help | grep index-strategy` 可用，`pip install -e .[gpu]` 失败前能看到 cu128 URL。
- [x] 1.3 用 `uv lock --upgrade-package torch==2.9.0+cu128 --index-strategy unsafe-best-match` 重生 lockfile，提交 `uv.lock`。验证：`grep -A1 "name = \"torch\"" uv.lock` 显示 `version = "2.9.0+cu128"` 或对应 cu128 source 段。

## 2. sync-torch-cu helper 脚本

- [x] 2.1 新建 `scripts/sync-torch-cu.sh`，加 shebang `#!/usr/bin/env bash` + `set -euo pipefail`；用 `.venv/Scripts/python`（Windows Git Bash）或 `python` 读 `torch.__version__` 与 `torch.version.cuda`；仅在当前是 CPU 版时 reinstall。验证：`bash scripts/sync-torch-cu.sh` 在 CPU 环境下退出 0 并把 torch 重写到 cu128。
- [x] 2.2 支持 `EVT_TORCH_CU_TAG` 环境变量（默认 `cu128`）覆写 cu tag；脚本顶部 echo 提示用法。验证：`EVT_TORCH_CU_TAG=cu126 bash scripts/sync-torch-cu.sh --dry-run` 打印预期 cu126 URL。
- [x] 2.3 在 `README.md` 与 `kbs/使用说明.md` 的"快速上手"段加一行 `bash scripts/sync-torch-cu.sh`（GPU 机器一次性）。验证：`grep -n "sync-torch-cu" README.md` 命中。

## 3. spec 落地（apply 阶段）

- [x] 3.1 把 `specs/evtrade-architecture/spec.md` 的 delta 内容合并进 `openspec/specs/evtrade-architecture/spec.md`（`R: PyTorch 统一后端` 与 `R: CLI surface = --device {cpu, gpu, auto}` 修订段，新增 Scenario）。验证：`openspec validate --specs` 通过。
- [x] 3.2 同步 `kbs/15-PyTorch统一策略.md` §7.1（重写 `--device` 与安装约定）：承认 gpu extra 是受支持路径；列出 `uv sync --extra gpu` / `bash scripts/sync-torch-cu.sh` 两条命令。验证：`grep -n "optional-dependencies\|sync-torch-cu\|cu128" kbs/15-PyTorch统一策略.md` 命中。
- [x] 3.3 同步 `kbs/10-配置参数与运行指南.md` "设备选择"段：加 GPU 机器一次性安装步骤。验证：`grep -n "sync-torch-cu\|cu128" kbs/10-配置参数与运行指南.md` 命中。
- [x] 3.4 同步 `kbs/使用说明.md`（如存在 GPU 章节）：加同样命令。验证：`grep -n "sync-torch-cu\|cu128" kbs/使用说明.md` 命中。
- [x] 3.5 更新 `CLAUDE.md` §5：删去 "MUST NOT 存在独立 GPU extra" 硬约束，改为引用本 spec + KB。验证：`grep -n "optional-dependencies" CLAUDE.md` 命中（引述 spec，不是否定）。

## 4. 测试

- [x] 4.1 改写 `tests/test_pyproject.py`：增加 `test_gpu_optional_extra_present`、`test_no_other_gpu_extras`、`test_uv_sources_for_torch`、更新 `test_requires_python_at_least_38 -> test_requires_python_at_least_310`，反转 `test_no_optional_gpu_extra`。验证：`pytest tests/test_pyproject.py -v` 通过。
- [x] 4.2 `tests/test_torch_backend.py` 加一条断言：`get_xp("gpu")` 在 `torch.cuda.is_available()` 时返 `cuda` device（已有等价测试则跳过）。验证：`pytest tests/test_torch_backend.py -v` 通过。
- [x] 4.3 `tests/test_device_resolution.py` 加一条断言：`resolve_device("gpu", gpu_ok=False)` 抛 `ValueError` 且文案含"uv sync --extra gpu"或"sync-torch-cu"。验证：`pytest tests/test_device_resolution.py -v` 通过。
- [x] 4.4 新建 `tests/test_sync_torch_cu.py` 覆盖脚本存在/幂等/默认 cu128/接受环境变量。验证：`pytest tests/test_sync_torch_cu.py -v` 通过。
- [x] 4.5 修复 `tests/test_batched_sweep.py::test_sweep_batched_fallback_to_cpu_when_no_cuda`：用 monkeypatch 强制 `gpu_available()=False` 而非依赖环境。验证：`pytest tests/test_batched_sweep.py -v` 通过。

## 5. 验证清单（合入前必跑）

- [x] 5.1 `openspec validate --specs` 通过。
- [x] 5.2 `bash scripts/sync-torch-cu.sh` 退出 0；之后 `.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"` 打印 True。
- [x] 5.3 `uv run pytest -q` 全绿（192 passed, 1 skipped）。
- [x] 5.4 `python -m evtrade sweep ... --device gpu ...` 跑通（81 combos → 54 行，bitwise 与 baseline 一致）。
- [x] 5.5 hygiene grep 0 实质命中（仅"已删除/已下线"删注）。

## 6. archive

- [ ] 6.1 合入 main 后跑 `openspec archive add-gpu-extra-pyproject` 归档 change；确认 `openspec/specs/evtrade-architecture/spec.md` 的 delta 已合并 + `openspec/changes/add-gpu-extra-pyproject/` 进入 archive。（**待 commit + 合入后执行**）