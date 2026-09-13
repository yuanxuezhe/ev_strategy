## 1. pyproject.toml 去 gpu extra

- [x] 1.1 删 `[project.optional-dependencies].gpu = ["torch==2.9.0+cu128"]`。
  验证:`grep -c "optional-dependencies" pyproject.toml` → 活跃配置 0 命中。
- [x] 1.2 删 `[tool.uv.sources]` 的 torch cu128 覆盖 + `[[tool.uv.index]] pytorch-cu128`
  (已无引用)。验证:`grep -n "pytorch-cu128\|tool.uv.sources" pyproject.toml` → 活跃配置 0 命中。
- [x] 1.3 保留核心依赖 `torch>=2.0`; 加注释说明"GPU 不走 extra 的原因 + 改走 sync-torch-cu.sh"。
  验证:`grep -n '"torch>=' pyproject.toml` 命中。
- [x] 1.4 更新底部 uv 工具注释: `uv sync` 默认 CPU, GPU 走 `bash scripts/sync-torch-cu.sh`。

## 2. 代码 / 文案

- [x] 2.1 `evtrade/backends.py` `resolve_device` 无 CUDA 报错文案: 删 `uv sync --extra gpu`,
  改 `uv sync` 后 `bash scripts/sync-torch-cu.sh`。验证:`grep -n "extra gpu" evtrade/` → 0 命中。

## 3. spec + KB + CLAUDE.md

- [x] 3.1 `openspec/specs/evtrade-architecture/spec.md` 反转 `R: PyTorch 统一后端`
  (默认 pypi CPU, MUST NOT gpu extra / cu128 source, GPU 走脚本); 更新
  `R: CLI surface = --device` 报错文案 Scenario; 更新底部映射表行(kbs/15→kbs/07)。
- [x] 3.2 `kbs/07-PyTorch与性能.md` §3.1 重写为"默认 CPU + GPU 走脚本"; 更新
  `--device gpu` 提示、验证清单安装段、changelog 加 `drop-gpu-extra-cpu-default` 行。
- [x] 3.3 `kbs/05-引擎与CLI.md` / `kbs/README.md` 安装段: `uv sync` + `bash scripts/sync-torch-cu.sh`。
- [x] 3.4 `kbs/01-总览.md` 技术栈行: `[project.optional-dependencies].gpu` → "GPU 由脚本覆盖到 cu128"。
- [x] 3.5 `CLAUDE.md` §5 GPU 安装约定: 默认 CPU, MUST NOT gpu extra, GPU 走脚本; 修正 kbs/15→kbs/07。

## 4. tests

- [x] 4.1 `test_pyproject.py`: 反转为 `test_no_gpu_optional_extra` / `test_no_cupynum_gpu_extras` /
  `test_no_torch_cu128_source`(新增 `_active_pyproject()` 剔除注释行避免误伤文档注释)。
- [x] 4.2 `test_torch_backend.py`: 断言文案含 `sync-torch-cu.sh` 且不含 `--extra gpu`; 更新 docstring。

## 5. 验证(本机 Windows / py3.12 / 无 NVIDIA)

- [x] 5.1 `rm uv.lock && uv lock` → torch = pypi `2.14.0`, lock 内 0 个 `cu128`。
- [x] 5.2 `uv sync --dry-run` → `+ torch==2.14.0`, 无 nvidia 包。
- [x] 5.3 `uv sync` 真实安装 → `torch 2.14.0+cpu`, `cuda=None`, site-packages 零 nvidia。
- [x] 5.4 `uv run python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device cpu` 跑通。
- [x] 5.5 `uv run pytest -q` → 169 passed, 1 skipped(GPU 幂等测试在 CPU wheel 下正确跳过)。
- [x] 5.6 `openspec validate --specs` 通过。**顺带修复既有问题**:3 条 Requirement
  (filtered_mr / Engine drives step only / CLI is step-driver surface) 首段缺 MUST/SHALL,
  校验器按 header 后首段判 → 各补一句不改行为的规范 MUST。已用 `git stash` 对照确认
  该失败在原始 spec 上同样存在(非本 change 引入)。

## 6. archive

- [ ] 6.1 合入后 `openspec archive drop-gpu-extra-cpu-default`; 确认 spec delta 已合并进
  `openspec/specs/evtrade-architecture/spec.md`, change 目录移入 archive/。
  (注: `openspec validate --specs` 原本失败的既有问题已在本 change §5.6 顺带修复, 现通过。)
