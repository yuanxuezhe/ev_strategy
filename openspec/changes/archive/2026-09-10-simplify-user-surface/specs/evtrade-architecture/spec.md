# Spec Delta: simplify-user-surface

## MODIFIED

### Requirement: PyTorch 统一后端

策略代码 MUST 单份实现、CPU (`torch.device("cpu")`) 与 GPU (`torch.device("cuda")`) 双端统一运行；
`evtrade/backends.py::get_xp(device)` MUST 是唯一的后端路由点（返回 `torch.device`），
`gpu_available()` / `resolve_device(requested, gpu_ok)` 在 backends 单点实现。
策略代码 MUST NOT `import numba` / `import cupy`。同一份策略代码 MUST 在 `device="cpu"` 与
`device="gpu"`（torch CUDA 可用时）下产出 bitwise 一致的信号（浮点字段 ULP 容差内）。
`torch` MUST 声明为 `pyproject.toml` 核心依赖；MUST NOT 存在 `gpu`/`all` 等
`[project.optional-dependencies]` 独立安装路径（torch 单后端，CPU 与 GPU 是同一依赖的不同
运行时，不是两个 extra）。`evtrade/core/tsbucket.py` 只保留桶 ts/mark 预计算与 LRU 缓存
（标量与向量两形态共享 Hinnant 整数日历）。`gpu_info` 已删除。

#### Scenario: gpu 不再是独立安装路径

- **WHEN** 用户执行 `grep -n "optional-dependencies" pyproject.toml`
- **THEN** MUST 0 命中（无 `gpu`/`all` extra；torch 在核心 `dependencies`）
- **AND** `uv run pytest -q` 中 `tests/test_pyproject.py` 断言"无 optional-dependencies 段"

#### Scenario: cupy 不再是可选依赖

- **WHEN** 用户执行 `grep -r "import cupy" evtrade/`
- **THEN** MUST 0 命中（cupy 已完全下线）

## ADDED

### Requirement: 用户文档单一入口

用户向文档 MUST 收敛为两条线：**根 `README.md`**（唯一入口：安装 / 快速上手 / 工作流，
≤150 行）+ **`kbs/`**（中文详述，其中 `使用说明.md` 是操作手册、`10-配置参数与运行指南.md`
是参数权威）。仓库根 MUST NOT 存在 `docs/` 目录（原 `quickstart.md` / `params-workflow.md`
内容已并入 `kbs/使用说明.md` / `kbs/10`）。根 README 与 `kbs/` 中对安装/工作流的命令示例
MUST 一致（同一 flag 集合，不得出现已删 flag 或已删符号如 `BrokerExecutor`）。

#### Scenario: docs/ 目录已删除

- **WHEN** 用户执行 `ls docs/` 且 `grep -rn "docs/quickstart\|docs/params-workflow" README.md kbs/ CLAUDE.md`
- **THEN** 前者 MUST 报 No such file or directory；后者 MUST 0 命中

#### Scenario: 快速上手命令与 CLI 实际一致

- **WHEN** 按根 README"快速上手"中的回测命令逐字执行（`--device auto` 或 `cpu`）
- **THEN** MUST 不报 `unrecognized arguments`（示例只使用现行 CLI flag）
