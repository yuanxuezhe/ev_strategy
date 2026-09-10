# Proposal: simplify-user-surface — 用户文档单入口 + GPU 声明去冗余

## Why

`consolidate-simplify-core` 已把代码面收敛干净（成交收口、死代码删除、CLI 去重）。剩余两处
与"干净、简洁、CPU/GPU 统一"目标不符的**用户面**残留：

1. **用户文档 5 处并行**：根 README、`kbs/README`、`kbs/使用说明.md`、`docs/quickstart.md`、
   `docs/params-workflow.md` 内容重叠。改一处漏一处（`docs/params-workflow.md` 还残留
   `BrokerExecutor` 占位示例）。
2. **`pyproject.toml` 的 `gpu`/`all` extra 空转**：`gpu = ["torch>=2.0"]` 与核心依赖完全相同，
   `uv sync --extra gpu` 装出来的东西与 `uv sync` 无差别。它暗示"GPU 是另一条安装路径"，
   与"torch 单后端、CPU/GPU 同一份代码"的统一叙事矛盾。

## What Changes

### A. 文档收敛（docs/ 并入 kbs/）

- 删 `docs/quickstart.md`：内容并入 `kbs/使用说明.md` 第 0 章（uv 安装 + 故障排查 +
  离线/内网 + 与 pip 兼容 + uv.lock 协作）。
- 删 `docs/params-workflow.md`：四步工作流（扫描→挑参→落盘→回测）并入
  `kbs/使用说明.md` 新章（修掉 `BrokerExecutor` 占位示例，实盘接入 = 自定义 bar 流 +
  自定义 Executor）；参数解析优先级段并入 `kbs/10-配置参数与运行指南.md` §1。
- 根 `README.md` 保持唯一入口（安装 / 快速上手 / 工作流），"详见"指向 kbs/。
- `kbs/README.md` 文档索引保持；`使用说明.md` 章节重编号。
- 全仓 `docs/` 引用清零（`openspec validate` 不涉，纯文档）。

### B. pyproject GPU 声明去冗余

- 删 `[project.optional-dependencies]`（`gpu`/`all` 两段）；删 `test_gpu_extra_listed`，
  新增断言"无 optional-dependencies 段（torch 已在核心依赖，无独立 GPU 安装路径）"。
- 同步文案：`pyproject` 内 `uv sync --extra gpu` 注释、`README.md` 安装段（删
  `uv sync --extra gpu`）、`kbs/使用说明.md` 第 0 章、`kbs/15` 的 uv 命令示例。
- **不改** `--device` CLI 语义与 `backends.py`（hot-path 仍是 numpy；`--device` 为统一
  后端预留 `get_xp`，这一事实已在 kbs/15 写明）。

## Impact

- 删除：`docs/` 整个目录（2 文件，351 行）。
- 修改：`pyproject.toml`、`tests/test_pyproject.py`、`README.md`、`kbs/使用说明.md`、
  `kbs/10`、`kbs/15`、`kbs/README.md`、`CLAUDE.md`（"详见"段）。
- 行为：零代码行为变化（纯文档 + 元数据）；`uv sync` 后依赖集合不变
  （torch 本就是核心依赖）。
- 门禁：`uv run pytest -q` 全绿（142±）；`npx openspec validate --specs`；
  `grep -rn "docs/quickstart\|docs/params-workflow\|extra gpu" README.md kbs/ CLAUDE.md pyproject.toml`
  应 0 命中；`grep -n "optional-dependencies" pyproject.toml` 应 0 命中。
