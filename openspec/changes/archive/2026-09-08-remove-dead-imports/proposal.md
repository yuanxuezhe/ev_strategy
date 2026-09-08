## Why

上一轮"框架去特殊化"重构（commits `b0dc322`、`dfd4785` 等）落地后，`evtrade/` 包内遗留了少量未使用的 import 与一个仅本文件内部调用的辅助函数。这些零行为影响的卫生问题应在维护窗口集中清理，避免日后误判为"被外部需要"，降低阅读与维护成本。本次为 openspec 工作流的"开张案例"，选取**纯内部、零公共 API 影响**的子集先行落地，建立变更闭环。

## What Changes

清理 5 处私有死代码（4 行未用 import + 1 个函数改名），**零公共 API 变动**、**零 KB/spec 改动**：

- `evtrade/cli.py:639` — 从 `from .strategies._defaults_loader import save, params_from_csv_row, path_for` 中移除未使用的 `path_for`。
- `evtrade/strategies/_defaults_loader.py:155` — 移除未引用的 `import pandas as _pd`（pandas 由调用方导入）。
- `evtrade/strategies/_defaults_loader.py:28` — 从 `from typing import Any` 中移除未使用的 `Any`。
- `evtrade/strategies/dsl.py:460` — 移除未使用的 `from .base import get_strategy_state_spec`（本文件内从未调用该名字）。
- `evtrade/strategies/channel_deviation.py:191` — 函数 `compute_deviation_columns` 重命名为 `_compute_deviation_columns`（仅本文件 `get_extra_bucket_columns` 内部调用），同步更新 line 170 调用点。

## Capabilities

### New Capabilities
（无新增能力。）

### Modified Capabilities
（无现有 capability 的 Requirement 行为变更；openspec/specs/evtrade-architecture/spec.md 的 8 条 Requirement 均不引用上述 import/函数，本变更不影响 spec 表面。）

## Impact

- **代码量**：净减 4 行 import + 1 个函数签名重命名（caller 同步）。
- **公共 API**：零变化（这些 import 本就无 caller；`_compute_deviation_columns` 重命名后仍可通过 `evtrade.strategies.channel_deviation._compute_deviation_columns` 访问，但目前零调用方）。
- **测试**：期望 `pytest -q` 保持 279 passed / 3 skipped（GPU），行为零变化。
- **性能**：零影响。
- **KB / openspec spec**：`kbs/` 与 `openspec/specs/evtrade-architecture/spec.md` 均不引用上述符号，**无需同步**。
- **回滚**：单文件局部编辑，`git revert` 即恢复。
