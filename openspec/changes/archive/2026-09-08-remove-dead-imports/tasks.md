# Tasks: remove-dead-imports

## 1. 清理未使用的 import

- [x] 1.1 `evtrade/cli.py:639` — 从 `from .strategies._defaults_loader import save, params_from_csv_row, path_for` 中移除 `path_for`
- [x] 1.2 `evtrade/strategies/_defaults_loader.py:155` — 移除未使用的 `import pandas as _pd`
- [x] 1.3 `evtrade/strategies/_defaults_loader.py:28` — 从 `from typing import Any` 中移除 `Any`
- [x] 1.4 `evtrade/strategies/dsl.py:460` — 移除未使用的 `from .base import get_strategy_state_spec`

## 2. 重命名内部辅助函数

- [x] 2.1 `evtrade/strategies/channel_deviation.py:191` — 函数 `compute_deviation_columns` → `_compute_deviation_columns`
- [x] 2.2 `evtrade/strategies/channel_deviation.py:170` — 同步更新 `get_extra_bucket_columns` 内的调用点

## 3. 验证

- [x] 3.1 运行 `.venv/Scripts/python.exe -m pytest -q` 确认 279 passed / 3 skipped（GPU）✓
- [x] 3.2 `grep -rEn "from .base import get_strategy_state_spec" evtrade/strategies/dsl.py` 应无命中 ✓
- [x] 3.3 `grep -rEn "\bpath_for\b" evtrade/cli.py evtrade/strategies/_defaults_loader.py` 仅应剩 `_defaults_loader.py` 中 `path_for` 函数本身的定义 ✓
- [x] 3.4 `grep -rEn "\bcompute_deviation_columns\b" evtrade/strategies/channel_deviation.py` 应只剩 `_compute_deviation_columns` 的定义与调用 ✓
