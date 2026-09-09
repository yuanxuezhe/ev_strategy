"""共享 pytest fixture.

之前分散在多处的 fixture 集中到 conftest.py:
  - isolated_defaults: 把 EVTRADE_DEFAULTS_DIR 指向临时目录 (3 个测试文件原各写一份)

DSL/numba/CUDA 已下线 (2026-09 重构), _RUNNER_BY_KEY 不再存在,
所以不再需要 clear runner 缓存。
"""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_defaults(tmp_path, monkeypatch):
    """把 EVTRADE_DEFAULTS_DIR 指向隔离临时目录, 避免落盘写到仓库默认参数目录。"""
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(tmp_path))
    return tmp_path