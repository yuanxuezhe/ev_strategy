"""共享 pytest fixture

isolated_defaults: 把 EVTRADE_DEFAULTS_DIR 指向临时目录,
避免落盘写到仓库默认参数目录。
"""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_defaults(tmp_path, monkeypatch):
    """把 EVTRADE_DEFAULTS_DIR 指向隔离临时目录, 避免落盘写到仓库默认参数目录。"""
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(tmp_path))
    return tmp_path