"""共享 pytest fixture.

之前分散在多处的 fixture 集中到 conftest.py:
  - isolated_defaults: 把 EVTRADE_DEFAULTS_DIR 指向临时目录 (3 个测试文件原各写一份)
"""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_defaults(tmp_path, monkeypatch):
    """把 EVTRADE_DEFAULTS_DIR 指向隔离临时目录, 避免落盘写到仓库默认参数目录。

    同时清掉 DSL runner 缓存 (_RUNNER_BY_KEY), 避免其它测试残留的 runner
    影响本测试的默认参数解析 (test_cli_params 原需要此清理; 对其它测试无害)。
    """
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(tmp_path))
    from evtrade.strategies.dsl import _RUNNER_BY_KEY
    _RUNNER_BY_KEY.clear()
    return tmp_path
