"""_resolve_strategy_params 优先级

覆盖 (2026-09 重构后, 旧 CLI kwargs 兼容已删除):
  1. CLI --params 显式 > _defaults 落盘
  2. _defaults 落盘 (无 --params 时)
  3. 都不存在时返回 {}
"""
from __future__ import annotations

import pytest

from evtrade.cli import _resolve_strategy_params, _parse_params


# ---------- 优先级 ----------


def test_explicit_params_overrides_defaults(monkeypatch):
    """CLI --params 显式最高优先级, 不读 _defaults"""
    import tempfile
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", td)
        (Path(td) / "channel_deviation.json").write_text(json.dumps(
            {"strategy": "channel_deviation",
             "params": {"low1": 9.9, "low2": 9.9, "high1": 9.9, "high2": 9.9}})
        )

        got = _resolve_strategy_params(
            "channel_deviation", "low1:1.5;low2:1.0;high1:1.5;high2:0.5")
        assert got == {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}


def test_defaults_used_when_no_explicit_params(monkeypatch):
    """无 --params 时回退到 _defaults 落盘"""
    import tempfile, json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", td)
        (Path(td) / "channel_deviation.json").write_text(json.dumps(
            {"strategy": "channel_deviation",
             "params": {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}})
        )
        got = _resolve_strategy_params("channel_deviation", "")
        assert got == {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}


def test_empty_when_nothing_provided(monkeypatch):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", td)
        got = _resolve_strategy_params("dev_trigger", "")
        assert got == {}


def test_load_defaulted_params_used_even_when_strategy_differs():
    """_defaults 落盘对所有策略生效, 不止 channel_deviation"""
    import tempfile, json, os
    from pathlib import Path
    old = os.environ.get("EVTRADE_DEFAULTS_DIR")
    try:
        with tempfile.TemporaryDirectory() as td:
            os.environ["EVTRADE_DEFAULTS_DIR"] = td
            (Path(td) / "dev_trigger.json").write_text(json.dumps(
                {"strategy": "dev_trigger",
                 "params": {"entry_dev": 1.2}}))
            got = _resolve_strategy_params("dev_trigger", "")
            assert got == {"entry_dev": 1.2}
    finally:
        if old is None:
            os.environ.pop("EVTRADE_DEFAULTS_DIR", None)
        else:
            os.environ["EVTRADE_DEFAULTS_DIR"] = old


def test_parse_params_types():
    """_parse_params 类型自动推导 (int/float/bool/str)"""
    assert _parse_params("a:1;b:2.5;c:true;d:hello") == {
        "a": 1, "b": 2.5, "c": True, "d": "hello"}
    # 空字符串 / 空键 -> {}
    assert _parse_params("") == {}
    # 缺冒号报错
    import pytest as _p
    with _p.raises(ValueError):
        _parse_params("a=1")