"""_resolve_strategy_params 优先级与回测/sweep 默认参数接入

覆盖:
  1. CLI --params 显式 > _defaults 落盘 > 旧 CLI kwargs 兼容 > 空
  2. _defaults 落盘 > 旧 CLI kwargs (显式 --params 不传时)
  3. 旧 CLI kwargs 仅 channel_deviation 兼容, 其它策略不接收
  4. 都不存在时返回 {}
"""
from __future__ import annotations

import pytest

from evtrade.cli import _resolve_strategy_params


# ---------- 优先级 ----------


def test_explicit_params_overrides_defaults(monkeypatch):
    """CLI --params 显式最高优先级, 不读 _defaults"""
    from evtrade.cli import _parse_params
    # 假设 _defaults 已落盘 (本测试通过 monkeypatch env 隔离)
    import tempfile
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", td)
        (Path(td) / "channel_deviation.json").write_text(json.dumps(
            {"strategy": "channel_deviation",
             "params": {"low1": 9.9, "low2": 9.9, "high1": 9.9, "high2": 9.9}})
        )

        # 显式 --params 应该胜出
        got = _resolve_strategy_params(
            "channel_deviation", "low1:1.5;low2:1.0;high1:1.5;high2:0.5",
            legacy_low1=2.5, legacy_low2=2.5,
            legacy_high1=2.5, legacy_high2=2.5)
        assert got == {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}


def test_defaults_override_legacy_kwargs(monkeypatch):
    """_defaults 落盘 > 旧 CLI kwargs"""
    import tempfile, json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", td)
        (Path(td) / "channel_deviation.json").write_text(json.dumps(
            {"strategy": "channel_deviation",
             "params": {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}})
        )
        # 没传 --params, 旧 kwargs 应被忽略 (因为落盘存在)
        got = _resolve_strategy_params(
            "channel_deviation", "",
            legacy_low1=2.5, legacy_low2=2.5,
            legacy_high1=2.5, legacy_high2=2.5)
        assert got == {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}


def test_legacy_kwargs_used_when_no_defaults(monkeypatch):
    """没有 _defaults 也没有 --params, 旧 CLI kwargs 兜底 (channel_deviation)"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", td)
        got = _resolve_strategy_params(
            "channel_deviation", "",
            legacy_low1=1.5, legacy_low2=1.0,
            legacy_high1=1.5, legacy_high2=0.5)
        assert got == {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}


def test_empty_when_nothing_provided(monkeypatch):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", td)
        got = _resolve_strategy_params("dev_trigger", "")
        assert got == {}


def test_legacy_kwargs_only_for_channel_deviation(monkeypatch):
    """旧 kwargs 不应用于其它策略 (避免误传到 dev_trigger)"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", td)
        # 非 channel_deviation + 无 --params + 无 _defaults -> 空
        got = _resolve_strategy_params(
            "dev_trigger", "",
            legacy_low1=1.5, legacy_low2=1.0,
            legacy_high1=1.5, legacy_high2=0.5)
        assert got == {}


def test_load_defaulted_params_used_even_when_strategy_differs():
    """_defaults 落盘对所有策略生效, 不止 channel_deviation"""
    import tempfile, json
    from pathlib import Path
    # 不需要 monkeypatch fixture, 临时目录即可
    import os
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
