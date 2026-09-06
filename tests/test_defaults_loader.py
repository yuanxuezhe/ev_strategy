"""evtrade.strategies._defaults_loader 行为"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import evtrade.strategies._defaults_loader as dl


@pytest.fixture
def isolated_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(tmp_path))
    return tmp_path


# ---------- 基础存在性 / 路径 ----------


def test_path_for_returns_under_defaults_dir():
    p = dl.path_for("channel_deviation")
    assert p.name == "channel_deviation.json"
    assert p.parent.name == dl.DEFAULT_DIR_NAME


def test_exists_false_when_no_file():
    assert dl.exists("definitely_not_a_strategy_xyz") is False


def test_list_defaulted_empty_when_dir_missing(tmp_path, monkeypatch):
    """目录不存在时 list_defaulted 返回 []"""
    nonexistent = tmp_path / "no_such_dir"
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(nonexistent))
    assert dl.list_defaulted() == []


# ---------- save / load ----------


def test_save_creates_file_and_load_returns_it(isolated_defaults):
    p = dl.save("channel_deviation", {"low1": 1.5, "low2": 1.0,
                                       "high1": 1.5, "high2": 0.5},
                source={"kind": "from_params"})
    assert p.is_file()
    assert p.parent == isolated_defaults

    data = dl.load("channel_deviation")
    assert data["strategy"] == "channel_deviation"
    assert data["params"] == {"low1": 1.5, "low2": 1.0,
                              "high1": 1.5, "high2": 0.5}
    assert data["source"] == {"kind": "from_params"}
    assert "saved_at" in data


def test_save_then_exists(isolated_defaults):
    assert dl.exists("channel_deviation") is False
    dl.save("channel_deviation", {"low1": 1.5})
    assert dl.exists("channel_deviation") is True


def test_save_rejects_non_dict_params(isolated_defaults):
    with pytest.raises(TypeError, match="dict"):
        dl.save("channel_deviation", "not a dict")


def test_load_missing_raises_with_helpful_message(isolated_defaults):
    with pytest.raises(FileNotFoundError, match="params save"):
        dl.load("does_not_exist")


def test_save_overwrites_existing(isolated_defaults):
    dl.save("channel_deviation", {"low1": 1.5})
    dl.save("channel_deviation", {"low1": 2.5})
    assert dl.load("channel_deviation")["params"]["low1"] == 2.5


def test_save_is_atomic_no_tmp_left(isolated_defaults):
    """save 后不应残留 *.tmp"""
    dl.save("channel_deviation", {"low1": 1.5})
    leftovers = list(isolated_defaults.glob("*.tmp"))
    assert leftovers == [], f"残留临时文件: {leftovers}"


# ---------- list_defaulted ----------


def test_list_defaulted_returns_saved_names(isolated_defaults):
    dl.save("channel_deviation", {"low1": 1.5})
    dl.save("dev_trigger", {"entry_dev": 0.5})
    names = dl.list_defaulted()
    assert names == ["channel_deviation", "dev_trigger"]


def test_list_defaulted_ignores_dotfiles_and_non_json(isolated_defaults):
    dl.save("channel_deviation", {"low1": 1.5})
    (isolated_defaults / "README.md").write_text("hi")
    (isolated_defaults / ".gitkeep").write_text("")
    assert dl.list_defaulted() == ["channel_deviation"]


# ---------- params_from_csv_row + _auto_cast ----------


def test_params_from_csv_row_casts_types():
    row = {"low1": "1.5", "low2": "1.0", "high1": "2", "high2": "true"}
    out = dl.params_from_csv_row(row, ["low1", "low2", "high1", "high2", "missing"])
    assert out == {"low1": 1.5, "low2": 1.0, "high1": 2, "high2": True}


def test_params_from_csv_row_skips_missing_keys():
    row = {"low1": "1.5"}
    out = dl.params_from_csv_row(row, ["low1", "low2"])
    assert out == {"low1": 1.5}


def test_params_from_csv_row_passes_through_non_string():
    """已是 Python 原生类型时 (来自 in-memory dict) 不强转"""
    row = {"low1": 1.5, "low2": 1}
    out = dl.params_from_csv_row(row, ["low1", "low2"])
    assert out == {"low1": 1.5, "low2": 1}


def test_auto_cast_string_variants():
    assert dl._auto_cast("true") is True
    assert dl._auto_cast("false") is False
    assert dl._auto_cast("42") == 42
    assert dl._auto_cast("1.5") == 1.5
    assert dl._auto_cast("hello") == "hello"


# ---------- _defaults_dir 环境变量切换 ----------


def test_defaults_dir_honors_env_var(tmp_path, monkeypatch):
    custom = tmp_path / "my_defaults"
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(custom))
    p = dl.save("x", {"a": 1})
    assert p.parent == custom
    assert (custom / "x.json").is_file()
