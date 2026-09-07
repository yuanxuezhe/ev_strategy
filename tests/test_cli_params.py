"""evtrade.cli 'params' 子命令 (save / show / list)

覆盖:
  - params save channel_deviation --params "k:v;..." 写文件
  - params save channel_deviation --from-csv <csv> --rank 1 读第一名
  - params show 打印落盘参数
  - params show 缺默认时报清晰错误 (exit 1)
  - params list 空 / 有
  - 落盘文件路径在 evtrade/strategies/_defaults/<name>.json
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import evtrade.cli as cli
from evtrade.strategies import _defaults_loader as dl


# ---------- 测试夹具: isolated_defaults 由 tests/conftest.py 提供 ----------


# ---------- save / show / list 内部调用 ----------

def test_params_main_save_from_args(isolated_defaults):
    rc = cli.params_main(["save", "channel_deviation",
                          "--params", "low1:1.5;low2:1.0;high1:1.5;high2:0.5"])
    assert rc is None
    # 文件应当落在 tmp_path/channel_deviation.json
    p = isolated_defaults / "channel_deviation.json"
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["strategy"] == "channel_deviation"
    assert data["params"] == {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}
    assert data["source"] == {"kind": "from_params"}


def test_params_main_save_from_csv_rank_1(isolated_defaults, tmp_path):
    # 写一个临时 CSV
    csv_p = tmp_path / "sweep.csv"
    with csv_p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["low1", "low2", "high1", "high2",
                                          "score"])
        w.writeheader()
        # 第一行 score 高 -> rank=1 取它
        w.writerow({"low1": "2.0", "low2": "0.8", "high1": "2.0", "high2": "0.4",
                    "score": "0.10"})
        # 第二行 score 低
        w.writerow({"low1": "1.0", "low2": "1.0", "high1": "1.0", "high2": "0.5",
                    "score": "0.02"})

    rc = cli.params_main(["save", "channel_deviation",
                          "--from-csv", str(csv_p), "--rank", "1"])
    assert rc is None
    data = json.loads((isolated_defaults / "channel_deviation.json")
                      .read_text(encoding="utf-8"))
    assert data["params"]["low1"] == 2.0
    assert data["source"] == {"kind": "from_csv",
                              "csv": str(csv_p), "rank": 1}


def test_params_main_save_rank_out_of_range(isolated_defaults, tmp_path):
    csv_p = tmp_path / "sweep.csv"
    with csv_p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["low1", "score"])
        w.writeheader()
        w.writerow({"low1": "1.5", "score": "0.05"})

    rc = cli.params_main(["save", "channel_deviation",
                          "--from-csv", str(csv_p), "--rank", "5"])
    assert rc == 1  # 越界, 退出码 1


def test_params_main_show_existing(isolated_defaults, capsys):
    cli.params_main(["save", "channel_deviation",
                     "--params", "low1:1.5;low2:1.0;high1:1.5;high2:0.5"])
    cli.params_main(["show", "channel_deviation"])
    out = capsys.readouterr().out
    assert "low1" in out
    assert "1.5" in out
    assert "channel_deviation" in out


def test_params_main_show_missing(isolated_defaults, capsys):
    rc = cli.params_main(["show", "channel_deviation"])
    assert rc == 1
    err = capsys.readouterr()
    assert "[错误]" in err.out or "[错误]" in err.err
    assert "channel_deviation" in err.out + err.err


def test_params_main_list_empty(isolated_defaults, capsys):
    cli.params_main(["list"])
    out = capsys.readouterr().out
    assert "(无默认参数文件" in out


def test_params_main_list_populated(isolated_defaults, capsys):
    cli.params_main(["save", "channel_deviation",
                     "--params", "low1:1.5;low2:1.0;high1:1.5;high2:0.5"])
    cli.params_main(["list"])
    out = capsys.readouterr().out
    assert "channel_deviation" in out


# ---------- root dispatch ----------

def test_root_dispatch_params(monkeypatch, isolated_defaults):
    """'evtrade params save channel_deviation ...' 应被分发到 params_main"""
    called = {"n": 0}
    real = cli.params_main

    def spy(argv):
        called["n"] += 1
        return real(argv)

    monkeypatch.setattr(cli, "params_main", spy)
    cli.main(["params", "list"])
    assert called["n"] == 1


def test_root_help_mentions_params(capsys):
    """'evtrade --help' 应展示 params 子命令"""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "params" in out
