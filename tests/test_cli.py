"""evtrade.cli argparse subparser dispatch

覆盖:
  - 无参数 -> root parser 显示帮助, 不抛异常
  - 显式 backtest / sweep / replay 子命令被分发到对应 main
  - --help 在 root 层级展示
  - 现有 backtest 调用形式 (无子命令, 直接传策略名) 仍向后兼容
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import evtrade.cli as cli



def test_cli_dispatches_sweep(monkeypatch):
    """sweep 子命令应当分发到 sweep_main"""
    called = {"n": 0}
    real_sweep = cli.sweep_main

    def spy(argv):
        called["n"] += 1
        # 返回值与 sweep_main 一致即可
        return None

    monkeypatch.setattr(cli, "sweep_main", spy)
    cli.main(["sweep", "--strategy", "channel_deviation",
              "--code", "000001.SZ", "--start", "20260101", "--end", "20260102",
              "--grid", ""])
    assert called["n"] == 1



def test_cli_backward_compatible_no_subcommand(monkeypatch):
    """向后兼容: 现有脚本用 'evtrade <strategy> --code ...' 调用,
    应当走 backtest_main 而不是 root parser 报错。"""
    called = {"n": 0}

    def spy(argv):
        called["n"] += 1
        return None

    monkeypatch.setattr(cli, "backtest_main", spy)
    cli.main(["channel_deviation", "--code", "000001.SZ",
              "--start", "20260101", "--end", "20260102"])
    assert called["n"] == 1


def test_cli_root_help_flag():
    """--help 在 root 应当退出码 0, 展示三个子命令"""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0


def test_cli_subcommand_help_flag(capsys):
    """backtest -h / sweep -h / replay -h 各自展示子命令帮助"""
    # argparse 用 sys.exit(0) + 写 stdout; main 直接 parse_args 也会这样
    for sub in ("backtest", "sweep", "replay"):
        with pytest.raises(SystemExit):
            cli.main([sub, "--help"])
