"""SimulatedExecutor 不再硬编码 print 成交明细 (2026-09-12 cleanup)

历史: SimulatedExecutor.trade 内部曾有 `if self.verbose: print(f'>> BUY...')`,
framework 层硬编码中文文案与字段假设, 与 cli.py 的回放打印重复, 且违反
spec R 'Strategy display hooks are framework-agnostic' (framework 假定字段).
修法: SimulatedExecutor 仅调 trade_decision + Account.apply, 不 print.
CLI / Engine 仍是用户可见输出的来源.
"""
from __future__ import annotations

import io
import sys

from evtrade.execution import SimulatedExecutor
from evtrade.execution.account import Account


def test_simulated_executor_trade_does_not_print(capsys):
    """SimulatedExecutor.trade 触发 N 次成功成交, stdout/stderr 应为空"""
    account = Account(cash=100_000.0, position=0.0)
    exe = SimulatedExecutor(account, qty=100, buy_pct=1.0, sell_pct=1.0)

    # 一次买入
    assert exe.trade("BUY", price=10.0, ts="202601010930") is True
    # 一次卖出
    assert exe.trade("SELL", price=11.0, ts="202601011030") is True

    captured = capsys.readouterr()
    assert captured.out == "", (
        f"SimulatedExecutor.trade 不应 print 到 stdout, 实际: {captured.out!r}")
    assert captured.err == "", (
        f"SimulatedExecutor.trade 不应 print 到 stderr, 实际: {captured.err!r}")


def test_simulated_executor_trade_does_not_print_even_when_unfilled(capsys):
    """未成交也不应 print (历史 verbose=True 时也不会, 但保留回归)"""
    account = Account(cash=0.0, position=0.0)  # 无资金无法买入
    exe = SimulatedExecutor(account, qty=100, buy_pct=1.0, sell_pct=1.0)

    assert exe.trade("BUY", price=10.0, ts="202601010930") is False

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""