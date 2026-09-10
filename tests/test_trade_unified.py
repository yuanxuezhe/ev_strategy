"""trade_decision 单一成交实现: 与旧内联公式 bitwise 等价 (对照表)

双引擎路径 (SimulatedExecutor vs _execute_trades) 的逐笔一致性由
test_strategy_unified::test_vectorized_vs_engine_on_bars_reconcile 锁定。
"""
from __future__ import annotations

import numpy as np

from evtrade.execution.base import trade_decision


def _legacy_reference(side, price, cash, position, cur_qty, buy_pct, sell_pct):
    """旧 SimulatedExecutor.trade / _execute_trades 的内联公式 (bitwise 参考)"""
    if side > 0:
        if price > 0:
            max_by_cash = cash / price
            if buy_pct > 0:
                target = buy_pct * max_by_cash
                q = target if target < max_by_cash else max_by_cash
            else:
                q = cur_qty if cur_qty < max_by_cash else max_by_cash
        else:
            q = 0
        if q <= 0:
            return cash, position, 0.0, False
        return cash - q * price, position + q, q, True
    if sell_pct > 0:
        target = sell_pct * position
        q = target if target < position else position
    else:
        q = cur_qty if cur_qty < position else position
    if q <= 0:
        return cash, position, 0.0, False
    return cash + q * price, position - q, q, True


def test_trade_decision_matches_legacy_reference():
    rng = np.random.default_rng(7)
    for _ in range(4000):
        side = int(rng.choice([-1, 1]))
        price = float(rng.choice([0.0, 0.01, 1.0, 12.345, 100.0]))
        cash = float(rng.uniform(0, 5e5))
        position = float(rng.uniform(0, 5e5))
        cur_qty = float(rng.uniform(0, 2e5))
        buy_pct = float(rng.choice([0.0, 0.1, 0.5, 0.99, 1.0]))
        sell_pct = float(rng.choice([0.0, 0.25, 0.7, 1.0]))
        got = trade_decision(side, price, cash, position, cur_qty, buy_pct, sell_pct)
        want = _legacy_reference(side, price, cash, position, cur_qty, buy_pct, sell_pct)
        assert got[3] == want[3]
        assert got[:3] == want[:3], f"not bitwise equal: {got} vs {want}"
