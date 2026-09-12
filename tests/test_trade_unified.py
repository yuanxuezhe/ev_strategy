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


# ---------- _TradeStateTracker (scale 倍投状态机) ----------


def test_trade_state_tracker_scale_and_reset():
    """scale>1 倍投 / 方向翻转重置 的契约

    重构前 SimulatedExecutor 与 vectorized_engine._execute_trades 各持一份
    同款 last_side/cur_qty 维护; 现在收敛到 _TradeStateTracker, 行为必须 bitwise 一致。
    """
    from evtrade.execution.base import _TradeStateTracker

    # 1) 首次 BUY 应返 qty
    t = _TradeStateTracker(qty=100, scale=2.0)
    assert t.next_qty(1) == 100.0

    # 2) 同方向 BUY → 100 * 2 = 200
    assert t.next_qty(1) == 200.0
    # 3) 同方向 BUY → 400
    assert t.next_qty(1) == 400.0

    # 4) 翻转 SELL → 重置为基础数量 100
    assert t.next_qty(-1) == 100.0
    assert t.last_side == -1

    # 5) 再 SELL → 100 * 2 = 200
    assert t.next_qty(-1) == 200.0

    # 6) scale=1.0 关闭倍投
    t2 = _TradeStateTracker(qty=100, scale=1.0)
    assert t2.next_qty(1) == 100.0
    assert t2.next_qty(1) == 100.0   # 倍率 1, 保持
    assert t2.next_qty(1) == 100.0
    # 翻转
    assert t2.next_qty(-1) == 100.0


def test_simulated_executor_uses_tracker():
    """SimulatedExecutor 内部应持有 _TradeStateTracker, 暴露接口不变"""
    from evtrade.execution import SimulatedExecutor
    from evtrade.execution.account import Account
    from evtrade.execution.base import _TradeStateTracker

    exe = SimulatedExecutor(Account(cash=100_000, position=0), qty=100, scale=2.0)
    assert isinstance(exe._qty_tracker, _TradeStateTracker)
    assert exe._qty_tracker.qty == 100
    assert exe._qty_tracker.scale == 2.0
    assert exe._qty_tracker.last_side == 0
    assert exe._qty_tracker.cur_qty == 100


def test_simulated_executor_scale_field_removed():
    """重构后 SimulatedExecutor 不再有 self.scale / self.last_side / self.cur_qty
    直接公开字段 (全部收敛到 _qty_tracker)."""
    from evtrade.execution import SimulatedExecutor
    from evtrade.execution.account import Account

    exe = SimulatedExecutor(Account(cash=100_000, position=0), qty=100, scale=2.0)
    for attr in ("scale", "last_side", "cur_qty"):
        assert not hasattr(exe, attr), (
            f"SimulatedExecutor 不应再有 self.{attr} (已并入 _qty_tracker)")
