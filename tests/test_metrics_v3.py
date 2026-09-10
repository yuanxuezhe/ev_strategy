"""metrics.summarize v3 字段集 (25) 锁定 (2026-09-10 扩展)

覆盖:
  - 持仓行为: win_rate / profit_factor / avg_pnl / max_consecutive_*
              / avg_hold_bars / max_hold_bars
  - 信息比率: ir
  - 基准对比: baseline_max_dd / dd_excess
  - 修正: max_dd_recovered 三档 / x_mdd 删除 / cagr_excess 替换 ann_excess_pct
"""
from __future__ import annotations

import math

import numpy as np

from evtrade.core.metrics import summarize


# ============ helpers ============

def _state(cash=100000.0, position=100000.0, last_price=1.0,
           n_trades=4, n_buy=2, n_sell=2, turnover=200000.0):
    return {"cash": cash, "position": position, "last_price": last_price,
            "n_trades": n_trades, "n_buy": n_buy, "n_sell": n_sell,
            "turnover": turnover}


def _trades_win2_loss1():
    """2 笔盈利 + 1 笔亏损 (3 笔配对)"""
    return [
        {"ts": 20260101090000, "side": "BUY",  "qty": 10000.0, "price": 1.000},
        {"ts": 20260101093000, "side": "SELL", "qty": 10000.0, "price": 1.020},
        {"ts": 20260101100000, "side": "BUY",  "qty": 10000.0, "price": 1.015},
        {"ts": 20260101103000, "side": "SELL", "qty": 10000.0, "price": 1.030},
        {"ts": 20260101110000, "side": "BUY",  "qty": 10000.0, "price": 1.020},
        {"ts": 20260101113000, "side": "SELL", "qty": 10000.0, "price": 1.015},
    ]


def _trades_2wins_then_1_loss_then_1_win():
    """WWL W: 最大连续盈利=2 (开头), 最大连续亏损=1 (中间)"""
    return [
        {"ts": 20260101090000, "side": "BUY",  "qty": 10000.0, "price": 1.000},
        {"ts": 20260101093000, "side": "SELL", "qty": 10000.0, "price": 1.020},
        {"ts": 20260101100000, "side": "BUY",  "qty": 10000.0, "price": 1.020},
        {"ts": 20260101103000, "side": "SELL", "qty": 10000.0, "price": 1.040},
        {"ts": 20260101110000, "side": "BUY",  "qty": 10000.0, "price": 1.030},
        {"ts": 20260101113000, "side": "SELL", "qty": 10000.0, "price": 1.020},
        {"ts": 20260101120000, "side": "BUY",  "qty": 10000.0, "price": 1.020},
        {"ts": 20260101123000, "side": "SELL", "qty": 10000.0, "price": 1.040},
    ]


def _trades_long_hold():
    """2 笔, 不同持仓周期: 30min 和 60min (5m 桶 => 6 桶 / 12 桶)"""
    return [
        {"ts": 20260101090000, "side": "BUY",  "qty": 10000.0, "price": 1.000},
        {"ts": 20260101093000, "side": "SELL", "qty": 10000.0, "price": 1.020},
        {"ts": 20260101100000, "side": "BUY",  "qty": 10000.0, "price": 1.020},
        {"ts": 20260101110000, "side": "SELL", "qty": 10000.0, "price": 1.040},
    ]


# ============ 持仓行为 ============

def test_win_rate_profit_factor_avg_pnl():
    trades = _trades_win2_loss1()
    s = summarize(_state(), 100000, 100000, trades=trades)
    # 3 笔配对: +200, +150, -50 = 净 +300; 胜率=2/3
    assert math.isclose(s["win_rate"], 2 / 3, rel_tol=1e-9)
    # profit_factor = (200+150) / 50 = 7.0
    assert math.isclose(s["profit_factor"], 7.0, rel_tol=1e-9)
    # avg_pnl = 300 / 3 = 100
    assert math.isclose(s["avg_pnl"], 100.0, rel_tol=1e-9)


def test_max_consecutive_wins_losses():
    trades = _trades_2wins_then_1_loss_then_1_win()
    s = summarize(_state(), 100000, 100000, trades=trades)
    # +200, +200, -100, +200 => 最大连盈=2 (开头), 最大连亏=1
    assert s["max_consecutive_wins"] == 2
    assert s["max_consecutive_losses"] == 1


def test_avg_max_hold_bars():
    trades = _trades_long_hold()
    s = summarize(_state(), 100000, 100000, trades=trades)
    # 30min / 5min = 6 桶; 60min / 5min = 12 桶
    assert s["max_hold_bars"] == 12
    assert math.isclose(s["avg_hold_bars"], 9.0, rel_tol=1e-9)


def test_no_trades_returns_zeros():
    s = summarize(_state(n_trades=0, n_buy=0, n_sell=0, turnover=0),
                  100000, 100000, trades=[])
    assert s["win_rate"] == 0.0
    assert s["profit_factor"] == 0.0
    assert s["avg_pnl"] == 0.0
    assert s["max_consecutive_wins"] == 0
    assert s["max_consecutive_losses"] == 0
    assert s["avg_hold_bars"] == 0.0
    assert s["max_hold_bars"] == 0


# ============ 回撤 ============

def test_max_dd_recovered_recovered():
    """trough 后能恢复: max_dd_recovered = recovery_idx - trough_idx

    单位换算: max_drawdown 现在是占 peak 的小数 (= 30/110 = 0.2727)
    """
    # 5 个 bar 序列: [100, 110, 90(peak前), 80(trough), 100(恢复), 110]
    eq = np.array([100.0, 110.0, 90.0, 80.0, 100.0, 110.0])
    # running_peak = [100, 110, 110, 110, 110, 110]
    # drawdown    = [0,    0, -20, -30, -10,   0]  (绝对)
    # 归一:        [0,    0, -2/11, -3/11, -1/11, 0]
    # trough_idx=3, peak_idx=1, eq[5]=110 >= 110 (peak_at_trough) => recovered at i=5
    # max_dd_recovered = 5 - 3 = 2
    s = summarize(_state(), 100, 0,
                  equity_curve=eq, baseline_curve=np.zeros_like(eq),
                  first_ts=20260101090000, last_ts=20260101093000)
    assert math.isclose(s["max_drawdown"], 30.0 / 110.0, abs_tol=1e-9)
    assert s["max_dd_recovered"] == 2


def test_max_dd_recovered_never_recovered():
    """trough 后从未恢复: max_dd_recovered = -1 (sentinel)

    单位: max_drawdown = 40/110 = 0.3636
    """
    # [100, 110, 90, 80, 70] 末端持续低于前高 110
    eq = np.array([100.0, 110.0, 90.0, 80.0, 70.0])
    s = summarize(_state(), 100, 0,
                  equity_curve=eq, baseline_curve=np.zeros_like(eq),
                  first_ts=20260101090000, last_ts=20260101093000)
    assert math.isclose(s["max_drawdown"], 40.0 / 110.0, abs_tol=1e-9)
    assert s["max_dd_recovered"] == -1


def test_max_dd_recovered_immediate_recovery():
    """trough 下一根就恢复"""
    eq = np.array([100.0, 110.0, 90.0, 80.0, 110.0])
    # trough_idx=3, peak_at_trough=110 (peak at i=1), eq[4]=110 >= 110
    # recovery_idx=4, max_dd_recovered = 4-3 = 1
    s = summarize(_state(), 100, 0,
                  equity_curve=eq, baseline_curve=np.zeros_like(eq),
                  first_ts=20260101090000, last_ts=20260101093000)
    assert s["max_dd_recovered"] == 1


# ============ 基准对比 ============

def test_baseline_max_dd_and_dd_excess():
    """baseline 也有回撤, dd_excess = 策略回撤 - baseline 回撤

    单位: max_drawdown / baseline_max_dd / dd_excess 均为小数 (占 peak 比例)
    """
    # 策略 equity: [100, 110, 95, 90, 105]  -> max_dd = (110-90)/110 = 0.1818
    eq = np.array([100.0, 110.0, 95.0, 90.0, 105.0])
    # baseline: [100, 105, 95, 80, 90]  -> max_dd = (105-80)/105 = 0.2381
    bl = np.array([100.0, 105.0, 95.0, 80.0, 90.0])
    s = summarize(_state(), 100, 0,
                  equity_curve=eq, baseline_curve=bl,
                  first_ts=20260101090000, last_ts=20260101093000)
    assert math.isclose(s["max_drawdown"], 20.0 / 110.0, abs_tol=1e-9)
    assert math.isclose(s["baseline_max_dd"], 25.0 / 105.0, abs_tol=1e-9)
    # 策略 dd_excess = 0.1818 - 0.2381 = -0.0563 (策略比基准回撤更浅)
    assert math.isclose(s["dd_excess"],
                        20.0/110.0 - 25.0/105.0, abs_tol=1e-9)


def test_baseline_max_dd_no_drawdown():
    """baseline 单调上升: max_dd=0"""
    eq = np.array([100.0, 90.0, 100.0, 110.0])
    bl = np.array([100.0, 105.0, 110.0, 115.0])  # 单调上升
    s = summarize(_state(), 100, 0,
                  equity_curve=eq, baseline_curve=bl,
                  first_ts=20260101090000, last_ts=20260101093000)
    assert s["baseline_max_dd"] == 0.0
    # 策略回撤: (100-90)/100 = 0.10
    assert math.isclose(s["max_drawdown"], 0.10, abs_tol=1e-9)


# ============ 信息比率 ============

def test_ir_with_known_excess():
    """已知 excess 序列, 验证 IR 数值"""
    # equity 单调 +1%/bar, baseline 不变 => excess_rets 全部 ≈ +0.01
    n = 252
    eq = np.array([100.0 * (1.01 ** i) for i in range(n)])
    bl = np.full(n, 100.0)
    s = summarize(_state(), 100, 0,
                  equity_curve=eq, baseline_curve=bl,
                  first_ts=20260101090000, last_ts=20270101090000)  # 1 年
    # 年化超额约 252 * 1% ≈ 252%; sharpe / IR: 全正无下行 -> 数值很大但 > 0
    assert s["ir"] > 0.0
    assert s["ir"] == s["sharpe_excess"]  # rf=0 时 ir ≡ sharpe_excess


# ============ 重命名字段 ============

def test_x_mdd_present():
    """x_mdd = 累计超额曲线回撤 (unify-metrics-units 保留字段; 单位 = 小数)"""
    s = summarize(_state(), 100, 100,
                  equity_curve=np.array([100.0, 105.0]),
                  baseline_curve=np.array([100.0, 100.0]),
                  first_ts=20260101090000, last_ts=20260101093000)
    assert "x_mdd" in s
    # 累计超额曲线单调上升时 x_mdd=0
    assert s["x_mdd"] == 0.0


def test_cagr_excess_present():
    """ann_excess_pct 已重命名为 cagr_excess"""
    s = summarize(_state(), 100, 100,
                  equity_curve=np.array([100.0, 110.0]),
                  baseline_curve=np.array([100.0, 105.0]),
                  first_ts=20260101090000, last_ts=20270101090000)
    assert "cagr_excess" in s
    assert "ann_excess_pct" not in s


# ============ 字段集完整性 ============

REQUIRED_KEYS = {
    "final_price", "final_cash", "final_position", "final_equity", "baseline",
    "n_trades", "n_buy", "n_sell", "turnover", "excess_pct",
    "years", "cagr_excess",
    "cagr", "sharpe_excess", "sortino_excess", "calmar", "ir",
    "max_drawdown", "max_dd_days", "max_dd_recovered", "x_mdd",
    "win_rate", "profit_factor", "avg_pnl",
    "max_consecutive_wins", "max_consecutive_losses",
    "avg_hold_bars", "max_hold_bars",
    "baseline_max_dd", "dd_excess",
}


def test_field_set_exactly_26():
    """字段集恰好是这 26 个 (不依赖 equity 序列也能调通; 含 unify-units 保留的 x_mdd)"""
    s = summarize(_state(), 100, 100)
    assert set(s.keys()) == REQUIRED_KEYS, (
        f"差异:\n  缺: {REQUIRED_KEYS - set(s.keys())}\n"
        f"  多: {set(s.keys()) - REQUIRED_KEYS}")
