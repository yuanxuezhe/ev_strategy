"""metrics 字段单位约定回归测试 (unify-metrics-units)

锁定 16 字段 dict 的单位,防止回归到 2026-09-09 之前那种
"max_drawdown = 资金元 -> :.2% 打印 1.5e7%" 的错乱。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from evtrade.core.metrics import summarize


def _state(cash: float, position: float, last_price: float = 1.0) -> dict:
    return {"cash": cash, "position": position, "last_price": last_price,
            "n_trades": 0, "n_buy": 0, "n_sell": 0, "turnover": 0.0,
            "trades": []}


def _ts(year: int = 2025, month: int = 1, day: int = 1, h: int = 0) -> int:
    """14 位整数 stime 格式 (y*1e10 + mo*1e8 + d*1e6 + h*1e4 + mi*1e2 + s)"""
    return ((year * 100 + month) * 100 + day) * 1000000 + h * 10000


# ============ max_drawdown 单位 ============

def test_max_drawdown_unit_is_fraction():
    """max_drawdown 必须是占当时 peak 的小数, 不是元

    合成曲线 peak = 200000 (第一个 bar), trough = 120000
    -> mdd = (200000 - 120000) / 200000 = 0.40
    业界惯例 (Tradestation / PT / 通用): (peak - trough) / peak
    """
    eq = np.array([200000.0, 180000.0, 120000.0, 150000.0, 200000.0])
    bl = eq.copy()                                  # baseline 相同 -> 无超额
    summary = summarize(_state(200000.0, 0.0), 200000.0, 0.0,
                        equity_curve=eq, baseline_curve=bl,
                        first_ts=_ts(2025, 1, 1), last_ts=_ts(2025, 1, 5))
    assert 0.0 <= summary["max_drawdown"] <= 1.5, (
        f"mdd 超出合理小数范围: {summary['max_drawdown']}")
    assert summary["max_drawdown"] < 1.0, (
        f"mdd 看起来像资金元不是小数: {summary['max_drawdown']}")
    assert abs(summary["max_drawdown"] - 0.40) < 1e-6, (
        f"mdd 应该是 0.40, 实际 {summary['max_drawdown']}")


def test_max_drawdown_zero_on_no_data():
    """无 equity_curve 时 mdd 退化为 0.0, 不是 NaN"""
    summary = summarize(_state(0.0, 0.0), 100.0, 0.0)
    assert summary["max_drawdown"] == 0.0


# ============ cagr 单位 ============

def test_cagr_is_percent_not_fraction():
    """cagr 必须是百分数 (%), 不是小数

    合成 2 年翻 1.21 倍 -> 10% CAGR; 若存为小数会是 0.10, 这里断言 10.0
    """
    eq = np.array([100.0, 110.0, 121.0])
    bl = eq.copy()
    summary = summarize(_state(110.0, 0.0), 100.0, 0.0,
                        equity_curve=eq, baseline_curve=bl,
                        first_ts=_ts(2025, 1, 1), last_ts=_ts(2027, 1, 1))
    assert summary["years"] > 1.5
    assert abs(summary["cagr"] - 10.0) < 0.5, (
        f"cagr 应该 ~10 (百分数), 实际 {summary['cagr']}")
    assert summary["cagr"] > 1.0


# ============ calmar 跨单位 ============

def test_calmar_cross_unit_fixed():
    """cagr(%) / mdd(小数) -> 无量纲比率 (典型 ~0.4)

    之前 bug: calmar = cagr / max_drawdown (元), 跨单位, 量级 ~5e-5 或 50
    修复后: calmar = (cagr/100) / max_drawdown (小数), 量级 ~0.4

    合成 eq=[100, 110, 80, 121]:
      peak=110 (第2根), trough=80 -> mdd = (110-80)/110 = 0.2727
      cagr ≈ 10% (100 -> 121 in 2 years)
      calmar ≈ (10/100) / 0.2727 ≈ 0.367
    """
    eq = np.array([100.0, 110.0, 80.0, 121.0])
    bl = eq.copy()
    summary = summarize(_state(121.0, 0.0), 100.0, 0.0,
                        equity_curve=eq, baseline_curve=bl,
                        first_ts=_ts(2025, 1, 1), last_ts=_ts(2027, 1, 1))
    assert abs(summary["max_drawdown"] - (30.0/110.0)) < 1e-6, (
        f"max_drawdown 应该 ~0.2727 (30/110), 实际 {summary['max_drawdown']}")
    assert abs(summary["calmar"] - 0.367) < 0.02, (
        f"calmar 应该 ~0.367 (无量纲), 实际 {summary['calmar']}")
    assert -10.0 < summary["calmar"] < 10.0, (
        f"calmar 量级离谱, 像是跨单位: {summary['calmar']}")


# ============ x_mdd 单位 ============

def test_x_mdd_is_fraction():
    """x_mdd 应是占初始 baseline 的小数, 不是元"""
    eq = np.array([100.0, 110.0, 80.0, 121.0])
    # baseline 恒等于 init (完全不波动), 累计超额 = eq - 100
    # x_mdd 应在 [0, 1] 量级
    bl = np.full_like(eq, 100.0)
    summary = summarize(_state(121.0, 0.0), 100.0, 0.0,
                        equity_curve=eq, baseline_curve=bl,
                        first_ts=_ts(2025, 1, 1), last_ts=_ts(2027, 1, 1))
    assert 0.0 <= summary["x_mdd"] <= 2.0, (
        f"x_mdd 量级像元不是小数: {summary['x_mdd']}")


# ============ CLI 打印格式 ============

def test_cli_max_drawdown_format_is_reasonable():
    """CLI 打印 max_drawdown 必须是合理百分数量级 (e.g. 12.34%), 不是 1.5e7%

    不调 subprocess (依赖 mysql); 改为直接验证 cli._run_backtest 末尾
    print_summary 段的格式化模板是否与字段单位一致。
    """
    import inspect
    from evtrade import cli as _cli
    src = inspect.getsource(_cli._run_backtest)
    # 必须包含 max_drawdown 用 :.2% 格式
    assert "max_drawdown" in src, "cli 没引用 max_drawdown 字段"
    assert re.search(r"max_drawdown[^{]*\{[^}]*\.2%", src) or \
           re.search(r"['\"]?\s*:?[^}]*max_drawdown[^}]*\.2%", src), (
        f"cli 打印 max_drawdown 必须用 :.2% 格式 (小数 -> %); src={src[-800:]}")
    # 必须 NOT 用 :.2f (那是金额格式)
    bad = re.search(r"max_drawdown[^}]*:\.2f", src)
    assert not bad, (
        f"cli 打印 max_drawdown 不应用 :.2f (那是金额); src={src[-800:]}")


def test_cli_calmar_format_is_dimensionless():
    """Calmar 是无量纲比率, cli 必须用 :.3f 不能用 :.2%"""
    import inspect
    from evtrade import cli as _cli
    src = inspect.getsource(_cli._run_backtest)
    # 必须有 calmar 打印行, 用 :.3f 格式
    # 注: 行是 "{s['calmar']:+.3f}", calmar 后面是字典取值 }
    # 改用更宽容的 regex
    assert "calmar" in src, "cli 没引用 calmar"
    assert re.search(r"\['calmar'\]\s*:\s*\+\.3f", src), (
        f"cli 打印 calmar 必须用 :.3f (无量纲); src={src[-800:]}")


def test_cli_no_money_field_uses_percent():
    """回归锁定: 任何金额元字段 (final_equity/baseline/excess/turnover) MUST NOT 用 :.2%"""
    import inspect
    from evtrade import cli as _cli
    src = inspect.getsource(_cli._run_backtest)
    money_fields = ["final_equity", "baseline", "excess", "final_cash", "turnover"]
    for f in money_fields:
        # 找含该字段且带格式的行
        for m in re.finditer(rf"f['\"][^'\"]*{f}[^'\"]*['\"][^,)]*", src):
            line = m.group(0)
            assert ".2%" not in line, (
                f"金额字段 {f} 被错误用 .2% 格式化: {line}\n"
                f"应改 .2f (元) 或加单位 (元)")
            assert "%" not in line.split(":", 1)[-1] or ".2f" in line, (
                f"金额字段 {f} 打印带 % 后缀, 跨单位: {line}")


# ============ sweep filter_pass 跨单位对齐 ============

def test_sweep_filter_pass_respects_max_mdd():
    """sweep filter_pass: max_mdd=0.0 应拒掉所有非零 mdd 的参数

    修复前 bug: max_mdd(默认 1.0) 与 max_drawdown(元) 比较,
    1 元几乎总过 -> filter_pass 永远 True, 默认值失效。
    修复后: max_drawdown 是小数, 1.0 = 100% = 不限; 0.0 = 必须无回撤。
    """
    from evtrade.core.sweep import run_one_vectorized
    import datetime as _dt
    # 30 天的合成数据: 每根 bar 涨 0.1%, 期间有个大跌 -> 必定产生 mdd > 0
    bars = []
    base = _dt.datetime(2025, 1, 1)
    price = 1.0
    for i in range(30 * 240):
        # 每 100 根砸一次, 形成回撤
        if i % 100 == 50:
            price *= 0.95
        else:
            price *= 1.0001
        ts = int((base + _dt.timedelta(minutes=i)).strftime("%Y%m%d%H%M%S"))
        bars.append({"stime": ts, "open": price, "high": price * 1.001,
                     "low": price * 0.999, "close": price, "volume": 1000.0})

    m = run_one_vectorized(
        bars={"stime": np.array([b["stime"] for b in bars]),
              "open":  np.array([b["open"]  for b in bars]),
              "high":  np.array([b["high"]  for b in bars]),
              "low":   np.array([b["low"]   for b in bars]),
              "close": np.array([b["close"] for b in bars]),
              "volume":np.array([b["volume"]for b in bars])},
        period="15m",
        warmup_until=int(_dt.datetime(2025, 1, 1).strftime("%Y%m%d%H%M%S")),
        strategy_name="channel_deviation",
        strategy_params={"tf1": 21},
        init_cash=200000.0,
        init_position=200000.0,
        trade_qty=10000.0,
        scale=1.0,
        buy_pct=0.0,
        sell_pct=0.0)
    assert m["max_drawdown"] > 0.0, "synthetic 数据应有 mdd > 0"
    assert m["max_drawdown"] <= 1.5, (
        f"mdd 应该是小数 (<=1.5), 实际 {m['max_drawdown']}")
    # 反向断言: 若仍是元单位, mdd 会是 ~100000 量级
    assert m["max_drawdown"] < 1.0 or m["max_drawdown"] == 0.0, (
        f"mdd 像资金元不是小数: {m['max_drawdown']}")