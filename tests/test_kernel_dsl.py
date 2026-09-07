"""步骤 2 测试: kernel._strategy_check 通用化 (channel_deviation 用 p0..p3 别名)

锁定: 72 项差分测试通过 (向后兼容);
       DSL 字段映射 (p0/p1/p2/p3 = low1/low2/high1/high2)
       新策略参数化路径正确
       DSL 渲染特化内核与冻结内核 bitwise 一致 (kernel_dsl)
"""
from __future__ import annotations

import numpy as np
import pytest

from evtrade.strategies import get_strategy, get_strategy_param_spec
from evtrade.core.kernel import (bars_to_arrays, make_state, run_backtest,
                                  summarize)
from evtrade.replay import replay_engine, replay_kernel


def _synthetic():
    from evtrade.data import synthetic_bars
    return synthetic_bars(days=30, start_ymd="20241101", seed=42)


# ============ KernelState p0..p3 别名正确性 ============

def test_kernelstate_p0p3_alias_low1high2():
    """KernelState 构造时若 p0..p3=0 默认, 用 low1..high2 填 p0..p3"""
    from evtrade.core.kernel import KernelState
    st = KernelState(
        period_seconds=300, warmup_until=0, tf1=21,
        low1=1.5, low2=1.0, high1=2.0, high2=0.8,
        init_cash=200000., init_position=200000., trade_qty=10000.,
        scale=1.0, buy_pct=0., sell_pct=0., all_in=False,
        record_trades=False, trade_cap=0,
    )
    # 默认映射: p0..p3 = low1..high2
    assert st.p0 == 1.5 and st.p1 == 1.0
    assert st.p2 == 2.0 and st.p3 == 0.8
    # low1..high2 别名仍然存在
    assert st.low1 == st.p0
    assert st.high2 == st.p3


def test_kernelstate_explicit_p0_overrides_low1():
    """显式传 p0..p3 时, 覆盖 low1..high2 (不再用 low1 填 p0)"""
    from evtrade.core.kernel import KernelState
    st = KernelState(
        period_seconds=300, warmup_until=0, tf1=21,
        low1=1.5, low2=1.0, high1=2.0, high2=0.8,
        init_cash=200000., init_position=200000., trade_qty=10000.,
        scale=1.0, buy_pct=0., sell_pct=0., all_in=False,
        record_trades=False, trade_cap=0,
        p0=9.9,  # 显式传 p0
    )
    assert st.p0 == 9.9
    assert st.low1 == 9.9   # low1 别名也跟着 p0
    # p1/p2/p3 未传, 仍默认 0; 不再映射到 low2/high1
    # (因为 p0 != 0, 触发"非默认"路径)
    assert st.p1 == 0.0
    # low2 仍是显式 1.0 (顶层参数, 独立于 p)
    assert st.low2 == 1.0
    # high1, high2 也是显式传
    assert st.high1 == 2.0 and st.high2 == 0.8


# ============ make_state 接受 p0..p15 ============

def test_make_state_with_explicit_p0p3():
    """make_state 显式传 p0..p3"""
    bars = bars_to_arrays(_synthetic())
    n = len(bars)
    st = make_state(period="5m", warmup_until=int("20241110")*1_000_000,
                    tf1=21, low1=1.5, low2=1.0, high1=2.0, high2=0.8,
                    init_cash=200000., init_position=200000., trade_qty=10000.,
                    p0=1.7, p1=0.9, p2=1.7, p3=0.6)  # 显式参数
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    s = summarize(st)
    assert s["n_trades"] >= 0


def test_make_state_no_p0p3_uses_low1high2():
    """make_state 不传 p0..p3, 自动用 low1..high2 填 (向后兼容)"""
    bars = bars_to_arrays(_synthetic())
    n = len(bars)
    st_a = make_state(period="5m", warmup_until=int("20241110")*1_000_000,
                    tf1=21, low1=1.5, low2=1.0, high1=1.5, high2=0.5)
    st_b = make_state(period="5m", warmup_until=int("20241110")*1_000_000,
                    tf1=21, low1=1.5, low2=1.0, high1=1.5, high2=0.5,
                    p0=1.5, p1=1.0, p2=1.5, p3=0.5)   # 显式相同
    run_backtest(st_a, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    run_backtest(st_b, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    # n_trades 应相同 (参数一致)
    assert summarize(st_a)["n_trades"] == summarize(st_b)["n_trades"]


# ============ kernel 路径与 ref 引擎 bitwise 一致 ============

def test_kernel_vs_ref_engine_bitwise():
    """kernel 路径 (用 p0..p3) 与 ref 引擎逐 bar 信号 + 成交笔数一致"""
    bars = _synthetic()
    warm = int("20241110")*1_000_000
    params = {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}
    # kernel 路径: make_state + run_backtest (记录成交)
    arr = bars_to_arrays(bars)
    st = make_state(period="5m", warmup_until=warm, tf1=21, **params,
                    init_cash=200000., init_position=200000., trade_qty=10000.,
                    record_trades=True, trade_cap=len(arr["stime"]))
    sig_k = np.zeros(len(arr["stime"]), np.int8)
    up = np.full(len(arr["stime"]), np.nan)
    dw = np.full(len(arr["stime"]), np.nan)
    run_backtest(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                 arr["close"], arr["volume"], sig_k, up, dw)
    # ref 路径: replay_engine (frozen 策略)
    rep = replay_engine(bars, "5m", warm, 21, **params,
                        init_cash=200000., init_position=200000.,
                        trade_qty=10000., scale=1.0)
    # 逐 bar 信号对齐 (rep 的 sig/up/dw 已是全 bar 对齐: 预热段为 0/NaN)
    assert np.array_equal(rep["sig"], sig_k), \
        "kernel 与 ref 引擎信号轨迹不一致"
    # 成交笔数一致 (两边同一执行语义; 信号数 != 成交数, 资金/持仓不足会跳过)
    assert len(rep["trades"]) == int(st.n_trades), \
        f"trade count mismatch: ref={len(rep['trades'])} ker={st.n_trades}"


# ============ 步骤 2: DSL 渲染特化内核 (kernel_dsl) ============

def test_strategy_has_dsl():
    """DSL 可渲染性探测: DSL 策略 True / 无 DSL 的 breakout False"""
    from evtrade.core.kernel_dsl import strategy_has_dsl
    assert strategy_has_dsl("channel_deviation") is True
    assert strategy_has_dsl("dev_trigger") is True
    assert strategy_has_dsl("breakout") is False
    with pytest.raises(ValueError):
        strategy_has_dsl("no_such_strategy")


def test_dsl_spliced_channel_deviation_bitwise():
    """DSL 渲染的 channel_deviation 特化内核 与 冻结 kernel 逐位一致

    这是渲染器忠实性的锁定: 特化模块 = kernel.py 源码整段替换策略段后 exec,
    若渲染改变了语义 (字段映射/表达式顺序/锁存时机), 这里立刻红。
    """
    from evtrade.core.kernel_dsl import build_dsl_kernel
    bars = bars_to_arrays(_synthetic())
    warm = int("20241110") * 1_000_000
    n = len(bars["stime"])

    # 冻结 kernel (low1..high2 路径)
    st_f = make_state(period="5m", warmup_until=warm, tf1=21,
                      low1=1.5, low2=1.0, high1=1.5, high2=0.5,
                      init_cash=200000., init_position=200000.,
                      trade_qty=10000., scale=2.0)
    sig_f = np.zeros(n, np.int8)
    run_backtest(st_f, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"], sig_f, np.empty(0), np.empty(0))

    # DSL 渲染特化模块 (p0..p3 路径, 含倍投)
    kmod = build_dsl_kernel("channel_deviation")
    st_d = kmod.make_state(period="5m", warmup_until=warm, tf1=21,
                           low1=0.0, low2=0.0, high1=0.0, high2=0.0,
                           init_cash=200000., init_position=200000.,
                           trade_qty=10000., scale=2.0,
                           p0=1.5, p1=1.0, p2=1.5, p3=0.5)
    sig_d = np.zeros(n, np.int8)
    kmod.run_backtest(st_d, bars["stime"], bars["open"], bars["high"],
                      bars["low"], bars["close"], bars["volume"],
                      sig_d, np.empty(0), np.empty(0))

    assert (sig_f != 0).sum() > 10, "有效信号样本过少"
    assert np.array_equal(sig_f, sig_d), "信号轨迹不一致"
    assert summarize(st_f) == kmod.summarize(st_d), "绩效汇总不一致"


def test_make_state_general_maps_spec_order():
    """make_state_general 按params_spec 声明顺序填 p0..pN"""
    from evtrade.core.kernel_dsl import make_state_general
    st = make_state_general("dev_trigger", "5m", 0, tf1=13,
                            init_cash=100000., init_position=0.0,
                            trade_qty=5000., strategy_params={"entry_dev": 0.7})
    assert st.p0 == 0.7                       # entry_dev 是 spec 第 1 个参数
    assert st.tf1 == 13 and st.period_seconds == 300
    assert st.init_cash == 100000.0 and st.trade_qty == 5000.0


def test_run_one_dsl_channel_deviation_matches_run_one():
    """run_one_dsl(channel_deviation) 与冻结 run_one 核心指标 bitwise 一致"""
    from evtrade.core.kernel_dsl import run_one_dsl
    from evtrade.core.sweep import run_one
    bars = bars_to_arrays(_synthetic())
    warm = int("20241110") * 1_000_000
    m_dsl = run_one_dsl(bars, "5m", warm, 200000., 200000., 10000.,
                        strategy_name="channel_deviation",
                        strategy_params={"low1": 1.5, "low2": 1.0,
                                         "high1": 1.5, "high2": 0.5})
    m_ref = run_one(bars, "5m", warm, 21, 1.5, 1.0, 1.5, 0.5,
                    200000., 200000., 10000.)
    for k in ("n_trades", "final_cash", "final_position", "final_equity",
              "final_price", "excess", "excess_pct", "turnover", "max_drawdown"):
        assert m_dsl[k] == m_ref[k], k


def test_dev_trigger_kernel_vs_ref_engine_bitwise():
    """dev_trigger (DSL 三端同源): numba 特化内核 与 参考引擎 逐 bar 信号 + 绩效一致"""
    from evtrade.core.kernel_dsl import dsl_kernel, make_state_general, run_one_dsl
    from evtrade.core.sweep import run_one_general
    from evtrade.core.aggregator import BarAggregator
    from evtrade.execution.account import Account
    from evtrade.core.engine import Engine
    from evtrade.execution.base import SimulatedExecutor

    bars_list = _synthetic()
    bars = bars_to_arrays(bars_list)
    warm = int("20241110") * 1_000_000
    params = {"entry_dev": 0.5}

    # 参考引擎 (Python DSL runner)
    account = Account(cash=200000., position=200000.)
    executor = SimulatedExecutor(account, qty=10000., verbose=False)
    strategy = get_strategy("dev_trigger", params=params)
    aggregator = BarAggregator(300, on_bars=None, warmup_until=str(warm))
    eng = Engine(type("F", (), {"stream": lambda s: iter(bars_list)})(),
                 aggregator, strategy, executor, tf1=21, verbose=False)
    eng.run()
    n_ref = len(account.trades)

    # numba 特化内核
    st = make_state_general("dev_trigger", "5m", warm, tf1=21,
                            init_cash=200000., init_position=200000.,
                            trade_qty=10000., strategy_params=params)
    sig_k = np.zeros(len(bars["stime"]), np.int8)
    dsl_kernel("dev_trigger").run_backtest(
        st, bars["stime"], bars["open"], bars["high"], bars["low"],
        bars["close"], bars["volume"], sig_k, np.empty(0), np.empty(0))

    m = run_one_dsl(bars, "5m", warm, 200000., 200000., 10000.,
                    strategy_name="dev_trigger", strategy_params=params)
    g = run_one_general(bars, "5m", warm, 200000., 200000., 10000.,
                        strategy_name="dev_trigger", strategy_params=params)

    assert int(sig_k.nonzero()[0].size) > 10, "有效信号样本过少"
    assert m["n_trades"] == n_ref == g["n_trades"], \
        (m["n_trades"], n_ref, g["n_trades"])
    for k in ("n_buy", "n_sell", "final_cash", "final_position",
              "final_equity", "final_price", "excess_pct", "turnover"):
        assert m[k] == g[k], f"{k}: kernel={m[k]!r} ref={g[k]!r}"
