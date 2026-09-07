"""步骤 2 测试: kernel._strategy_check 通用化 (任意 DSL 策略走 p0..pN)

锁定: 72 项差分测试通过 (向后兼容);
       策略参数按 params_spec 声明顺序填 p0..pN (不绑定任何具体策略参数名)
       新策略参数化路径正确
       DSL 渲染特化内核与冻结内核 bitwise 一致 (kernel_dsl)
"""
from __future__ import annotations

import numpy as np
import pytest

from evtrade.strategies import get_strategy, get_strategy_param_spec
from evtrade.core.kernel import bars_to_arrays, summarize
from evtrade.core.kernel_dsl import dsl_kernel
from evtrade.replay import replay_engine, replay_kernel


# 单源: 走 dsl_kernel("channel_deviation") 特化模块 (2026-09 重构后冻结本尊 _strategy_check 已清空)
_KMOD = dsl_kernel("channel_deviation")
make_state = _KMOD.make_state
run_backtest = _KMOD.run_backtest


def _synthetic():
    from evtrade.data import synthetic_bars
    return synthetic_bars(days=30, start_ymd="20241101", seed=42)


# ============ KernelState p0..pN 通用参数 ============

def test_kernelstate_p0p3_default_zeros():
    """KernelState 不传 p0..p3 时, 默认全 0 (与策略层无关)"""
    from evtrade.core.kernel import KernelState
    st = KernelState(
        period_seconds=300, warmup_until=0, tf1=21,
        init_cash=200000., init_position=200000., trade_qty=10000.,
        scale=1.0, buy_pct=0., sell_pct=0., all_in=False,
        record_trades=False, trade_cap=0,
    )
    assert st.p0 == 0.0 and st.p1 == 0.0 and st.p2 == 0.0 and st.p3 == 0.0


def test_kernelstate_explicit_p0_kept():
    """显式传 p0..p3 时, 框架原样保留 (无别名映射)"""
    from evtrade.core.kernel import KernelState
    st = KernelState(
        period_seconds=300, warmup_until=0, tf1=21,
        init_cash=200000., init_position=200000., trade_qty=10000.,
        scale=1.0, buy_pct=0., sell_pct=0., all_in=False,
        record_trades=False, trade_cap=0,
        p0=9.9, p1=0.0, p2=0.0, p3=0.0,
    )
    assert st.p0 == 9.9 and st.p1 == 0.0 and st.p2 == 0.0 and st.p3 == 0.0


# ============ make_state 接受 p0..p15 ============

def test_make_state_with_explicit_p0p3():
    """make_state 显式传 p0..p3"""
    bars = bars_to_arrays(_synthetic())
    n = len(bars)
    st = make_state(period="5m", warmup_until=int("20241110")*1_000_000,
                    tf1=21,
                    init_cash=200000., init_position=200000., trade_qty=10000.,
                    p0=1.7, p1=0.9, p2=1.7, p3=0.6)  # 显式参数
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    s = summarize(st)
    assert s["n_trades"] >= 0


def test_make_state_no_p0p3_default_zeros():
    """make_state 不传 p0..p3 时, 默认全 0 (策略无关)"""
    bars = bars_to_arrays(_synthetic())
    st = make_state(period="5m", warmup_until=int("20241110")*1_000_000,
                    tf1=21,
                    init_cash=200000., init_position=200000., trade_qty=10000.)
    assert st.p0 == 0.0 and st.p1 == 0.0 and st.p2 == 0.0 and st.p3 == 0.0


# ============ kernel 路径与 ref 引擎 bitwise 一致 ============

def test_kernel_vs_ref_engine_bitwise():
    """kernel 路径 (用 p0..p3) 与 ref 引擎逐 bar 信号 + 成交笔数一致"""
    bars = _synthetic()
    warm = int("20241110")*1_000_000
    params = {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}
    # kernel 路径: dsl_kernel("channel_deviation") 特化模块 (走 build_dsl_kernel 渲染管线)
    arr = bars_to_arrays(bars)
    kmod = dsl_kernel("channel_deviation")
    st = kmod.make_state(period="5m", warmup_until=warm, tf1=21,
                         p0=params["low1"], p1=params["low2"],
                         p2=params["high1"], p3=params["high2"],
                         init_cash=200000., init_position=200000., trade_qty=10000.,
                         record_trades=True, trade_cap=len(arr["stime"]))
    sig_k = np.zeros(len(arr["stime"]), np.int8)
    up = np.full(len(arr["stime"]), np.nan)
    dw = np.full(len(arr["stime"]), np.nan)
    kmod.run_backtest(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                      arr["close"], arr["volume"], sig_k, up, dw)
    # ref 路径: replay_engine (frozen 策略)
    rep = replay_engine(bars, "5m", warm, 21,
                        strategy_name="channel_deviation", strategy_params=params,
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


def test_dsl_spliced_channel_deviation_vs_ref_engine_bitwise():
    """DSL 渲染的 channel_deviation 特化内核 与 Python ref 引擎逐位一致

    这是渲染器忠实性的锁定: 特化模块 = kernel.py 源码整段替换策略段后 exec,
    若渲染改变了语义 (字段映射/表达式顺序), 这里立刻红。

    历史 (2026-09 重构前): 与"冻结 kernel 本尊"对比; 重构后冻结本尊的 _strategy_check
    已被清空 (单源 = DSL docstring), 改与 Python ref 引擎对比, 含义更强:
    渲染产物直接决定 numba 信号序列, 必须与 Python DSL docstring 执行结果逐位一致。
    """
    from evtrade.core.kernel_dsl import build_dsl_kernel
    bars_list = _synthetic()
    bars = bars_to_arrays(bars_list)
    warm = int("20241110") * 1_000_000
    n = len(bars["stime"])

    # DSL 渲染特化模块 (p0..p3 路径, 含倍投)
    kmod = build_dsl_kernel("channel_deviation")
    st_d = kmod.make_state(period="5m", warmup_until=warm, tf1=21,
                           init_cash=200000., init_position=200000.,
                           trade_qty=10000., scale=2.0,
                           p0=1.5, p1=1.0, p2=1.5, p3=0.5)
    sig_d = np.zeros(n, np.int8)
    kmod.run_backtest(st_d, bars["stime"], bars["open"], bars["high"],
                      bars["low"], bars["close"], bars["volume"],
                      sig_d, np.empty(0), np.empty(0))

    # Python ref 引擎: ChannelDeviationStrategy.check (走 DSL docstring exec 路径)
    # replay_engine 接受原始 Bar 对象列表 (不是 numpy dict)
    rep = replay_engine(bars_list, "5m", warm, 21,
                        strategy_name="channel_deviation",
                        strategy_params={"low1": 1.5, "low2": 1.0,
                                         "high1": 1.5, "high2": 0.5},
                        init_cash=200000., init_position=200000.,
                        trade_qty=10000., scale=2.0)
    sig_ref = rep["sig"]

    assert (sig_d != 0).sum() > 10, "有效信号样本过少"
    assert np.array_equal(sig_d, sig_ref), \
        "DSL 渲染特化内核 与 Python ref 引擎 信号轨迹不一致"
    # 信号序列 bitwise 一致即说明渲染产物忠实于 DSL docstring (核心锁定)。
    # 绩效汇总由 run_one_dsl 单独锁定 (test_run_one_dsl_vs_run_one_from_dict)


def test_make_state_general_maps_spec_order():
    """make_state_general 按params_spec 声明顺序填 p0..pN"""
    from evtrade.core.kernel_dsl import make_state_general
    st = make_state_general("dev_trigger", "5m", 0, tf1=13,
                            init_cash=100000., init_position=0.0,
                            trade_qty=5000., strategy_params={"entry_dev": 0.7})
    assert st.p0 == 0.7                       # entry_dev 是 spec 第 1 个参数
    assert st.tf1 == 13 and st.period_seconds == 300
    assert st.init_cash == 100000.0 and st.trade_qty == 5000.0


def test_run_one_dsl_vs_run_one_from_dict():
    """run_one_dsl 与 run_one_from_dict (strategy-agnostic 入口) 核心指标 bitwise 一致"""
    from evtrade.core.kernel_dsl import run_one_dsl
    from evtrade.core.sweep import run_one_from_dict
    bars = bars_to_arrays(_synthetic())
    warm = int("20241110") * 1_000_000
    params = {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}
    m_dsl = run_one_dsl(bars, "5m", warm, 200000., 200000., 10000.,
                        strategy_name="channel_deviation",
                        strategy_params=params)
    m_ref = run_one_from_dict(
        bars,
        {"period": "5m", "tf1": 21,
         "init_cash": 200000., "init_position": 200000.,
         "trade_qty": 10000.,
         "params": params},
        warm,
        strategy_name="channel_deviation")
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
