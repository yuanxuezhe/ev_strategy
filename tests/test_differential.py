from __future__ import annotations
"""差分测试: 参考 Python 实现 (Engine 全链路) vs numba 流式内核

逐 bar 对比: 信号 / 通道值 (UP, DW, None<->NaN) ;
逐笔对比: 成交 (ts, side, qty, price) —— qty/price 要求逐位一致 ;
终态对比: cash / position / last_price / 汇总口径。
任何一项漂移都意味着内核与参考实现语义不一致。
"""

import numpy as np
import pytest

from evtrade import (Account, BarAggregator, ChannelDeviationStrategy, Engine,
                     Feed, SimulatedExecutor)
from evtrade.config import PERIODS
from evtrade.data import synthetic_bars
from evtrade.kernel import bars_to_arrays, make_state, run_backtest, summarize, trades_to_list

SIG_NUM = {"BUY": 1, "SELL": -1, None: 0}


class ListFeed(Feed):
    def __init__(self, bars):
        self.bars = bars

    def stream(self):
        yield from self.bars


class RecordingStrategy:
    """代理策略: 记录每根 bar 的 (signal, up, dw), 逻辑完全委托真实策略"""

    def __init__(self, inner):
        self.inner = inner
        self.sig = []
        self.up = []
        self.dw = []

    def check(self, cur, up, dw):
        s, info = self.inner.check(cur, up, dw)
        self.sig.append(SIG_NUM[s])
        self.up.append(np.nan if up is None else up)
        self.dw.append(np.nan if dw is None else dw)
        return s, info


class RecordingExecutor(SimulatedExecutor):
    """代理执行器: 记录逐笔成交 (走真实 SimulatedExecutor + Account 记账)"""

    def __init__(self, account, qty, scale=1.0):
        super().__init__(account, qty, verbose=False, scale=scale)
        self.records = []

    def trade(self, signal, price, ts):
        ok = super().trade(signal, price, ts)
        if ok:
            t = self.account.trades[-1]
            self.records.append((int(ts), SIG_NUM[signal],
                                 float(t["qty"]), float(t["price"])))
        return ok


def run_reference(bars, period, warmup_until, tf1, params, init_cash, init_position, trade_qty,
                  scale=1.0):
    from evtrade.kernel import resolve_period_seconds
    feed = ListFeed(bars)
    account = Account(cash=init_cash, position=init_position)
    executor = RecordingExecutor(account, qty=trade_qty, scale=scale)
    strategy = RecordingStrategy(ChannelDeviationStrategy(**params))
    aggregator = BarAggregator(resolve_period_seconds(period), on_bars=None,
                               warmup_until=warmup_until)
    engine = Engine(feed, aggregator, strategy, executor, tf1=tf1, verbose=False)
    engine.run()
    return strategy, executor, account


def run_kernel(bars, period, warmup_until_int, tf1, params, init_cash, init_position, trade_qty,
               scale=1.0):
    arr = bars_to_arrays(bars)
    n = len(bars)
    st = make_state(period=period, warmup_until=warmup_until_int, tf1=tf1,
                    low1=params["low1"], low2=params["low2"],
                    high1=params["high1"], high2=params["high2"],
                    init_cash=init_cash, init_position=init_position,
                    trade_qty=trade_qty, scale=scale,
                    record_trades=True, trade_cap=n)
    sig = np.zeros(n, np.int8)
    up = np.full(n, np.nan)
    dw = np.full(n, np.nan)
    run_backtest(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                 arr["close"], arr["volume"], sig, up, dw)
    return st, sig, up, dw


def assert_differential(bars, period, warmup_str, warmup_until_int, tf1, params,
                        init_cash=200000.0, init_position=200000.0, trade_qty=10000.0,
                        scale=1.0):
    strat, executor, account = run_reference(
        bars, period, warmup_str, tf1, params, init_cash, init_position, trade_qty,
        scale=scale)
    st, sig, up, dw = run_kernel(
        bars, period, warmup_until_int, tf1, params, init_cash, init_position, trade_qty,
        scale=scale)

    # 参考实现的 strategy.check 只在 mark=1 (stime >= warmup_until) 的 bar 被调用;
    # 内核轨迹为全 bar 记录, 对比时按预热偏移切片。
    arr = bars_to_arrays(bars)
    if warmup_until_int:
        offset = int(np.searchsorted(arr["stime"], warmup_until_int))
    else:
        offset = 0

    # 1) 逐 bar 信号
    #    参考实现 check 调用次数 = mark=1 的 bar 数 + 1 (Engine.run 结尾 flush()
    #    会再回调一次 on_bars; 因单桶锁, flush 永不产生新信号, 通道值与最后一根相同)
    ref_sig = np.array(strat.sig, dtype=np.int8)
    n_strategy = len(bars) - offset
    assert len(ref_sig) == n_strategy + (1 if n_strategy else 0)
    if n_strategy:
        assert ref_sig[-1] == 0, "flush 收尾回调不应产生信号"
    assert np.array_equal(sig[offset:], ref_sig[:n_strategy]), (
        f"信号序列不一致: 首个差异 idx="
        f"{int(np.argmax(sig[offset:] != ref_sig[:n_strategy])) + offset}, "
        f"ref_n_sig={int((ref_sig != 0).sum())}, ker_n_sig={int((sig != 0).sum())}")

    # 2) 逐 bar 通道值 (None<->NaN, 其余逐位一致)
    ref_up = np.array(strat.up, dtype=np.float64)
    ref_dw = np.array(strat.dw, dtype=np.float64)
    if n_strategy:
        assert ref_up[-1] == ref_up[-2] and ref_dw[-1] == ref_dw[-2], \
            "flush 收尾回调的通道值应与最后一根 bar 相同"
    assert np.array_equal(up[offset:], ref_up[:n_strategy], equal_nan=True), "UP 轨不一致"
    assert np.array_equal(dw[offset:], ref_dw[:n_strategy], equal_nan=True), "DW 轨不一致"

    # 3) 逐笔成交 (bitwise)
    kt = trades_to_list(st)
    rt = executor.records
    assert len(kt) == len(rt), f"成交笔数不一致: ref={len(rt)} ker={len(kt)}"
    for i, (a, b) in enumerate(zip(kt, rt)):
        assert a["ts"] == b[0] and SIG_NUM[a["side"]] == b[1], f"trade#{i} 时间/方向不一致"
        assert a["qty"] == b[2], f"trade#{i} qty: ref={b[2]!r} ker={a['qty']!r}"
        assert a["price"] == b[3], f"trade#{i} price: ref={b[3]!r} ker={a['price']!r}"

    # 4) 账户终态 (bitwise)
    assert st.cash == account.cash, f"cash: ref={account.cash!r} ker={st.cash!r}"
    assert st.position == account.position
    assert st.last_price == account.last_price

    # 5) 汇总口径一致
    s = summarize(st)
    assert s["final_equity"] == account.equity()
    assert s["baseline"] == account.baseline_equity()
    return s


PARAM_CASES = [
    {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5},    # 默认
    {"low1": 0.4, "low2": 0.25, "high1": 0.4, "high2": 0.2},   # 灵敏 (信号多)
    {"low1": 0.5, "low2": 0.5, "high1": 0.5, "high2": 0.5},    # 迟滞退化边界
    {"low1": 2.0, "low2": 0.2, "high1": 2.0, "high2": 0.1},    # 宽置位窄触发
]

ALL_PERIODS = list(PERIODS.keys())


@pytest.mark.parametrize("params", PARAM_CASES)
def test_differential_5m_params(params):
    bars = synthetic_bars(days=25, start_ymd="20250101", seed=11)
    assert_differential(bars, "5m", "20250110000000", 20250110000000, 21, params)


def test_differential_scale_martingale():
    """倍投系数 scale=2.0: 参考引擎与内核逐笔一致 (数量累乘/反向重置)"""
    bars = synthetic_bars(days=25, start_ymd="20250101", seed=11)
    s = assert_differential(bars, "5m", "20250110000000", 20250110000000, 5,
                            PARAM_CASES[1], scale=2.0)
    assert s["n_trades"] > 0


@pytest.mark.parametrize("period", ALL_PERIODS)
def test_differential_all_periods(period):
    bars = synthetic_bars(days=25, start_ymd="20250101", seed=11)
    assert_differential(bars, period, "20250110000000", 20250110000000, 21,
                        PARAM_CASES[1])


@pytest.mark.parametrize("period,tf1", [("7m", 21), ("90m", 21), ("2h", 13),
                                        ("3d", 5)])
def test_differential_custom_periods(period, tf1):
    """任意周期 (7m/90m/2h/3d): 参考引擎(通用桶算法) 与内核逐笔一致"""
    bars = synthetic_bars(days=30, start_ymd="20250101", seed=29)
    assert_differential(bars, period, "20250110000000", 20250110000000, tf1,
                        PARAM_CASES[1])


@pytest.mark.parametrize("tf1", [5, 13, 60, 300])
def test_differential_tf1(tf1):
    bars = synthetic_bars(days=25, start_ymd="20250101", seed=11)
    assert_differential(bars, "5m", "20250110000000", 20250110000000, tf1,
                        PARAM_CASES[1])


def test_differential_no_warmup():
    """warmup_until=None/0: 全程策略期 (含指标未就绪期的 None 通道)"""
    bars = synthetic_bars(days=15, start_ymd="20250101", seed=13)
    assert_differential(bars, "5m", None, 0, 21, PARAM_CASES[0])


def test_differential_longer_series():
    """更长序列 (≈2.9万根): 检验锁存长序列不漂移"""
    bars = synthetic_bars(days=120, start_ymd="20250101", seed=17)
    s = assert_differential(bars, "5m", "20250115000000", 20250115000000, 21,
                            PARAM_CASES[1])
    assert s["n_trades"] > 0


def test_differential_seed_sweep():
    """多份随机数据: 覆盖不同价格路径"""
    for seed in (21, 22, 23):
        bars = synthetic_bars(days=20, start_ymd="20250101", seed=seed)
        assert_differential(bars, "5m", "20250108000000", 20250108000000, 21,
                            PARAM_CASES[1])
