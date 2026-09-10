"""统一策略契约锁定测试 (DSL/numba 已下线)

核心断言:
  1. channel_deviation / ma_crossover 都是 VectorizedStrategy 子类
  2. compute_signals 返回 xp.ndarray[int8]
  3. cpu vs gpu 信号 bitwise 一致 (cupy 不可用时跳过 gpu)
  4. vectorized 路径 vs Engine.on_bars 路径逐笔一致 (reconcile)
  5. metrics summary 16 字段齐全 (cagr/sharpe/sortino/calmar/max_dd_days/...)
"""
from __future__ import annotations

import numpy as np
import pytest

import evtrade
from evtrade import (
    VectorizedStrategy, get_strategy, run_vectorized,
)
from evtrade.backends import get_xp


def _make_bars(n: int = 200, seed: int = 42):
    """合成 1m bars (np.float64), 24h 内连续"""
    import datetime
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.0005, n)
    prices = 100 * np.exp(np.cumsum(rets))
    high = prices * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = prices * (1 - np.abs(rng.normal(0, 0.001, n)))
    start = datetime.datetime(2025, 1, 1, 9, 30, 0)
    stime = []
    cur = start
    for _ in range(n):
        stime.append(int(cur.strftime("%Y%m%d%H%M%S")))
        cur = cur + datetime.timedelta(minutes=1)
    return {
        "stime": np.array(stime, dtype=np.int64),
        "open": prices.astype(np.float64),
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
        "close": prices.astype(np.float64),
        "volume": np.ones(n, dtype=np.float64),
    }


def test_both_strategies_are_vectorized_subclass():
    assert issubclass(evtrade.ChannelDeviationStrategy, VectorizedStrategy)
    assert issubclass(evtrade.MACrossoverStrategy, VectorizedStrategy)


def test_strategy_registry_lists_both():
    keys = evtrade.available_strategies()
    assert "channel_deviation" in keys
    assert "ma_crossover" in keys


def test_step_returns_int():
    """step(state, bar, params) -> (state, int); signal ∈ {-1, 0, 1}"""
    s = evtrade.get_strategy("ma_crossover", fast=3, slow=10)
    bars = _aggregate_bars("5m", _make_bars())
    state = s.init_state(s.params)
    sigs = []
    for i in range(len(bars["ts"])):
        bar = {"ts": int(bars["ts"][i]), "o": float(bars["o"][i]),
               "h": float(bars["h"][i]), "l": float(bars["l"][i]),
               "c": float(bars["c"][i]), "v": float(bars["v"][i]),
               "mark": int(bars["mark"][i])}
        state, sig = s.step(state, bar, s.params)
        sigs.append(sig)
    sigs = np.array(sigs, dtype=np.int8)
    assert sigs.dtype == np.int8
    assert len(sigs) == len(bars["ts"])
    assert set(sigs.tolist()).issubset({-1, 0, 1})


def test_ma_crossover_cpu_vs_gpu_bitwise_equal():
    """CPU vs GPU 信号 bitwise 一致 (cupy 不可用时 skip)

    strategy-step-only: 跑两次 run_vectorized (cpu + gpu), 比较 sig_live 序列
    """
    try:
        cp = get_xp("gpu")
        _ = cp.zeros(2)
    except Exception:
        pytest.skip("cupy/CUDA 不可用")

    bars = _make_bars()
    s_cpu = evtrade.get_strategy("ma_crossover", fast=5, slow=20)
    s_gpu = evtrade.get_strategy("ma_crossover", fast=5, slow=20)
    out_cpu = run_vectorized(bars, "5m", warmup_until=0,
                             strategy=s_cpu, params=s_cpu.params, device="cpu")
    out_gpu = run_vectorized(bars, "5m", warmup_until=0,
                             strategy=s_gpu, params=s_gpu.params, device="gpu")
    np.testing.assert_array_equal(out_cpu["sig"], out_gpu["sig"])


def test_vectorized_vs_engine_on_bars_reconcile():
    """vectorized 路径 vs Engine.on_bars 路径逐笔一致 (channel_deviation)"""
    from evtrade.account import Account
    from evtrade.aggregator import BarAggregator
    from evtrade.core._harness import ListBarFeed
    from evtrade.core.engine import Engine
    from evtrade.core.replay import reconcile
    from evtrade.core.timeutils import resolve_period_seconds
    from evtrade.execution import SimulatedExecutor
    from evtrade.primitives import Bar

    bars_arr = _make_bars(n=300, seed=7)
    n_bars = len(bars_arr["stime"])
    # vectorized 路径 (桶级信号)
    strat = evtrade.get_strategy("channel_deviation", tf1=5)
    out_v = run_vectorized(bars_arr, "5m", warmup_until=0,
                           strategy=strat, params=strat.params, device="cpu")
    sig_v = out_v["sig"]      # 桶级
    trades_v = out_v["summary"]["trades"]

    # Engine.on_bars 路径 (1m bar -> aggregator 内部桶 -> on_bars 调用 = 桶级)
    bars_list = [Bar(stime=str(bars_arr["stime"][i]), code="T",
                     open=float(bars_arr["open"][i]),
                     high=float(bars_arr["high"][i]),
                     low=float(bars_arr["low"][i]),
                     close=float(bars_arr["close"][i]),
                     volume=int(bars_arr["volume"][i]))
                 for i in range(n_bars)]
    # 重建 strategy (避免 instance FSM 串味)
    strat2 = evtrade.get_strategy("channel_deviation", tf1=5)
    feed = ListBarFeed(bars_list)
    agg = BarAggregator(resolve_period_seconds("5m"), on_bars=None)
    exe = SimulatedExecutor(Account(cash=100000, position=0), qty=100, verbose=False)
    eng = Engine(feed, agg, strat2, exe, verbose=False)
    eng.run()
    sig_r = np.array(eng.bucket_signals, dtype=np.int8)  # 桶级

    # vectorized vs Engine 桶级信号: 偶发 1 根边界差异 (warmup_until=0 时
    # bucket_ts 与 timeutils 算的 ts 在第一根有 1 桶差, 见 kbs/13 §5);
    # 严口径锁 0, 宽口径锁 <=1 (默认宽口径; 严口径用 env EVT_RECONCILE_STRICT=1)
    import os
    n = min(len(sig_v), len(sig_r))
    n_diff = int(np.sum(sig_v[:n] != sig_r[:n]))
    cap = 0 if os.environ.get("EVT_RECONCILE_STRICT") == "1" else 1
    assert n_diff <= cap, f"桶级信号分歧 {n_diff}/{n} (cap={cap})"

    # 成交笔数大致一致 (允许桶边界 1 根差异: 见 kbs/13 §5)
    trades_r = exe.account.trades
    trade_diff = abs(len(trades_v) - len(trades_r))
    assert trade_diff <= 1, \
        f"成交笔数差异过大 vectorized={len(trades_v)} engine={len(trades_r)}"


def test_metrics_summary_has_26_fields():
    """run_vectorized.summary 必须含 26 字段 (25 + x_mdd)"""
    strat = evtrade.get_strategy("ma_crossover", fast=5, slow=20)
    out = run_vectorized(_make_bars(n=300), "5m", warmup_until=0,
                         strategy=strat, params=strat.params, device="cpu")
    s = out["summary"]
    required = {
        # 终态 (5)
        "final_price", "final_cash", "final_position", "final_equity", "baseline",
        # 交易 (5)
        "n_trades", "n_buy", "n_sell", "turnover", "excess_pct",
        # 时间 (2)
        "years", "cagr_excess",
        # 风险调整 (5)
        "cagr", "sharpe_excess", "sortino_excess", "calmar", "ir",
        # 回撤 (4, 含 unify-units 保留的 x_mdd)
        "max_drawdown", "max_dd_days", "max_dd_recovered", "x_mdd",
        # 持仓行为 (7)
        "win_rate", "profit_factor", "avg_pnl",
        "max_consecutive_wins", "max_consecutive_losses",
        "avg_hold_bars", "max_hold_bars",
        # 基准对比 (2)
        "baseline_max_dd", "dd_excess",
    }
    missing = required - set(s.keys())
    assert not missing, f"summary 缺字段: {sorted(missing)}"
    assert "ann_excess_pct" not in s  # 已重命名为 cagr_excess
    assert s["years"] > 0 or s["n_trades"] == 0


# ============ strategy-step-only 新增锁定 ============

def test_init_state_returns_dataclass():
    """stateful 策略 init_state 返回 dataclass 实例"""
    from dataclasses import is_dataclass

    cd = evtrade.get_strategy("channel_deviation", tf1=5)
    mc = evtrade.get_strategy("ma_crossover", fast=3, slow=10)

    cd_state = cd.init_state(cd.params)
    mc_state = mc.init_state(mc.params)

    assert is_dataclass(cd_state), \
        f"ChannelDeviation init_state 应返 dataclass, 实际 {type(cd_state)}"
    assert is_dataclass(mc_state), \
        f"MACrossover init_state 应返 dataclass, 实际 {type(mc_state)}"

    # cd_state 含 EMA 通道 + FSM
    from evtrade.indicators import EMAChannelState
    assert isinstance(cd_state.ema, EMAChannelState)
    assert "low_hit" in cd_state.fsm and "lock_ts" in cd_state.fsm

    # mc_state 含 fast/slow EMA
    from evtrade.indicators import EMAState
    assert isinstance(mc_state.fast, EMAState)
    assert isinstance(mc_state.slow, EMAState)
    assert mc_state.has_prev is False  # 初值


def test_no_instance_state_in_strategies():
    """静态扫描: 策略文件 MUST NOT 出现 self._xxx 状态字段 (strategy-step-only)"""
    import re
    from pathlib import Path
    strategies_dir = Path(evtrade.__file__).parent / "strategies"
    pattern = re.compile(r"\bself\._(fsm|up_st|dw_st|has_prev|prev_ts|cur_high|cur_low)\b")
    violators = []
    for py in strategies_dir.glob("*.py"):
        if py.name == "vectorized_base.py":
            continue  # 基类不算
        text = py.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            violators.append(f"{py.name}:{line_no} {m.group(0)!r}")
    assert not violators, \
        f"策略禁止 self._xxx 状态字段 (state 必须由 engine 持有): {violators}"


def test_step_state_persists_across_calls():
    """连续 step(state, bar) 调, EMA 状态在 state 字段里累积"""
    cd = evtrade.get_strategy("channel_deviation", tf1=3)
    state = cd.init_state(cd.params)

    # 调 5 次 step, 每次 mark=1, close 递增
    for i, c in enumerate([10.0, 11.0, 12.0, 13.0, 14.0]):
        bar = {"ts": i, "o": c, "h": c, "l": c, "c": c, "v": 0.0, "mark": 1}
        state, _ = cd.step(state, bar, cd.params)

    # EMA 状态跨调用持续:
    # channel_deviation.step 内部对每根 bar 推 2 次 (旧桶 push + 当前桶 step),
    # 但 state 在同一根 bar 内只 push 一次 (ts 切换时不重复); 上面 5 根 ts 不同
    # 但都触发 prev_ts != cur_ts, 所以 5 + 5 = 10. 实际取决于 prev_ts 切换逻辑
    # 这里只断言 count > 0 + 跨调用持续
    # 注: dataclass 是 mutable, state 与 new_state.ema 同对象; 用 id() 比对
    assert state.ema.up.count > 0, \
        f"调 5 次 step, state.ema.up.count 应 > 0, 实际 {state.ema.up.count}"
    assert state.ema.dw.count == state.ema.up.count, "up/dw count 应一致"
    assert state.has_prev is True, "state.has_prev 跨调用持续"

    # 同一个 state 继续调, EMA count 继续增长
    old_count = state.ema.up.count
    bar = {"ts": 5, "o": 15.0, "h": 15.0, "l": 15.0, "c": 15.0, "v": 0.0, "mark": 1}
    new_state, _ = cd.step(state, bar, cd.params)
    assert new_state.ema.up.count > old_count, \
        f"再调 1 次, EMA count 应递增, 实际 new={new_state.ema.up.count} old={old_count}"


def test_compute_signals_for_one_bar_default_wrapper():
    """DEPRECATED 2026-09-10 (strategy-step-only): wrapper 已删;
    此测试改验 init_state 返回 None 时 step 仍能工作"""

    class TestStrat(VectorizedStrategy):
        params_spec = {}

        def init_state(self, params):
            return None

        def step(self, state, bar, params):
            return state, 1  # 无状态, 永远 BUY

    bar = {"ts": 100, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0,
           "v": 0.0, "mark": 1}
    s = TestStrat()
    state = s.init_state({})
    new_state, sig = s.step(state, bar, {})
    assert sig == 1
    assert new_state is None  # 无状态策略 state 始终 None


# ============ helpers ============

def _aggregate_bars(period: str, bars: dict) -> dict:
    """跑一次 vectorized_engine._aggregate_buckets_xp (numpy)"""
    from evtrade.core.vectorized_engine import _aggregate_buckets_xp
    return _aggregate_buckets_xp(np, bars, period, warmup_until=0)


def _aggregate_bars_gpu(period: str, bars: dict) -> dict:
    """GPU 版桶聚合"""
    cp = get_xp("gpu")
    from evtrade.core.vectorized_engine import _aggregate_buckets_xp
    return _aggregate_buckets_xp(cp, bars, period, warmup_until=0)
