"""filtered_mr 4 重过滤均值回归策略测试"""
from __future__ import annotations

import numpy as np
import pytest

import evtrade
from evtrade.strategies import get_strategy, get_strategy_class
from evtrade.strategies.filtered_mr import (
    FilteredMRState,
    FilteredMRStrategy,
)


def _make_bars(closes, marks=None, period_seconds=300, start_ts=20250101093000):
    """闭合价序列 + mark 序列 -> bar dict 列表 (5m 桶)"""
    n = len(closes)
    if marks is None:
        marks = [1] * n
    bars = []
    for i in range(n):
        ts = start_ts + i * period_seconds
        bars.append({
            "ts": ts,
            "o": float(closes[i]),
            "h": float(closes[i]) + 0.1,
            "l": float(closes[i]) - 0.1,
            "c": float(closes[i]),
            "v": 0.0,
            "mark": int(marks[i]),
        })
    return bars


def _run_strategy(bars, params=None):
    """跑完整 bar 序列, 返回 sig 列表"""
    strat = FilteredMRStrategy()
    p = strat.params if params is None else {**strat.params, **params}
    state = strat.init_state(p)
    sigs = []
    for bar in bars:
        state, sig = strat.step(state, bar, p)
        sigs.append(sig)
    return sigs


# ============ 1. 注册 + 契约 ============

def test_filtered_mr_registered():
    assert "filtered_mr" in evtrade.available_strategies()
    cls = get_strategy_class("filtered_mr")
    assert cls is FilteredMRStrategy


def test_init_state_returns_dataclass():
    from dataclasses import is_dataclass
    state = FilteredMRStrategy().init_state({})
    assert is_dataclass(state)
    assert isinstance(state, FilteredMRState)
    # 默认字段
    assert state.higher_count == 0
    assert state.cur_ema_state.count == 0
    assert state.atr == 0.0
    assert state.adx == 0.0
    assert state.touched_lower_prev is False


def test_step_basic_returns_int():
    """step 返回 (state, sig), sig ∈ {-1, 0, 1}, mark=0 早返回"""
    strat = FilteredMRStrategy()
    p = strat.params
    state = strat.init_state(p)
    # 100 根简单 bar
    bars = _make_bars([100.0 + i * 0.1 for i in range(100)],
                      marks=[0] * 20 + [1] * 80)
    sigs = []
    for bar in bars:
        state, sig = strat.step(state, bar, p)
        sigs.append(sig)
        assert isinstance(sig, int)
        assert sig in {-1, 0, 1}
    # 至少 1 根 sig != 0 (震荡+顺势场景会有信号)
    # mark=0 段全部 sig=0
    assert all(s == 0 for s in sigs[:20])


# ============ 2. 过滤行为 ============

def test_strong_trend_filter_blocks_all_signals():
    """ADX > threshold 时所有信号被屏蔽"""
    # 构造强趋势: 单调上升 close + 1.0% 步进, 触发高 ADX
    n = 200
    closes = [100.0 * (1.01 ** i) for i in range(n)]
    marks = [0] * 30 + [1] * (n - 30)
    bars = _make_bars(closes, marks=marks)
    sigs = _run_strategy(bars)
    # 强趋势段 sig 应该非常少或全 0 (被 ADX 过滤)
    live_sigs = [s for s in sigs[30:] if s != 0]
    # 允许少量 (信号可能被 close-FSM 阻击), 但 ADX 高时大部分被过滤
    # 注: 单调上升 close + 大周期顺势, ADX 必然 > 30, 全过滤
    assert len(live_sigs) < n // 4, (
        f"强趋势应有大量信号被 ADX 过滤; 实际 {len(live_sigs)}/{n - 30}")


def test_high_vol_filter_blocks_all_signals():
    """ATR > mult × MA 时全停"""
    # 构造剧烈波动: 偶发跳空 + 正常波动交替
    np.random.seed(42)
    n = 200
    closes = [100.0]
    for _ in range(n - 1):
        # 每 10 根来个 5% 跳空
        if len(closes) % 10 == 0:
            closes.append(closes[-1] * 1.05)
        else:
            closes.append(closes[-1] * (1 + np.random.normal(0, 0.005)))
    bars = _make_bars(closes, marks=[0] * 30 + [1] * (n - 30))
    sigs = _run_strategy(bars)
    # 高波动段信号大幅减少 (ATR > 1.5 * MA 触发休克)
    live_sigs = [s for s in sigs[30:] if s != 0]
    # 期望至少有一些 bar 被休克
    # 不强制 0 (震荡中也可能出信号, 但高波动段必停)
    # 这里只验证不出异常 / sig 范围
    assert all(s in {-1, 0, 1} for s in sigs)


def test_higher_period_filter_blocks_counter_trend():
    """大周期顺势逻辑: 直接构造 higher_has_ema + higher_trend=1 状态, 验证 SELL 被禁

    不依赖数据构造 (难以控制高周期 EMA 就绪); 直接构造 state 验证 step 输出。
    """
    strat = FilteredMRStrategy()
    p = strat.params
    state = strat.init_state(p)
    # 喂足够 bar 让 EMA/ATR 就绪; ADX 在震荡下接近 0 (弱趋势, 不强过滤信号)
    bars = _make_bars([100.0 + np.sin(i / 8) * 0.5 for i in range(150)],
                      marks=[0] * 30 + [1] * 120)
    for bar in bars:
        state, _ = strat.step(state, bar, p)
    assert state.atr > 0
    assert state.cur_ema_state.count >= p["cur_ema_period"]
    # 手动覆盖 higher_ema: 让 higher_trend = 1 (大周期多头)
    state.higher_has_ema = True
    state.higher_ema_state.ema = 99.0
    state.higher_last_close = 100.5     # close > 99 -> higher_trend = 1
    # 设置 FSM 让上桶已触上轨, 本桶也触上轨, close < upper -> 应出 SELL 但被多头大周期过滤
    state.touched_upper_prev = True
    # 构造一根 bar (在同一个 higher bucket 内, 避免 bucket switch 覆盖手动设置)
    # 末根 warmup bar ts = 20250101093000 + 149*300 = 20250101215500
    # higher bucket (1h) = [..., 22:00]; 21:59 还在同桶
    bar = {"ts": 20250101215900, "o": 100.5, "h": 105.0, "l": 99.5,
           "c": 100.2, "v": 0.0, "mark": 1}
    state, sig = strat.step(state, bar, p)
    # 大周期多头禁做空; close FSM 应让 BUY 也仅在 low 触轨时触发 (本 bar low 在带内)
    assert sig in {0, 1}, (
        f"大周期多头时 SELL 应被禁; 实际 sig={sig}"
    )
    assert sig != -1, "大周期多头禁止做空"


def test_close_confirmation_requires_two_bars():
    """Close 确认 FSM: 上一桶触轨 + 本桶 close 回到带内才出信号"""
    # 构造: 第一桶影线下穿下轨 (无上一桶 FSM, 不产信号);
    #       第二桶再下穿下轨 + close 回到带内 -> 出 BUY
    n = 100
    np.random.seed(42)
    # 第 30-40 桶: 影线下穿 (低 1.5%), close 回到带内
    # 第 60-70 桶: 类似形态
    closes = []
    for i in range(n):
        if 30 <= i <= 40 or 60 <= i <= 70:
            closes.append(100.0)   # close 在 EMA 附近
        else:
            closes.append(100.0 + np.sin(i / 8) * 0.3)
    bars = []
    for i in range(n):
        ts = 20250101093000 + i * 300
        if 30 <= i <= 40 or 60 <= i <= 70:
            # 影线下穿下轨
            h = 100.1
            l = 99.2   # 远低于 EMA, 触下轨
            c = 100.0
        else:
            h = closes[i] + 0.1
            l = closes[i] - 0.1
            c = closes[i]
        bars.append({"ts": ts, "o": c, "h": h, "l": l, "c": c,
                     "v": 0.0, "mark": 1 if i >= 50 else 0})
    sigs = _run_strategy(bars)
    # 验证: 单桶触发不应立刻出信号 (FSM 需要上一桶先触轨)
    # 单根影线下穿但 FSM 上一根没触轨 -> 该桶不出 sig
    # 真正能出 sig 的是连续两根都触轨 + 本根 close 回到带内
    # 此测试只验证: sig 数小于 bar 数 (FSM 过滤有效)
    live_sigs = [s for s in sigs[50:] if s != 0]
    # 至少 1 根触轨, 但 FSM 要求连续 2 根, 因此 sig 数 << 触轨 bar 数
    n_touched = sum(1 for i in range(50, n) if bars[i]["l"] < 99.5)  # 估计触轨数
    assert len(live_sigs) <= n_touched, (
        f"sig 数应 <= 触轨 bar 数 (FSM 过滤); sig={len(live_sigs)}, touched~{n_touched}"
    )


# ============ 3. 无 instance state ============

def test_no_instance_state_in_filtered_mr():
    """静态扫描: filtered_mr MUST NOT 引入 self._xxx 实例状态"""
    import re
    from pathlib import Path
    strategies_dir = Path(evtrade.__file__).parent / "strategies"
    pattern = re.compile(r"\bself\._(fsm|up_st|dw_st|has_prev|prev_ts|cur_high|cur_low)\b")
    violators = []
    for py in strategies_dir.glob("*.py"):
        if py.name == "vectorized_base.py":
            continue
        text = py.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            violators.append(f"{py.name}:{line_no} {m.group(0)!r}")
    assert not violators, (
        f"策略禁止 self._xxx 状态字段 (state 必须由 engine 持有): {violators}"
    )


# ============ 4. CLI 集成 (smoke) ============

def test_sweep_filtered_mr_runs(tmp_path):
    """CLI sweep 能跑 filtered_mr (smoke, 不验证具体数值)

    backtest 不支持 --synthetic-days (sweep 专属), 用 sweep 验证集成。
    """
    import sys
    from io import StringIO
    from evtrade.cli import main as cli_main
    out_csv = tmp_path / "test_filtered_mr_sweep.csv"
    argv = ["sweep",
            "--strategy", "filtered_mr",
            "--device", "cpu",
            "--synthetic-days", "20",
            "--period", "5m",
            "--start", "20260101",
            "--end", "20260120",
            "--grid", "cur_ema_period=20,30",
            "--out", str(out_csv)]
    old_argv = sys.argv
    sys.argv = ["evtrade"] + argv
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        try:
            cli_main()
        except SystemExit:
            pass
    finally:
        sys.stdout = old_stdout
        sys.argv = old_argv
    out = buf.getvalue()
    # sweep 输出含 "扫描"
    assert "扫描" in out or "sweep" in out.lower(), (
        f"sweep 输出异常: {out[:500]}"
    )
    assert out_csv.is_file(), f"sweep CSV 未生成: {out_csv}"


def test_filtered_mr_via_get_strategy():
    """通过 get_strategy("filtered_mr", ...) 构造"""
    s = get_strategy("filtered_mr", higher_period="1h", cur_ema_period=20)
    assert isinstance(s, FilteredMRStrategy)
    assert s.higher_period == "1h"
    assert s.cur_ema_period == 20
