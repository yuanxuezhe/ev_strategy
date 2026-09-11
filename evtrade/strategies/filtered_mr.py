"""FilteredMRStrategy: 4 重过滤均值回归策略 (stateful step)

在 channel_deviation 基础上叠加 4 重过滤, 避免单边暴涨暴跌"接飞刀":

  1. 大周期顺势过滤   (--higher-period, --higher-ema-period)
  2. ADX 趋势强度过滤  (--adx-period, --adx-threshold)
  3. ATR 波动率过滤    (--atr-period, --atr-ma-period, --atr-vol-mult)
  4. Close 确认 FSM    (上一桶触轨 + 本桶 close 回到带内)

State dataclass 含大周期桶跟踪 / EMA / ATR Wilder / ADX Wilder / 触轨 FSM。
mark=0 段 EMA/ATR/ADX 仍累积 (跟 ma_crossover 同语义); 仅不产信号。

不实现 batched_step (FSM 难向量化, sweep 自动走 ThreadPool)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.timeutils import bucket_ts_encoded, resolve_period_seconds
from ..indicators import EMAState, ema_step
from .vectorized_base import VectorizedStrategy, register_strategy


# ============ State dataclass ============

@dataclass
class FilteredMRState:
    """filtered_mr 持久状态: 大周期桶 + 增量指标 + 触轨 FSM"""

    # ---- 大周期桶跟踪 ----
    higher_bucket_ts: int = 0            # 当前大周期桶 ts (整型)
    higher_close: float = 0.0           # 大周期桶已累积的 close (最后一根)
    higher_high: float = 0.0
    higher_low: float = 0.0
    higher_count: int = 0               # 当前大周期已累积 bar 数 (>0 才有桶 finalized)
    higher_ema_state: EMAState = field(default_factory=EMAState)
    higher_last_close: float = 0.0      # 上一大周期 finalized close (用于顺势判断)
    higher_has_ema: bool = False         # 大周期 EMA 是否就绪 (避免未就绪下判断顺势)

    # ---- 本周期 EMA ----
    cur_ema_state: EMAState = field(default_factory=EMAState)

    # ---- ATR (Wilder smoothing) ----
    atr: float = 0.0
    atr_prev_close: float = 0.0
    atr_count: int = 0
    atr_window: list = field(default_factory=list)   # ATR 历史窗口 (长度 = atr_ma_period)

    # ---- ADX (Wilder smoothing of +DM / -DM / TR / DX) ----
    plus_dm: float = 0.0
    minus_dm: float = 0.0
    tr_smooth: float = 0.0
    adx: float = 0.0
    adx_count: int = 0

    # ---- 触轨 FSM ----
    touched_lower_prev: bool = False
    touched_upper_prev: bool = False


# ============ 增量指标辅助 (per-bar, scalar) ============

def _update_atr(state: FilteredMRState, bar: dict, atr_period: int) -> FilteredMRState:
    """ATR Wilder 平滑: TR_t -> ATR_t = ATR_{t-1} + (TR_t - ATR_{t-1}) / period"""
    h = float(bar["h"]); l = float(bar["l"]); c = float(bar["c"])
    if state.atr_count == 0:
        # 首根 bar: TR = H - L (无 prev_close)
        tr = h - l
        state.atr = tr
        state.atr_prev_close = c
        state.atr_count = 1
        state.atr_window.append(tr)
        return state
    tr = max(h - l, abs(h - state.atr_prev_close), abs(l - state.atr_prev_close))
    state.atr = state.atr + (tr - state.atr) / atr_period
    state.atr_prev_close = c
    state.atr_count += 1
    state.atr_window.append(tr)
    return state


def _atr_ma(window: list, period: int) -> float:
    """ATR 滑动 SMA"""
    if len(window) < period:
        return 0.0
    return sum(window[-period:]) / float(period)


def _update_adx(state: FilteredMRState, bar: dict,
                adx_period: int, atr_period: int) -> FilteredMRState:
    """ADX Wilder 平滑: +DM/-DM/TR 平滑 -> +DI/-DI -> DX -> ADX 平滑"""
    h = float(bar["h"]); l = float(bar["l"]); c = float(bar["c"])
    # 用 ATR Wilder 的 prev_close 作为 ADX 的 prev_close (两者共享, 一致)
    prev_close = state.atr_prev_close if state.atr_count > 0 else c
    if state.adx_count == 0:
        # 首根 bar: 0 +DM/-DM/TR, 不更新
        state.adx_count = 1
        return state
    if state.adx_count == 1:
        # 第二根 bar: 算 +DM/-DM/TR 但用 SMA 初始化 (Wilder 标准)
        up = h - prev_close
        dn = prev_close - l
        plus_dm_t = max(up, 0.0) if up > dn else 0.0
        minus_dm_t = max(dn, 0.0) if dn > up else 0.0
        tr_t = max(h - l, abs(h - prev_close), abs(l - prev_close))
        state.plus_dm = plus_dm_t
        state.minus_dm = minus_dm_t
        state.tr_smooth = tr_t
        state.adx_count = 2
        return state
    # 之后: Wilder 平滑 (跟 ATR 同式)
    up = h - prev_close
    dn = prev_close - l
    plus_dm_t = max(up, 0.0) if up > dn else 0.0
    minus_dm_t = max(dn, 0.0) if dn > up else 0.0
    tr_t = max(h - l, abs(h - prev_close), abs(l - prev_close))
    state.plus_dm = state.plus_dm + (plus_dm_t - state.plus_dm) / adx_period
    state.minus_dm = state.minus_dm + (minus_dm_t - state.minus_dm) / adx_period
    state.tr_smooth = state.tr_smooth + (tr_t - state.tr_smooth) / adx_period
    state.adx_count += 1
    # DX & ADX 平滑 (Wilder 标准)
    if state.tr_smooth > 0 and state.adx_count >= adx_period:
        plus_di = 100.0 * state.plus_dm / state.tr_smooth
        minus_di = 100.0 * state.minus_dm / state.tr_smooth
        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
        if state.adx == 0.0:
            state.adx = dx
        else:
            state.adx = state.adx + (dx - state.adx) / adx_period
    return state


# ============ Strategy ============

@register_strategy("filtered_mr")
class FilteredMRStrategy(VectorizedStrategy):
    """4 重过滤均值回归策略"""

    params_spec = {
        "higher_period":    {"default": "1h", "type": str, "min": None, "max": None},
        "higher_ema_period": {"default": 60, "type": int, "min": 2, "max": 1000},
        "cur_ema_period":    {"default": 20, "type": int, "min": 2, "max": 1000},
        "atr_period":        {"default": 14, "type": int, "min": 2, "max": 1000},
        "atr_ma_period":     {"default": 20, "type": int, "min": 2, "max": 1000},
        "atr_vol_mult":      {"default": 1.5, "type": float, "min": 1.0, "max": 5.0},
        "adx_period":        {"default": 14, "type": int, "min": 2, "max": 1000},
        "adx_threshold":     {"default": 30.0, "type": float, "min": 10.0, "max": 100.0},
        "band_mult":         {"default": 1.8, "type": float, "min": 0.5, "max": 10.0},
    }

    def init_state(self, params: dict) -> FilteredMRState:
        return FilteredMRState()

    def step(self, state: FilteredMRState, bar: dict, params: dict
             ) -> tuple[FilteredMRState, int]:
        higher_p_sec = resolve_period_seconds(params["higher_period"])
        cur_ema_p = int(params["cur_ema_period"])
        higher_ema_p = int(params["higher_ema_period"])
        atr_p = int(params["atr_period"])
        atr_ma_p = int(params["atr_ma_period"])
        atr_vol_mult = float(params["atr_vol_mult"])
        adx_p = int(params["adx_period"])
        adx_th = float(params["adx_threshold"])
        band_mult = float(params["band_mult"])

        ts_int = int(bar["ts"])
        c = float(bar["c"])
        h = float(bar["h"])
        l = float(bar["l"])

        # ---- 1) 大周期桶跟踪 ----
        higher_ts = bucket_ts_encoded(ts_int, higher_p_sec)
        bucket_switched = (state.higher_count > 0
                           and higher_ts != state.higher_bucket_ts)
        if bucket_switched:
            # 旧桶 finalize: 用旧桶 close 推进 higher EMA
            state.higher_ema_state, _ = ema_step(
                state.higher_ema_state, state.higher_close, higher_ema_p)
            if state.higher_ema_state.count >= higher_ema_p:
                state.higher_has_ema = True
            state.higher_last_close = state.higher_close
            # 重置新桶 OHLC 累加
            state.higher_open = float(bar["o"])
            state.higher_high = h
            state.higher_low = l
            state.higher_count = 1
        else:
            # 同桶: 更新 H/L/C
            if state.higher_count == 0:
                state.higher_open = float(bar["o"])
                state.higher_high = h
                state.higher_low = l
            else:
                if h > state.higher_high:
                    state.higher_high = h
                if l < state.higher_low:
                    state.higher_low = l
            state.higher_count += 1
        state.higher_close = c
        state.higher_bucket_ts = higher_ts

        # ---- 2) 本周期 EMA 累积 (mark=0 也累积) ----
        state.cur_ema_state, _ = ema_step(state.cur_ema_state, c, cur_ema_p)

        # ---- 3) ATR / ADX 累积 (mark=0 也累积) ----
        state = _update_atr(state, bar, atr_p)
        state = _update_adx(state, bar, adx_p, atr_p)

        # ---- 4) mark=0 早返回 ----
        if bar["mark"] == 0:
            return state, 0

        # ---- 5) 指标就绪检查 ----
        ema_ready = state.cur_ema_state.count >= cur_ema_p and state.cur_ema_state.ema > 0
        atr_ready = state.atr_count >= atr_p and state.atr > 0
        adx_ready = state.adx_count >= 2 * adx_p and state.adx > 0
        atr_ma_ready = len(state.atr_window) >= atr_ma_p
        if not (ema_ready and atr_ready and atr_ma_ready):
            state.touched_lower_prev = False
            state.touched_upper_prev = False
            return state, 0

        # ---- 6) 4 重过滤矩阵 ----
        # 过滤 1: 大周期顺势
        higher_trend = 0
        if state.higher_has_ema:
            higher_trend = 1 if state.higher_last_close > state.higher_ema_state.ema else -1
        # 过滤 2: ADX 趋势强度
        is_strong_trend = adx_ready and state.adx > adx_th
        # 过滤 3: ATR 波动率
        atr_ma = _atr_ma(state.atr_window, atr_ma_p)
        is_high_vol = atr_ma > 0 and state.atr > atr_vol_mult * atr_ma

        if is_strong_trend or is_high_vol:
            state.touched_lower_prev = False
            state.touched_upper_prev = False
            return state, 0

        # ---- 7) 通道 + 触轨检测 ----
        upper = state.cur_ema_state.ema + band_mult * state.atr
        lower = state.cur_ema_state.ema - band_mult * state.atr
        touched_lower = l < lower
        touched_upper = h > upper
        close_in_lower = c > lower   # 本桶 close 回到带内 (反转确认)
        close_in_upper = c < upper

        # ---- 8) Close 确认 FSM + 顺势过滤 ----
        sig = 0
        if state.touched_lower_prev and touched_lower and close_in_lower:
            # 大周期多头或震荡 (higher_trend >= 0) → 允许低吸做多
            if higher_trend >= 0:
                sig = 1
        elif state.touched_upper_prev and touched_upper and close_in_upper:
            # 大周期空头或震荡 (higher_trend <= 0) → 允许高抛做空
            if higher_trend <= 0:
                sig = -1

        # 更新触轨 FSM
        state.touched_lower_prev = touched_lower
        state.touched_upper_prev = touched_upper
        return state, sig
