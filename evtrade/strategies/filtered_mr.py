"""FilteredMRStrategy: 4 重过滤均值回归策略 (stateful step)

2026-09-13: 策略自管资金/持仓/账本。

在 channel_deviation 基础上叠加 4 重过滤, 避免单边暴涨暴跌"接飞刀":
  1. 大周期顺势过滤
  2. ADX 趋势强度过滤
  3. ATR 波动率过滤
  4. Close 确认 FSM

不实现 batched_step (FSM 难向量化, sweep 自动走 ThreadPool)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.timeutils import bucket_ts_encoded, resolve_period_seconds
from ..indicators import EMAState, ema_step
from ..primitives import fmt, sig_to_side
from .vectorized_base import VectorizedStrategy, register_strategy


# ============ 撮合数学 (内联) ============

def _trade_decision(side: int, price: float, cash: float, position: float,
                    cur_qty: float, buy_pct: float, sell_pct: float
                    ) -> tuple[float, float, float, bool]:
    if side > 0:
        if price > 0:
            max_by_cash = cash / price
            if buy_pct > 0:
                q = min(buy_pct * max_by_cash, max_by_cash)
            else:
                q = min(cur_qty, max_by_cash)
        else:
            q = 0.0
        if q <= 0:
            return cash, position, 0.0, False
        return cash - q * price, position + q, q, True
    if sell_pct > 0:
        q = min(sell_pct * position, position)
    else:
        q = min(cur_qty, position)
    if q <= 0:
        return cash, position, 0.0, False
    return cash + q * price, position - q, q, True


# ============ State dataclass (含资金/持仓/账本) ============

@dataclass
class FilteredMRState:
    """filtered_mr 持久状态: 大周期桶 + 增量指标 + 触轨 FSM + 资金/持仓/账本"""
    higher_bucket_ts: int = 0
    higher_close: float = 0.0
    higher_high: float = 0.0
    higher_low: float = 0.0
    higher_count: int = 0
    higher_ema_state: EMAState = field(default_factory=EMAState)
    higher_last_close: float = 0.0
    higher_has_ema: bool = False

    cur_ema_state: EMAState = field(default_factory=EMAState)

    atr: float = 0.0
    atr_prev_close: float = 0.0
    atr_count: int = 0
    atr_window: list = field(default_factory=list)

    plus_dm: float = 0.0
    minus_dm: float = 0.0
    tr_smooth: float = 0.0
    adx: float = 0.0
    adx_count: int = 0

    touched_lower_prev: bool = False
    touched_upper_prev: bool = False

    # 资金 / 持仓 / 账本 (2026-09-13)
    cash: float = 0.0
    position: float = 0.0
    init_cash: float = 0.0
    init_position: float = 0.0
    trades: list = field(default_factory=list)


# ============ 增量指标辅助 (per-bar, scalar) ============

def _update_atr(state: FilteredMRState, bar: dict, atr_period: int) -> FilteredMRState:
    h = float(bar["h"]); l = float(bar["l"]); c = float(bar["c"])
    if state.atr_count == 0:
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
    if len(window) < period:
        return 0.0
    return sum(window[-period:]) / float(period)


def _update_adx(state: FilteredMRState, bar: dict,
                adx_period: int, atr_period: int) -> FilteredMRState:
    h = float(bar["h"]); l = float(bar["l"]); c = float(bar["c"])
    prev_close = state.atr_prev_close if state.atr_count > 0 else c
    if state.adx_count == 0:
        state.adx_count = 1
        return state
    if state.adx_count == 1:
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
    up = h - prev_close
    dn = prev_close - l
    plus_dm_t = max(up, 0.0) if up > dn else 0.0
    minus_dm_t = max(dn, 0.0) if dn > up else 0.0
    tr_t = max(h - l, abs(h - prev_close), abs(l - prev_close))
    state.plus_dm = state.plus_dm + (plus_dm_t - state.plus_dm) / adx_period
    state.minus_dm = state.minus_dm + (minus_dm_t - state.minus_dm) / adx_period
    state.tr_smooth = state.tr_smooth + (tr_t - state.tr_smooth) / adx_period
    state.adx_count += 1
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

    params_spec = {
        "higher_period":    {"default": "1h", "type": str, "min": None, "max": None,
                             "desc": "大周期(顺势过滤); 任意 m/h/d 字符串, 走 resolve_period_seconds 解析"},
        "higher_ema_period": {"default": 60, "type": int, "min": 2, "max": 1000,
                              "desc": "大周期 EMA 周期; 顺势判定 = 大周期末根 close 与 higher_ema 比较"},
        "cur_ema_period":    {"default": 20, "type": int, "min": 2, "max": 1000,
                              "desc": "本周期 EMA 周期; 通道中轨"},
        "atr_period":        {"default": 14, "type": int, "min": 2, "max": 1000,
                              "desc": "ATR Wilder 平滑周期; 滚动真实波幅"},
        "atr_ma_period":     {"default": 20, "type": int, "min": 2, "max": 1000,
                              "desc": "ATR 滑动 MA 周期; 波动率基线"},
        "atr_vol_mult":      {"default": 1.5, "type": float, "min": 1.0, "max": 5.0,
                              "desc": "ATR 异常放大阈值倍数; ATR > atr_ma × mult 暂停 (极端暴动过滤)"},
        "adx_period":        {"default": 14, "type": int, "min": 2, "max": 1000,
                              "desc": "ADX Wilder 平滑周期; 趋势强度"},
        "adx_threshold":     {"default": 30.0, "type": float, "min": 10.0, "max": 100.0,
                              "desc": "ADX 强趋势阈值; ADX > threshold 时仅允许顺势侧信号"},
        "band_mult":         {"default": 1.8, "type": float, "min": 0.5, "max": 10.0,
                              "desc": "通道上下轨乘数; 上轨 = cur_ema + band_mult × ATR, 下轨 = cur_ema - band_mult × ATR"},
        "init_cash":         {"default": 200000.0, "type": float, "min": 0.0, "max": 1e12,
                              "desc": "期初资金(元); 策略 state 自管, framework 不汇总"},
        "init_position":     {"default": 0.0,     "type": float, "min": 0.0, "max": 1e9,
                              "desc": "期初持仓股数; 默认 0 空仓起步"},
        "trade_qty":         {"default": 10000.0, "type": float, "min": 0.0, "max": 1e9,
                              "desc": "每笔交易股数; buy_pct/sell_pct=0 时生效"},
        "buy_pct":           {"default": 0.0,     "type": float, "min": 0.0, "max": 1.0,
                              "desc": "BUY 时按当前 cash 比例下注; 0=关闭走 trade_qty, 1=全仓"},
        "sell_pct":          {"default": 0.0,     "type": float, "min": 0.0, "max": 1.0,
                              "desc": "SELL 时按当前 position 比例卖; 0=关闭走 trade_qty, 1=全清"},
    }

    def format_signal_line(self, ts: int, sig: int, info: dict | None = None) -> str:
        """策略展示 hook: 一行展示方向 + ts + 价格 + 触发该 sig 的 K 线 OHLCV
        (与 channel_deviation 视觉一致; sig==0 不被 framework 调用但仍防御)
        """
        info = info or {}
        side = sig_to_side(sig) or info.get("side", "") or ""
        prefix = f"{side} >>xxx> " if side else "             "
        return (f"{prefix}[{ts}] | price={fmt(info.get('price'))} "
                f"o={fmt(info.get('o'))} h={fmt(info.get('h'))} "
                f"l={fmt(info.get('l'))} c={fmt(info.get('c'))} "
                f"v={fmt(info.get('v'))}")

    def init_state(self, params: dict) -> FilteredMRState:
        st = FilteredMRState()
        st.init_cash = float(params.get("init_cash", 0.0))
        st.init_position = float(params.get("init_position", 0.0))
        st.cash = st.init_cash
        st.position = st.init_position
        return st

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
        o = float(bar.get("o", c))
        v = float(bar.get("v", 0.0))

        # 入口先写 _last_info: 所有早返回路径也能让 hook 拿到本 bar OHLCV
        # (spec R "Strategy signal line shows trigger bar context" Scenario
        #  `_last_info 字段集与 step 触发 bar 一致`)
        self._last_info = {
            "side":  "",  # sig 未知, 先空; 真正触发时再覆写
            "price": float(c),
            "o": float(o), "h": float(h), "l": float(l), "c": float(c),
            "v": float(v),
        }

        # ---- 大周期桶跟踪 ----
        higher_ts = bucket_ts_encoded(ts_int, higher_p_sec)
        bucket_switched = (state.higher_count > 0
                           and higher_ts != state.higher_bucket_ts)
        if bucket_switched:
            state.higher_ema_state, _ = ema_step(
                state.higher_ema_state, state.higher_close, higher_ema_p)
            if state.higher_ema_state.count >= higher_ema_p:
                state.higher_has_ema = True
            state.higher_last_close = state.higher_close
            state.higher_open = float(bar["o"])
            state.higher_high = h
            state.higher_low = l
            state.higher_count = 1
        else:
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

        state.cur_ema_state, _ = ema_step(state.cur_ema_state, c, cur_ema_p)
        state = _update_atr(state, bar, atr_p)
        state = _update_adx(state, bar, adx_p, atr_p)

        if bar["mark"] == 0:
            return state, 0

        ema_ready = state.cur_ema_state.count >= cur_ema_p and state.cur_ema_state.ema > 0
        atr_ready = state.atr_count >= atr_p and state.atr > 0
        adx_ready = state.adx_count >= 2 * adx_p and state.adx > 0
        atr_ma_ready = len(state.atr_window) >= atr_ma_p
        if not (ema_ready and atr_ready and atr_ma_ready):
            state.touched_lower_prev = False
            state.touched_upper_prev = False
            return state, 0

        higher_trend = 0
        if state.higher_has_ema:
            higher_trend = 1 if state.higher_last_close > state.higher_ema_state.ema else -1
        is_strong_trend = adx_ready and state.adx > adx_th
        atr_ma = _atr_ma(state.atr_window, atr_ma_p)
        is_high_vol = atr_ma > 0 and state.atr > atr_vol_mult * atr_ma

        if is_strong_trend or is_high_vol:
            state.touched_lower_prev = False
            state.touched_upper_prev = False
            return state, 0

        upper = state.cur_ema_state.ema + band_mult * state.atr
        lower = state.cur_ema_state.ema - band_mult * state.atr
        touched_lower = l < lower
        touched_upper = h > upper
        close_in_lower = c > lower
        close_in_upper = c < upper

        sig = 0
        if state.touched_lower_prev and touched_lower and close_in_lower:
            if higher_trend >= 0:
                sig = 1
        elif state.touched_upper_prev and touched_upper and close_in_upper:
            if higher_trend <= 0:
                sig = -1

        state.touched_lower_prev = touched_lower
        state.touched_upper_prev = touched_upper

        # 撮合 (策略自负责)
        if sig != 0:
            trade_qty = float(params["trade_qty"])
            buy_pct = float(params["buy_pct"])
            sell_pct = float(params["sell_pct"])
            new_cash, new_pos, q, filled = _trade_decision(
                sig, c, state.cash, state.position,
                trade_qty, buy_pct, sell_pct)
            if filled:
                state.cash = new_cash
                state.position = new_pos
                state.trades.append({"ts": ts_int,
                                     "side": "BUY" if sig > 0 else "SELL",
                                     "qty": float(q),
                                     "price": float(c)})

        # 覆写 _last_info["side"] (入口已先写一次, sig 已知, 补 side 字段)
        self._last_info["side"] = "BUY" if sig > 0 else ("SELL" if sig < 0 else "")
        return state, sig