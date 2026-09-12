"""ChannelDeviationStrategy: 通道偏离回撤策略

EMA 通道 (stateful) + 偏离 (stateless) + 桶级锁存 FSM (stateful) +
自管资金/持仓/账本 (2026-09-13 重构)。

参数:
  low1/high1  下/上轨极端偏离阈值 (%)  触发锁存
  low2/high2  下/上轨回撤确认阈值 (%)  触发下单
  tf1         EMA 通道周期
  init_cash          初始资金 (默认 200000)
  init_position      初始持仓股数 (默认 0)
  trade_qty          每笔交易股数 (默认 10000)
  buy_pct            BUY 时按现金比例下注; 0=关闭走 trade_qty (默认 0)
  sell_pct           SELL 时按持仓比例卖; 0=关闭走 trade_qty (默认 0)
"""
from dataclasses import dataclass, field

from ..indicators import EMAChannelState, ema_channel_step
from ..primitives import fmt, sig_to_side
from .vectorized_base import VectorizedStrategy, register_strategy


# ============ 偏离量 (单桶 OHLCV vs 通道上/下轨) ============

@dataclass
class Deviations:
    """单桶 4 个偏离量 (单位 %); 通道未就绪时全 0"""
    low: float = 0.0      # (DW - L) / DW * 100  桶低 vs 下轨
    high: float = 0.0     # (H - UP) / UP * 100  桶高 vs 上轨
    low_h: float = 0.0    # (DW - H) / DW * 100  桶高回测下轨
    high_l: float = 0.0   # (L - UP) / UP * 100  桶低回测上轨


def _compute_devs(up: float, dw: float, h: float, l: float) -> Deviations:
    """通道上/下轨 + 单桶 OHLCV -> 4 个偏离"""
    if up == 0.0 or dw == 0.0:
        return Deviations()
    return Deviations(
        low=(dw - l) / dw * 100.0,
        high=(h - up) / up * 100.0,
        low_h=(dw - h) / dw * 100.0,
        high_l=(l - up) / up * 100.0,
    )


# ============ FSM 桶级锁存 ============

@dataclass
class DeviationFSM:
    """桶级锁存: 同一桶至多一次 BUY + 一次 SELL"""
    lock_ts: int = 0
    low_hit: bool = False
    high_hit: bool = False
    low_acted: bool = False
    high_acted: bool = False


def _fsm_step(fsm: DeviationFSM, cur_ts: int, devs: Deviations,
              low1: float, low2: float, high1: float, high2: float) -> int:
    """FSM 单步: 桶切换清锁 -> 触发检测/确认下单"""
    if cur_ts != fsm.lock_ts:
        fsm.lock_ts = cur_ts
        fsm.low_acted = False
        fsm.high_acted = False

    signal = 0
    if fsm.low_hit and devs.low_h < low2 and not fsm.low_acted:
        signal = 1
        fsm.low_hit = False
        fsm.low_acted = True
    elif fsm.high_hit and devs.high_l < high2 and not fsm.high_acted:
        signal = -1
        fsm.high_hit = False
        fsm.high_acted = True

    if devs.low > low1 and not fsm.low_acted:
        fsm.low_hit = True
        fsm.low_acted = True
    if devs.high > high1 and not fsm.high_acted:
        fsm.high_hit = True
        fsm.high_acted = True

    return signal


# ============ 撮合数学 (内联; 2026-09-13 framework 不再提供 trade_decision) ============

def _trade_decision(side: int, price: float, cash: float, position: float,
                    cur_qty: float, buy_pct: float, sell_pct: float
                    ) -> tuple[float, float, float, bool]:
    """单笔成交决策 (与旧 evtrade.execution.base.trade_decision 语义一致)

    side: 1=BUY, -1=SELL
    cur_qty: 本轮基础数量
    返回: (new_cash, new_position, qty, filled)
    """
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


# ============ 跨字段硬约束 ============

def _validate_latch_order(params: dict) -> None:
    low1, low2 = params["low1"], params["low2"]
    high1, high2 = params["high1"], params["high2"]
    if low1 <= low2:
        raise ValueError(
            f"channel_deviation: low1 ({low1}) 必须 > low2 ({low2}); "
            f"否则迟滞结构退化")
    if high1 <= high2:
        raise ValueError(
            f"channel_deviation: high1 ({high1}) 必须 > high2 ({high2}); "
            f"否则迟滞结构退化")


# ============ 策略持久状态 (含资金/持仓/账本) ============

@dataclass
class ChannelDeviationState:
    """策略持久状态: EMA 通道 + FSM + 桶切换检测 + 资金/持仓/账本"""
    ema: EMAChannelState = field(default_factory=EMAChannelState)
    fsm: DeviationFSM = field(default_factory=DeviationFSM)
    prev_ts: int = 0
    cur_high: float = 0.0
    cur_low: float = 0.0
    has_prev: bool = False
    # 资金 / 持仓 / 账本 (2026-09-13: framework 不再持这些; 策略自管)
    cash: float = 0.0
    position: float = 0.0
    init_cash: float = 0.0
    init_position: float = 0.0
    trades: list = field(default_factory=list)


# ============ 策略类 ============

@register_strategy("channel_deviation")
class ChannelDeviationStrategy(VectorizedStrategy):
    """通道偏离回撤策略"""

    params_spec = {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "low2":  {"default": 1.0, "type": float, "min": 0.0, "max": 100.0},
        "high1": {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "high2": {"default": 0.5, "type": float, "min": 0.0, "max": 100.0},
        "tf1":   {"default": 21,  "type": int,   "min": 2,   "max": 1000},
        # 资金/持仓/撮合参数 (2026-09-13: framework 不再有 CLI flag, 走 --params)
        "init_cash":     {"default": 200000.0, "type": float, "min": 0.0, "max": 1e12},
        "init_position": {"default": 0.0,     "type": float, "min": 0.0, "max": 1e9},
        "trade_qty":     {"default": 10000.0, "type": float, "min": 0.0, "max": 1e9},
        "buy_pct":       {"default": 0.0,     "type": float, "min": 0.0, "max": 1.0},
        "sell_pct":      {"default": 0.0,     "type": float, "min": 0.0, "max": 1.0},
    }

    validators = [_validate_latch_order]

    def init_state(self, params: dict) -> ChannelDeviationState:
        st = ChannelDeviationState()
        st.init_cash = float(params.get("init_cash", 0.0))
        st.init_position = float(params.get("init_position", 0.0))
        st.cash = st.init_cash
        st.position = st.init_position
        return st

    def step(self, state: ChannelDeviationState, bar: dict, params: dict
             ) -> tuple[ChannelDeviationState, int]:
        if bar["mark"] == 0:
            return state, 0

        tf1 = int(params["tf1"])
        cur_ts = int(bar["ts"])
        cur_high = float(bar["h"])
        cur_low = float(bar["l"])
        close = float(bar["c"])
        o = float(bar.get("o", close))
        v = float(bar.get("v", 0.0))

        # 桶切换: 旧桶 high/low 闭锁入 EMA
        if state.has_prev and state.prev_ts != cur_ts:
            state.ema, _, _ = ema_channel_step(
                state.ema, state.cur_high, state.cur_low, tf1)

        state.prev_ts = cur_ts
        state.cur_high = cur_high
        state.cur_low = cur_low
        state.has_prev = True

        # 当前通道值
        state.ema, up, dw = ema_channel_step(state.ema, cur_high, cur_low, tf1)
        devs = _compute_devs(up, dw, cur_high, cur_low)
        sig = _fsm_step(state.fsm, cur_ts, devs,
                        float(params["low1"]), float(params["low2"]),
                        float(params["high1"]), float(params["high2"]))

        # 撮合 (策略自负责)
        if sig != 0:
            trade_qty = float(params["trade_qty"])
            buy_pct = float(params["buy_pct"])
            sell_pct = float(params["sell_pct"])
            new_cash, new_pos, q, filled = _trade_decision(
                sig, close, state.cash, state.position,
                trade_qty, buy_pct, sell_pct)
            if filled:
                state.cash = new_cash
                state.position = new_pos
                state.trades.append({"ts": cur_ts,
                                     "side": "BUY" if sig > 0 else "SELL",
                                     "qty": float(q),
                                     "price": float(close)})

        # 暴露给 format_signal_line 的元数据 (含触发该 sig 的 finalized bar OHLCV)
        self._last_info = {"up": up, "dw": dw,
                           "low_dev": devs.low, "high_dev": devs.high,
                           "cash": state.cash, "position": state.position,
                           "side":  "BUY" if sig > 0 else ("SELL" if sig < 0 else ""),
                           "price": float(close),
                           "o": float(o), "h": float(cur_high),
                           "l": float(cur_low), "c": float(close),
                           "v": float(v)}
        return state, sig

    def format_signal_line(self, ts: int, sig: int, info: dict | None = None) -> str:
        """策略展示 hook: 一行展示方向 + ts + 价格 + 触发 K 线 OHLCV + 通道元数据"""
        info = info or {}
        side = sig_to_side(sig) or info.get("side", "") or ""
        prefix = f"{side} >>xxx> " if side else "             "
        return (f"{prefix}[{ts}] | price={fmt(info.get('price'))} "
                f"o={fmt(info.get('o'))} h={fmt(info.get('h'))} "
                f"l={fmt(info.get('l'))} c={fmt(info.get('c'))} "
                f"v={fmt(info.get('v'))} | "
                f"UP={fmt(info.get('up'))} DW={fmt(info.get('dw'))} | "
                f"low_dev(L/DW)={fmt(info.get('low_dev'))}% "
                f"high_dev(H/UP)={fmt(info.get('high_dev'))}% "
                f"cash={fmt(info.get('cash'))} pos={fmt(info.get('position'))}")