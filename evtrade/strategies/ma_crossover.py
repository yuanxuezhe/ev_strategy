"""MACrossoverStrategy: 双均线交叉策略 (stateful step + batched_step)

EMA fast 上穿/slow -> BUY; 下穿 -> SELL; 每桶至多一个信号。
2026-09-13: 策略自管资金/持仓/账本。

参数:
  fast          fast EMA 周期
  slow          slow EMA 周期
  init_cash     初始资金 (默认 200000)
  init_position 初始持仓股数 (默认 0)
  trade_qty     每笔交易股数 (默认 10000)
  buy_pct       BUY 比例; 0=关闭走 trade_qty
  sell_pct      SELL 比例; 0=关闭走 trade_qty
"""
from dataclasses import dataclass, field
from typing import Any

from ..indicators import EMAState, ema_step, torch_ema
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


@dataclass
class MACrossoverState:
    """策略持久状态: fast EMA + slow EMA + 上一桶 diff + 资金/持仓/账本"""
    fast: EMAState = field(default_factory=EMAState)
    slow: EMAState = field(default_factory=EMAState)
    prev_diff: float = 0.0
    has_prev: bool = False
    cash: float = 0.0
    position: float = 0.0
    init_cash: float = 0.0
    init_position: float = 0.0
    trades: list = field(default_factory=list)


@dataclass
class MABatchedState:
    """batched_step 状态 (仅 GPU 化指标累加, 不含 cash/position; 现金/持仓由串行 step 补)"""
    fast_count: Any = None  # Tensor [N] int64
    slow_count: Any = None  # Tensor [N] int64
    prev_diff: Any = None   # Tensor [N] float64
    has_prev: Any = None    # Tensor [N] bool


def _batched_ema_by_group(close_t, period):
    """按 period 唯一值分组调 torch_ema, 拼回 [T, N]"""
    import torch
    n_bars, n_combos = close_t.shape[0], period.shape[0]
    out = torch.empty(n_bars, n_combos, dtype=torch.float64, device=close_t.device)
    unique_p, inverse = torch.unique(period, return_inverse=True)
    for j in range(unique_p.shape[0]):
        p = int(unique_p[j].item())
        mask = (inverse == j)
        ema_col = torch_ema(close_t, p)             # [T]
        out[:, mask] = ema_col.unsqueeze(1)
    return out


@register_strategy("ma_crossover")
class MACrossoverStrategy(VectorizedStrategy):
    """双均线交叉策略"""

    params_spec = {
        "fast": {"default": 5,  "type": int, "min": 2, "max": 1000},
        "slow": {"default": 20, "type": int, "min": 2, "max": 1000},
        "init_cash":     {"default": 200000.0, "type": float, "min": 0.0, "max": 1e12},
        "init_position": {"default": 0.0,     "type": float, "min": 0.0, "max": 1e9},
        "trade_qty":     {"default": 10000.0, "type": float, "min": 0.0, "max": 1e9},
        "buy_pct":       {"default": 0.0,     "type": float, "min": 0.0, "max": 1.0},
        "sell_pct":      {"default": 0.0,     "type": float, "min": 0.0, "max": 1.0},
    }

    def init_state(self, params: dict) -> MACrossoverState:
        st = MACrossoverState()
        st.init_cash = float(params.get("init_cash", 0.0))
        st.init_position = float(params.get("init_position", 0.0))
        st.cash = st.init_cash
        st.position = st.init_position
        return st

    def step(self, state: MACrossoverState, bar: dict, params: dict
             ) -> tuple[MACrossoverState, int]:
        fast_p = int(params["fast"])
        slow_p = int(params["slow"])
        close = float(bar["c"])

        # 推 EMA (mark=0 也推, 让 state 累积)
        state.fast, fast = ema_step(state.fast, close, fast_p)
        state.slow, slow = ema_step(state.slow, close, slow_p)

        # 预热段 / 任一 EMA 未就绪: 不产信号
        if bar["mark"] == 0:
            return state, 0
        if state.fast.count < fast_p or state.slow.count < slow_p:
            return state, 0

        diff = fast - slow
        if not state.has_prev:
            state.prev_diff = diff
            state.has_prev = True
            return state, 0

        sig = 0
        if diff > 0 and state.prev_diff <= 0:
            sig = 1
        elif diff < 0 and state.prev_diff >= 0:
            sig = -1

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
                state.trades.append({"ts": int(bar["ts"]),
                                     "side": "BUY" if sig > 0 else "SELL",
                                     "qty": float(q),
                                     "price": float(close)})

        state.prev_diff = diff
        return state, sig

    @classmethod
    def batched_step(cls, state: MABatchedState, bars: dict, params: dict,
                     *, n_combos: int, n_bars: int):
        """GPU 批量 hook: 一次调用产出 [T, N] int8 sig (仅产 sig, 不计 cash/position)"""
        import torch
        device = state.fast_count.device
        fast_p = params["fast"].to(device=device, dtype=torch.int64)
        slow_p = params["slow"].to(device=device, dtype=torch.int64)
        close_t = bars["c"].to(device=device, dtype=torch.float64)   # [T]
        mark_t = bars["mark"].to(device=device, dtype=torch.int8)    # [T]

        fast_ema_TN = _batched_ema_by_group(close_t, fast_p)
        slow_ema_TN = _batched_ema_by_group(close_t, slow_p)

        diff_TN = fast_ema_TN - slow_ema_TN                # [T, N]

        sig = torch.zeros(n_combos, n_bars, dtype=torch.int8, device=device)

        fast_count = state.fast_count.clone()
        slow_count = state.slow_count.clone()
        prev_diff = state.prev_diff.clone()
        has_prev = state.has_prev.clone()

        mark_ok_T = (mark_t != 0)                          # [T] bool

        for t in range(n_bars):
            fast_count = fast_count + 1
            slow_count = slow_count + 1
            if not bool(mark_ok_T[t]):
                continue
            ready_N = (fast_count >= fast_p) & (slow_count >= slow_p)
            d = diff_TN[t]

            buy_mask = has_prev & ready_N & (d > 0) & (prev_diff <= 0)
            sell_mask = has_prev & ready_N & (d < 0) & (prev_diff >= 0)

            sig_t = torch.where(
                buy_mask, torch.ones_like(buy_mask, dtype=torch.int8),
                torch.where(
                    sell_mask, -torch.ones_like(sell_mask, dtype=torch.int8),
                    torch.zeros_like(buy_mask, dtype=torch.int8),
                ),
            )
            sig[:, t] = sig_t

            prev_diff = torch.where(ready_N, d, prev_diff)
            has_prev = has_prev | ready_N

        new_state = MABatchedState(
            fast_count=fast_count,
            slow_count=slow_count,
            prev_diff=prev_diff,
            has_prev=has_prev,
        )
        return new_state, sig