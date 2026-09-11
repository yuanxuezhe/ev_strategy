"""MACrossoverStrategy: 双均线交叉策略 (stateful step + batched_step)

EMA fast 上穿/slow -> BUY; 下穿 -> SELL; 每桶至多一个信号。

参数:
  fast  fast EMA 周期
  slow  slow EMA 周期
"""
from dataclasses import dataclass, field
from typing import Any

from ..indicators import EMAState, ema_step, torch_ema
from .vectorized_base import VectorizedStrategy, register_strategy


@dataclass
class MACrossoverState:
    """策略持久状态: fast EMA + slow EMA + 上一桶 diff"""
    fast: EMAState = field(default_factory=EMAState)
    slow: EMAState = field(default_factory=EMAState)
    prev_diff: float = 0.0
    has_prev: bool = False


@dataclass
class MABatchedState:
    """batched_step 状态: 跨桶需持续的最小集 (EMA 累加在 batched_step 内每次现算)

    每个字段都是 [N] Tensor (N = n_combos):
      fast_count / slow_count  每 combo 已推入 EMA 的桶数 (跟 step 同: mark=0 也推)
      prev_diff                上一根已就绪桶的 fast_ema - slow_ema
      has_prev                 是否已有 prev_diff 可比
    """
    fast_count: Any = None  # Tensor [N] int64
    slow_count: Any = None  # Tensor [N] int64
    prev_diff: Any = None   # Tensor [N] float64
    has_prev: Any = None    # Tensor [N] bool


def _batched_ema_by_group(close_t, period):
    """按 period 唯一值分组调 torch_ema, 拼回 [T, N]

    close_t: Tensor [T] float64
    period:  Tensor [N] int64
    返回:    Tensor [T, N] float64
    """
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
    }

    def init_state(self, params: dict) -> MACrossoverState:
        return MACrossoverState()

    def step(self, state: MACrossoverState, bar: dict, params: dict
             ) -> tuple[MACrossoverState, int]:
        fast_p = int(params["fast"])
        slow_p = int(params["slow"])
        close = float(bar["c"])

        # 推 EMA (mark=0 也推, 让 state 累积; 与原版 vectorized 路径语义一致)
        state.fast, fast = ema_step(state.fast, close, fast_p)
        state.slow, slow = ema_step(state.slow, close, slow_p)

        # 预热段 / 任一 EMA 未就绪: 不产信号, 也不记录 prev_diff
        # (快线先就绪时慢线 ema 仍为 0, 会让 diff 持号触发虚信号)
        if bar["mark"] == 0:
            return state, 0
        if state.fast.count < fast_p or state.slow.count < slow_p:
            return state, 0

        # 交叉检测: diff 符号变化 (首桶仅记录 prev_diff, 不产信号)
        diff = fast - slow
        if not state.has_prev:
            state.prev_diff = diff
            state.has_prev = True
            return state, 0

        sig = 0
        if diff > 0 and state.prev_diff <= 0:
            sig = 1   # BUY: 上穿
        elif diff < 0 and state.prev_diff >= 0:
            sig = -1  # SELL: 下穿

        state.prev_diff = diff
        return state, sig

    @classmethod
    def batched_step(cls, state: MABatchedState, bars: dict, params: dict,
                     *, n_combos: int, n_bars: int):
        """GPU 批量 hook: 一次调用产出 [T, N] int8 sig

        跟 step 同语义:
          - mark=0 段 fast/slow EMA 都累积 (state.fast_count/slow_count += 1)
          - 任一 EMA 未就绪: 该桶该 combo 产 sig=0, 不记录 prev_diff
          - 首桶两 EMA 同时就绪: 记录 prev_diff, 不产 sig, has_prev=True
          - 后续桶: cross 检测 diff 符号变化
        """
        import torch
        device = state.fast_count.device
        fast_p = params["fast"].to(device=device, dtype=torch.int64)
        slow_p = params["slow"].to(device=device, dtype=torch.int64)
        close_t = bars["c"].to(device=device, dtype=torch.float64)   # [T]
        mark_t = bars["mark"].to(device=device, dtype=torch.int8)    # [T]

        # 按 fast/slow 周期值分组算 EMA [T, N]
        fast_ema_TN = _batched_ema_by_group(close_t, fast_p)
        slow_ema_TN = _batched_ema_by_group(close_t, slow_p)

        diff_TN = fast_ema_TN - slow_ema_TN                # [T, N]

        sig = torch.zeros(n_combos, n_bars, dtype=torch.int8, device=device)

        # count 逐 bar 累积 (mark=0 也推, 跟 step 一致)
        fast_count = state.fast_count.clone()             # [N] int64
        slow_count = state.slow_count.clone()             # [N] int64
        prev_diff = state.prev_diff.clone()               # [N] float64
        has_prev = state.has_prev.clone()                 # [N] bool

        mark_ok_T = (mark_t != 0)                          # [T] bool

        # 逐 bar 处理 (T 串行; 每个 bar 内 N 维并行)
        # 注: cross 检测跟 step 同公式 (diff > 0 & prev_diff <= 0 -> BUY, 等)
        for t in range(n_bars):
            fast_count = fast_count + 1                    # mark=0 也累, 跟 step 一致
            slow_count = slow_count + 1
            if not bool(mark_ok_T[t]):
                continue
            ready_N = (fast_count >= fast_p) & (slow_count >= slow_p)   # [N] bool
            d = diff_TN[t]                                # [N]

            # BUY 候选: has_prev & ready & (d>0 & prev_diff<=0)
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

            # 更新 prev_diff / has_prev (仅 ready 时更新, 跟 step 同语义)
            prev_diff = torch.where(ready_N, d, prev_diff)
            has_prev = has_prev | ready_N

        new_state = MABatchedState(
            fast_count=fast_count,
            slow_count=slow_count,
            prev_diff=prev_diff,
            has_prev=has_prev,
        )
        return new_state, sig
