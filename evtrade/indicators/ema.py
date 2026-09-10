from __future__ import annotations
"""EMA / EMAChannel 指标

四种形态:
  1. xp 版 (xp_ema, xp_ema_channel): 兼容 numpy/torch 模块, 接收 xp 模块
  2. torch 版 (xp_ema_torch, xp_ema_channel_torch): 输入输出均为 torch.Tensor
  3. step 增量版 (ema_step, ema_channel_step): 策略 step() 调, state 用 dataclass
  4. 纯函数版 (ema, ema_channel): numpy 数组输入, ndarray 输出 (jupyter 用)

EMA 公式:
  - count < p:   累加 sum; count += 1; ema 仍 NaN (批量版) / 0 (step 版)
  - count == p:  ema = sum / p  (SMA seed)
  - count >= p:  ema = value*k + ema*(1-k), k = 2/(p+1)
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class EMAState:
    """EMA 增量 state; ema_step in/out"""
    sum: float = 0.0
    count: int = 0
    ema: float = 0.0


@dataclass
class EMAChannelState:
    """EMA 通道增量 state"""
    up: EMAState = field(default_factory=EMAState)
    dw: EMAState = field(default_factory=EMAState)


def _resolve_period_torch(p, B: int, device: torch.device) -> torch.Tensor:
    if isinstance(p, int):
        return torch.full((B,), int(p), dtype=torch.int64, device=device)
    return torch.as_tensor(p, dtype=torch.int64, device=device).flatten()


def xp_ema_torch(values: torch.Tensor, p) -> torch.Tensor:
    """EMA 批量版 (torch Tensor); 形状 (B, T) -> (B, T)。

    period 支持:
      - int: 所有行用同一 period
      - (B,) tensor / list: per-row period

    算法: 前缀和 + per-row period 切片 + 递推。
    返回: 前 p-1 根为 NaN, 之后为递推值 (B, T)。
    """
    if values.dim() == 1:
        values = values.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    B, T = values.shape
    device = values.device
    values_f = values.to(torch.float64)
    p_t = _resolve_period_torch(p, B, device)
    out = torch.full((B, T), float('nan'), dtype=torch.float64, device=device)

    p_idx = (p_t - 1).clamp(min=0)
    cumsum = torch.cumsum(values_f, dim=1)
    safe_idx = torch.clamp(p_idx, min=0, max=T - 1)
    seed_vals = cumsum.gather(1, safe_idx.unsqueeze(1)).squeeze(1) / p_t.to(torch.float64)

    col_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    seed_mask = col_idx == p_idx.unsqueeze(1)
    out = torch.where(seed_mask, seed_vals.unsqueeze(1).expand_as(out), out)

    k = 2.0 / (p_t.to(torch.float64) + 1.0)
    one_minus_k = 1.0 - k
    for t in range(int(p_t.max().item()), T):
        row_ready = t + 1 >= p_t
        if not row_ready.any():
            continue
        prev = out[:, t - 1]
        cur = values_f[:, t] * k + prev * one_minus_k
        out[:, t] = torch.where(row_ready, cur, out[:, t])

    if squeeze_back:
        out = out.squeeze(0)
    return out


def xp_ema_channel_torch(highs: torch.Tensor, lows: torch.Tensor, p) -> tuple:
    """EMA 通道 torch 版: (up, dw), 各 (B, T)。"""
    return xp_ema_torch(highs, p), xp_ema_torch(lows, p)


def xp_ema(xp, values, p: int):
    """EMA 批量版 (xp 兼容: numpy 或 torch 模块)

    接受旧签名 `xp_ema(xp, values, p)`; 内部用 xp 模块做前缀和 + 递推。
    返回: xp.ndarray, 形状 (T,) (1D 输入) 或 (B, T) (2D 输入); 前 p-1 根 NaN。
    """
    import numpy as _np
    # 把 numpy.ndarray 或 list 转 xp 数组
    if not hasattr(values, 'dtype') or isinstance(values, list):
        values = xp.asarray(values, dtype=_np.float64)

    # 1D 输入 -> (T,)
    if values.ndim == 1:
        n = values.shape[0]
        out = xp.full(n, _np.nan, dtype=_np.float64)
        if n < p:
            return out
        k = 2.0 / (p + 1.0)
        seed = xp.sum(values[:p]) / p
        out_idx = xp.int64(p - 1)
        out[out_idx] = seed
        e = float(seed)
        for i in range(p, n):
            e = float(values[i]) * k + e * (1.0 - k)
            out[xp.int64(i)] = e
        return out

    # 2D 输入 -> (B, T)
    B, T = values.shape
    out = xp.full((B, T), _np.nan, dtype=_np.float64)
    if T < p:
        return out
    k = 2.0 / (p + 1.0)
    cumsum = xp.cumsum(values, axis=1)
    p_idx = xp.arange(p - 1, T, dtype=_np.int64)  # (T-p+1,)
    seed_vals = cumsum[:, p - 1] / p  # (B,)
    out[:, p - 1] = seed_vals
    for t in range(p, T):
        e = values[:, t] * k + out[:, t - 1] * (1.0 - k)
        out[:, t] = e
    return out


def xp_ema_channel(xp, highs, lows, p: int):
    """EMA 通道 xp 版: (up, dw)"""
    return xp_ema(xp, highs, p), xp_ema(xp, lows, p)


def ema_step(state: EMAState, value: float, p: int) -> tuple:
    """EMA 单步: state + 1 标量 -> (new_state, ema_value)"""
    if state.count < p:
        new_sum = state.sum + value
        new_count = state.count + 1
        if new_count < p:
            return EMAState(sum=new_sum, count=new_count, ema=0.0), 0.0
        new_ema = new_sum / p
        return EMAState(sum=new_sum, count=new_count, ema=new_ema), new_ema
    k = 2.0 / (p + 1.0)
    new_ema = value * k + state.ema * (1.0 - k)
    return EMAState(sum=state.sum, count=state.count + 1, ema=new_ema), new_ema


def ema_channel_step(state: EMAChannelState, h: float, l: float,
                     p: int) -> tuple:
    new_up, up = ema_step(state.up, h, p)
    new_dw, dw = ema_step(state.dw, l, p)
    return EMAChannelState(up=new_up, dw=new_dw), up, dw


def ema(values, p: int):
    """EMA(values, p): numpy ndarray 输入返 ndarray。前 p-1 个 NaN。"""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < p:
        return out
    k = 2.0 / (p + 1.0)
    seed = np.sum(values[:p]) / p
    out[p - 1] = seed
    e = float(seed)
    for i in range(p, n):
        e = float(values[i]) * k + e * (1.0 - k)
        out[i] = e
    return out


def ema_channel(highs, lows, p: int):
    return ema(highs, p), ema(lows, p)