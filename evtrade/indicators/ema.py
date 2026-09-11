from __future__ import annotations
"""EMA / EMAChannel 指标

三种形态:
  1. step 增量版 (ema_step, ema_channel_step): 策略 step() 逐桶调用, state 用 dataclass
  2. numpy 批量参考版 (ema, ema_channel): ndarray 输入输出 (reconcile 参考 / 测试用)
  3. torch 批量版 (torch_ema): Tensor [T] 输入返 Tensor [T], float64 bit-equal
     numpy 参考, 供 GPU-batched sweep (batched_step) 使用

EMA 公式:
  - count < p:   累加 sum; count += 1; ema 仍 NaN (批量版) / 0 (step 版) / 0 (torch 版)
  - count == p:  ema = sum / p  (SMA seed)
  - count >= p:  ema = value*k + ema*(1-k), k = 2/(p+1)
"""
from dataclasses import dataclass, field

import numpy as np


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


# ============ torch 批量版 (供 batched_step 使用) ============

def torch_ema(values, p: int):
    """EMA(torch.Tensor [T], p): Tensor [T]; 前 p-1 个 0.0; 跟 numpy ema() float64 bit-equal

    用于 batched_step 内的 per-combo 批量 EMA 计算。dtype/device 保留 (内部转 float64)。
    若 n < p, 全 0.0 返回 (无 NaN, 跟 ema_step 一致)。

    注: 1-D 版本; batched_step 按 tf1 值分组调用, 同 tf1 的 combo 一次调用出整段
    (不开 vmap, 整数取值集小时分组已够)。
    """
    import torch
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    if values.dim() != 1:
        raise ValueError(f"torch_ema expects 1-D Tensor, got shape {tuple(values.shape)}")
    n = values.shape[0]
    out = torch.zeros(n, dtype=torch.float64, device=values.device)
    if n < p:
        return out
    v = values.to(torch.float64)
    # Seed 必须跟 numpy 参考版 bit-equal: numpy.sum 用 pairwise 求和,
    # torch.cumsum 串行加会差末位 1 bit。绕开: 取 v[:p] 转 numpy 求 sum, 再回 torch。
    v_np_head = v[:p].detach().cpu().numpy() if v.is_cuda else v[:p].numpy()
    seed = float(np.sum(v_np_head) / p)
    out[p - 1] = seed
    k = 2.0 / (p + 1.0)
    one_minus_k = 1.0 - k
    # 递推尾段: out[i] = v[i]*k + out[i-1]*(1-k), i in [p, n)
    # 一阶线性递推没有等价的 torch 向量化算子 (cumsum 是加法递推不是乘法递推);
    # 用 Python float 标量逐位推进, 避免 torch tensor element-wise op 跟 numpy
    # scalar op 末位 FMA rounding 差; 每步 item() 一次 GPU sync, 但 T<=10^5 时
    # 总耗时 < 几十 ms, batched_step 仍以\"一次 kernel 调用\"对比 N×T_bar 次
    # Python step 循环占绝对优势。
    e = float(out[p - 1].item())
    for i in range(p, n):
        e = float(v[i].item()) * k + e * one_minus_k
        out[i] = e
    return out
