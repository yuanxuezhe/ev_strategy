from __future__ import annotations
"""EMA 与通道轨指标 (增量状态机版, 旧 indicators.py)

================================================================
⚠️  差分锁定参考实现  ⚠️  (原 frozen/incremental_indicators.py, 2026-09 重构迁移)
================================================================
本文件原名 frozen/_incremental_indicators.py (旧顶层 evtrade._incremental_indicators),
原名 indicators.py, 因新增 indicators/ 子包 (纯函数库),
被重命名为下划线前缀以避免命名冲突。

本文件定义的 IncrementalEMA / EMAChannel 是**热路径用** (numba njit 友好),
evtrade.kernel._ema_push / _ema_current 与之**逐位等价**,
由 tests/test_differential.py + tests/test_kernel_unit.py 锁定。

任何修改必须**同步**改以下位置:
  - evtrade/core/incremental_indicators.py (本文件)
  - evtrade/core/kernel.py 的 _ema_push / _ema_current
  - evtrade/core/gpu.py CUDA source 的对应段
  - kbs/05-指标计算-EMA通道.md

新增指标请在 indicators/ 子包写纯函数版 (jupyter 友好);
不要在本文件加新类 (会破坏 numba 流式内核的对账)。
================================================================
"""


# ============ 指标 (纯函数) ============

def ema(values: list[float], p: int):
    """EMA: 前 p 个 SMA 做 seed, 之后 EMA = price*k + EMA_prev*(1-k), k=2/(p+1)

    保留为纯函数 (用于一次性计算/校验)。热路径用 IncrementalEMA 增量化。
    """
    if len(values) < p:
        return None
    k = 2 / (p + 1)
    e = sum(values[:p]) / p          # SMA seed
    for v in values[p:]:
        e = v * k + e * (1 - k)
    return e


class IncrementalEMA:
    """增量 EMA 状态 (O(1)/桶, 替代每根 O(n) 重算)

    维护已闭合桶序列的 EMA:
      - 前 p-1 个桶: 累加 sum, 未出 EMA
      - 第 p 个桶: seed = sum/p, 出首个 EMA
      - 之后每个闭合桶: EMA = price*k + EMA_prev*(1-k), k=2/(p+1)
    当前未闭合桶: 基于已闭合 EMA 做一次临时递推 (不修改状态)。
    """

    def __init__(self, p: int):
        self.p = p
        self.k = 2 / (p + 1)
        self.sum = 0.0          # 前 p 个值的累加 (seed 用)
        self.count = 0          # 已闭合桶数
        self.ema = None         # 已闭合序列的最后一个 EMA (count>=p 时有值)

    def push(self, value: float):
        """闭合一个新桶, O(1) 更新 EMA 状态"""
        if self.count < self.p:
            self.sum += value
            self.count += 1
            if self.count == self.p:
                self.ema = self.sum / self.p   # SMA seed
        else:
            self.ema = value * self.k + self.ema * (1 - self.k)
            self.count += 1

    def current(self, pending_value: float):
        """带当前未闭合桶最新值的 EMA (不修改状态, O(1))

        pending_value: 当前桶最新 H 或 L。
        返回: 若已闭合 count>=p-1 则基于 pending 递推一次得到当前 EMA; 否则 None。
        """
        if self.count < self.p - 1:
            return None
        if self.count == self.p - 1:
            # 闭合了 p-1 个, 加当前 pending 凑齐 p 个 -> SMA seed
            return (self.sum + pending_value) / self.p
        # count >= p: 已有 ema, 用 pending 递推一次
        return pending_value * self.k + self.ema * (1 - self.k)


class EMAChannel:
    """通达信蓝色通道轨 (增量版)

    UP1 := EMA(H, TF1) 上轨, DW1 := EMA(L, TF1) 下轨
    桶闭合时 push(h, l); 每根 bar 调 channel(cur_h, cur_l) 得当前 (UP, DW), O(1)。
    """

    def __init__(self, tf1: int = 21):
        self.tf1 = tf1
        self.up_ema = IncrementalEMA(tf1)
        self.dw_ema = IncrementalEMA(tf1)

    def push(self, high: float, low: float):
        """桶闭合时调用"""
        self.up_ema.push(high)
        self.dw_ema.push(low)

    def channel(self, cur_high: float, cur_low: float):
        """返回当前 (UP, DW), 含未闭合桶最新值, O(1)。数据不足返回 (None, None)"""
        up = self.up_ema.current(cur_high)
        dw = self.dw_ema.current(cur_low)
        return up, dw


def ema_channel(bars: list[dict], tf1: int = 21):
    """通达信蓝色通道轨 (纯函数版, 向后兼容/校验用)

    UP1 := EMA(H, TF1)   上轨
    DW1 := EMA(L, TF1)   下轨
    入参: bars 周期K线集合 (末位=当前桶最新), tf1 EMA 周期
    出参: (UP, DW), 数据不足返回 (None, None)
    """
    if len(bars) < tf1:
        return None, None
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    return ema(highs, tf1), ema(lows, tf1)
