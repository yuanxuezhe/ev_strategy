from __future__ import annotations
"""通道偏离回撤策略 (顶层兼容导出, **冻结版**)

================================================================
⚠️⚠️  冻结层 (差分测试锁定)  ⚠️⚠️
================================================================
本文件的 ChannelDeviationStrategy 与 evtrade.kernel._strategy_check 是
**逐位等价**的两个实现 (tests/test_differential.py 等 56 项锁定)。

⚠️ 任何修改都会触发差分测试全红 ⚠️

新策略应放在 evtrade/strategies/ 子包下, 用 @register_strategy 注册;
本文件不要扩展新策略, 只保留作为 evtrade 顶层兼容导出。
================================================================
"""


class ChannelDeviationStrategy:
    """通道偏离回撤策略 (有状态) —— **冻结版, 与 kernel 逐位等价**

    ⚠️ 不要改这一份 ⚠️ 改 strategies/channel_deviation.py 不会改变这里 (本类独立)。
    """

    def __init__(self, low1=1.5, low2=1, high1=1.5, high2=0.5):
        self.low1, self.low2 = low1, low2
        self.high1, self.high2 = high1, high2
        self.low_hit = False      # 下轨极端偏离锁存标记
        self.high_hit = False     # 上轨极端偏离锁存标记
        self._bucket_ts = None
        self._low_acted = False
        self._high_acted = False

    def check(self, cur: dict, up, dw):
        """返回 (signal, info) signal 为 'BUY'/'SELL'/None"""
        if up is None or dw is None or up == 0 or dw == 0:
            return None, {}
        # 桶切换: 重置本桶操作锁
        if cur["ts"] != self._bucket_ts:
            self._bucket_ts = cur["ts"]
            self._low_acted = False
            self._high_acted = False

        low_dev = (dw - cur["low"]) / dw * 100
        high_dev = (cur["high"] - up) / up * 100
        low_dev_h = (dw - cur["high"]) / dw * 100
        high_dev_l = (cur["low"] - up) / up * 100
        info = {"low_dev": low_dev, "high_dev": high_dev,
                "low_dev_h": low_dev_h, "high_dev_l": high_dev_l,
                "low_hit_prev": self.low_hit, "high_hit_prev": self.high_hit}

        signal = None
        if self.low_hit and low_dev_h < self.low2 and not self._low_acted:
            signal = "BUY"
            self.low_hit = False
            self._low_acted = True
        elif self.high_hit and high_dev_l < self.high2 and not self._high_acted:
            signal = "SELL"
            self.high_hit = False
            self._high_acted = True

        if low_dev > self.low1 and not self._low_acted:
            self.low_hit = True
            self._low_acted = True
        if high_dev > self.high1 and not self._high_acted:
            self.high_hit = True
            self._high_acted = True
        return signal, info


__all__ = ["ChannelDeviationStrategy"]
