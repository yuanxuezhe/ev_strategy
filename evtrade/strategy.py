from __future__ import annotations
"""通道偏离回撤策略 (自 mysql_analyze_demo.py 原样迁移)"""


# ============ 策略 (有状态, 独立于行情源和账户) ============

class ChannelDeviationStrategy:
    """通道偏离回撤策略 (有状态)

    偏离定义 (百分比):
      low_dev  = (DW - L) / DW * 100   # L 低于下轨的程度 (>0 为下偏), 用于 low1 极端偏离判定
      high_dev = (H - UP) / UP * 100   # H 高于上轨的程度 (>0 为上偏), 用于 high1 极端偏离判定
      low_dev_h  = (DW - H) / DW * 100 # H 相对下轨的回撤偏离, 用于 low2 BUY 触发
      high_dev_l = (L - UP) / UP * 100 # L 相对上轨的回撤偏离, 用于 high2 SELL 触发

    逻辑:
      当 low_dev  > low1%  -> 标记 low_hit  (下轨极端偏离, 用最低价 L)
      当 high_dev > high1% -> 标记 high_hit (上轨极端偏离, 用最高价 H)
      下一根 low_dev_h  < low2%  -> BUY  (最高价 H 回升到下轨上方, 整根bar回通道内)
      下一根 high_dev_l < high2% -> SELL (最低价 L 回落到上轨下方, 整根bar回通道内)

    信号方向:
      BUY  = 超跌反弹
      SELL = 超涨回落
    """

    def __init__(self, low1=1.5, low2=1, high1=1.5, high2=0.5):
        self.low1, self.low2 = low1, low2
        self.high1, self.high2 = high1, high2
        self.low_hit = False      # 下轨极端偏离锁存标记
        self.high_hit = False     # 上轨极端偏离锁存标记
        # 同一根周期桶内每个状态只允许一次状态转换 (置位 或 触发解除, 二选一)
        self._bucket_ts = None
        self._low_acted = False
        self._high_acted = False

    def check(self, cur: dict, up, dw):
        """返回 (signal, info) signal 为 'BUY'/'SELL'/None

        锁存逻辑 (hysteresis, 需 low1 > low2 / high1 > high2):
          下轨: low_dev > low1 -> low_hit=True (置位, 保持)
                low_hit 且 low_dev_h < low2 -> BUY 并 low_hit=False (H 回升过下轨, 解除)
          上轨: high_dev > high1 -> high_hit=True (置位, 保持)
                high_hit 且 high_dev_l < high2 -> SELL 并 high_hit=False (L 回落过上轨, 解除)
        即标记一旦置位, 不会因 low_dev 跌破 low1 而清除, 必须等到回撤 < low2 才触发并解除。

        单桶单操作: 同一根周期K线桶内, low_hit / high_hit 各自只允许一次状态转换
          (置位true 或 触发信号并置false), 二者互斥, 防止桶内 H/L 大幅波动反复触发。
          桶切换(ts变化)时重置本桶操作锁。
        """
        if up is None or dw is None or up == 0 or dw == 0:
            return None, {}
        # 桶切换: 重置本桶操作锁
        if cur["ts"] != self._bucket_ts:
            self._bucket_ts = cur["ts"]
            self._low_acted = False
            self._high_acted = False

        low_dev = (dw - cur["low"]) / dw * 100
        high_dev = (cur["high"] - up) / up * 100
        low_dev_h = (dw - cur["high"]) / dw * 100    # BUY 触发用: H vs DW
        high_dev_l = (cur["low"] - up) / up * 100    # SELL 触发用: L vs UP
        info = {"low_dev": low_dev, "high_dev": high_dev,
                "low_dev_h": low_dev_h, "high_dev_l": high_dev_l,
                "low_hit_prev": self.low_hit, "high_hit_prev": self.high_hit}

        signal = None
        # 信号触发 + 解除锁存: 基于上一根的 hit 标记, 当前根回撤入场 (本桶未操作过才触发)
        if self.low_hit and low_dev_h < self.low2 and not self._low_acted:
            signal = "BUY"
            self.low_hit = False
            self._low_acted = True
        elif self.high_hit and high_dev_l < self.high2 and not self._high_acted:
            signal = "SELL"
            self.high_hit = False
            self._high_acted = True

        # 进入极端偏离: 置位 (已置位则保持; 本桶已触发信号则不再置位)
        if low_dev > self.low1 and not self._low_acted:
            self.low_hit = True
            self._low_acted = True
        if high_dev > self.high1 and not self._high_acted:
            self.high_hit = True
            self._high_acted = True
        return signal, info
