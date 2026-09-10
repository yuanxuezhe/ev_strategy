"""下单抽象: 模拟 / 真实

本文件定义下单撮合逻辑:

  - trade_decision (单一成交决策实现)
      取量 (buy_pct/sell_pct/fixed-qty) + 资金/持仓约束 + 现金持仓更新。
      SimulatedExecutor.trade 与 vectorized_engine._execute_trades 共用,
      保证两条引擎路径逐笔 bitwise 一致 (replay --against-ref 对账依赖)。

  - SimulatedExecutor.trade (回测模拟, 调 trade_decision)
      scale: 同方向连续信号时, 数量按 scale 倍投; 翻转时重置为基础数量。
      (scale/last_side 是调用方状态, 留在 executor 侧)

  - Account.apply (记账)

要加的常见功能:
  - 手续费/滑点: 在 apply 前扣 cash
  - 部分成交: account.apply 只接收已成交的部分
  - 撤单/拒单: trade 返回 False + 不调 apply
"""
from __future__ import annotations


def trade_decision(side: int, price: float, cash: float, position: float,
                   cur_qty: float, buy_pct: float, sell_pct: float
                   ) -> tuple[float, float, float, bool]:
    """单一成交决策实现 (SimulatedExecutor / vectorized 引擎共用)

    side: 1=BUY, -1=SELL (调用方保证非 0)
    cur_qty: 本轮基础数量 (scale 倍投后的量, 由调用方维护)
    返回 (new_cash, new_position, qty, filled); filled=False 时现金/持仓原样返回。
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


class Executor:
    """下单接口: 策略只调 trade(signal), 不关心是模拟还是真实委托"""

    def trade(self, signal: str, price: float, ts: str) -> bool:
        raise NotImplementedError


class SimulatedExecutor(Executor):
    """回测模拟下单: 以信号当根 close 成交, 资金/持仓不足则买满/卖完

    scale: 倍投系数。连续同方向信号时, 下一次数量 = 上一次 × scale (首次为
    trade_qty); 方向翻转重置为基础数量。信号即计数 (未成交也计)。1.0 = 关闭。

    资金模式:
      buy_pct  ∈ [0,1]: BUY 时按当前 cash 的比例下注
      sell_pct ∈ [0,1]: SELL 时按当前 position 的比例卖
      all_in=True: 等价于 buy_pct=sell_pct=1.0
      三者均 0 时走 fixed-qty 路径
    """

    def __init__(self, account, qty: float, verbose=True, scale: float = 1.0,
                 buy_pct: float = 0.0, sell_pct: float = 0.0, all_in: bool = False,
                 record_to: list | None = None):
        self.account = account
        self.qty = qty
        self.verbose = verbose
        self.scale = scale
        self.buy_pct = buy_pct
        self.sell_pct = sell_pct
        self.all_in = all_in
        if all_in:
            self.buy_pct = max(self.buy_pct, 1.0)
            self.sell_pct = max(self.sell_pct, 1.0)
        self.last_side = 0
        self.cur_qty = qty
        # record_to: 若非 None, 每次成功 trade 追加 {ts,side,qty,price} 副本 (replay/对账用)
        self.record_to = record_to

    def trade(self, signal: str, price: float, ts: str) -> bool:
        acc = self.account
        side = 1 if signal == "BUY" else -1
        if side == self.last_side:
            self.cur_qty = self.cur_qty * self.scale
        else:
            self.cur_qty = self.qty
            self.last_side = side
        cash, position, qty, filled = trade_decision(
            side, price, acc.cash, acc.position, self.cur_qty,
            self.buy_pct, self.sell_pct)
        if not filled:
            return False
        side_str = signal
        acc.apply(side_str, qty, price, ts)
        self._record(side=side_str, qty=qty, price=price, ts=ts)
        if self.verbose:
            verb = "BUY " if side > 0 else "SELL"
            cash_word = "花费" if side > 0 else "收入"
            print(f"        >> {verb}{qty:.0f}股 @ {price:.4f}  {cash_word} {qty*price:.2f}  "
                  f"剩余资金 {cash:.2f} 持仓 {position:.0f}", flush=True)
        return True

    def _record(self, side: str, qty: float, price: float, ts: str):
        """成功 trade 后: 同步追加副本到 record_to (供对账/回放对比成交)"""
        if self.record_to is None:
            return
        self.record_to.append({"ts": int(ts), "side": side,
                               "qty": float(qty), "price": float(price)})