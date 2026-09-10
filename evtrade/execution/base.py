"""下单抽象: 模拟 / 真实

本文件定义下单撮合逻辑, 是实盘对接的入口点:

  - SimulatedExecutor.trade (回测模拟)
      scale: 同方向连续信号时, 数量按 scale 倍投; 翻转时重置为基础数量。
      buy_pct/sell_pct/all_in: 按现金/持仓比例下注 (三者均 0 时走 fixed-qty)

  - BrokerExecutor.trade (实盘占位)
      这里是接券商 API 的入口: 调 broker.place_order,
      收到成交回报后调 self.account.apply(side, qty, price, ts)。
      当前的占位实现 print [LIVE] 后返回 False, 不改账户。

要加的常见功能:
  - 手续费/滑点: 在 apply 前扣 cash
  - 部分成交: account.apply 只接收已成交的部分
  - 撤单/拒单: trade 返回 False + 不调 apply
"""
from __future__ import annotations

from .account import Account


class Executor:
    """下单接口: 策略只调 trade(signal), 不关心是模拟还是真实委托"""

    def trade(self, signal: str, price: float, ts: str) -> bool:
        raise NotImplementedError

    def update_price(self, price: float) -> None:
        """更新账户最新价 (权益估值用; 由 Engine 在策略期每根 bar 调用)。

        封装 self.account.last_price = price, 避免引擎穿透执行器内部结构。
        """
        self.account.last_price = price


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
        if signal == "BUY":
            if price > 0:
                max_by_cash = acc.cash / price
                if self.buy_pct > 0:
                    target = self.buy_pct * max_by_cash
                    qty = target if target < max_by_cash else max_by_cash
                else:
                    qty = self.cur_qty if self.cur_qty < max_by_cash else max_by_cash
            else:
                qty = 0
            if qty <= 0:
                return False
            acc.apply("BUY", qty, price, ts)
            self._record(side="BUY", qty=qty, price=price, ts=ts)
            if self.verbose:
                print(f"        >> BUY  {qty:.0f}股 @ {price:.4f}  花费 {qty*price:.2f}  "
                      f"剩余资金 {acc.cash:.2f} 持仓 {acc.position:.0f}", flush=True)
            return True
        elif signal == "SELL":
            if self.sell_pct > 0:
                target = self.sell_pct * acc.position
                qty = target if target < acc.position else acc.position
            else:
                qty = self.cur_qty if self.cur_qty < acc.position else acc.position
            if qty <= 0:
                return False
            acc.apply("SELL", qty, price, ts)
            self._record(side="SELL", qty=qty, price=price, ts=ts)
            if self.verbose:
                print(f"        >> SELL {qty:.0f}股 @ {price:.4f}  收入 {qty*price:.2f}  "
                      f"剩余资金 {acc.cash:.2f} 持仓 {acc.position:.0f}", flush=True)
            return True
        return False

    def _record(self, side: str, qty: float, price: float, ts: str):
        """成功 trade 后: 同步追加副本到 record_to (供对账/回放对比成交)"""
        if self.record_to is None:
            return
        self.record_to.append({"ts": int(ts), "side": side,
                               "qty": float(qty), "price": float(price)})


class BrokerExecutor(Executor):
    """实盘下单: 调券商 API (占位, 接入时实现)"""

    def __init__(self, account: Account, qty: float, broker=None):
        self.account = account
        self.qty = qty
        self.broker = broker  # 券商 API 客户端

    def trade(self, signal: str, price: float, ts: str) -> bool:
        # TODO: 接入真实券商下单 API
        # order_id = self.broker.place_order(signal, self.qty, ...)
        # 成交回报后调 self.account.apply(...) 记账
        print(f"        >> [LIVE] {signal} {self.qty:.0f}股 @ {price:.4f} (未接入券商)",
              flush=True)
        return False