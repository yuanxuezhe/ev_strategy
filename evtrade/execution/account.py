"""资金/持仓记账

Account.apply() 的 BUY/SELL 记账逻辑是模拟/实盘的下单基础。

如果需要改手续费/滑点模型, 在外层 (Executor/BrokerExecutor) 扣费后再调 apply。
"""


class Account:
    """资金与持仓管理 (回测/实盘共用)"""

    def __init__(self, cash: float, position: float):
        self.init_cash = cash
        self.init_position = position
        self.cash = cash
        self.position = position
        self.trades: list[dict] = []
        self.last_price = 0.0

    def apply(self, side: str, qty: float, price: float, ts: str):
        """记账 (不含下单逻辑, 由 Executor 调用)"""
        if side == "BUY":
            self.cash -= qty * price
            self.position += qty
        elif side == "SELL":
            self.cash += qty * price
            self.position -= qty
        self.trades.append({"ts": ts, "side": side, "qty": qty, "price": price})

    def equity(self, price: float = None) -> float:
        """总资产 = 资金 + 持仓市值"""
        p = price if price is not None else self.last_price
        return self.cash + self.position * p

    def baseline_equity(self, price: float = None) -> float:
        """不操作基线 = 期初资金 + 期初持仓 * 期末价"""
        p = price if price is not None else self.last_price
        return self.init_cash + self.init_position * p