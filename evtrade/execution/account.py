"""资金/持仓记账

Account.apply() 的 BUY/SELL 记账逻辑是模拟/实盘的下单基础。

如果需要改手续费/滑点模型, 在外层 (Executor) 扣费后再调 apply。
"""


class Account:
    """资金与持仓管理 (回测/实盘共用)"""

    def __init__(self, cash: float, position: float):
        self.init_cash = cash
        self.init_position = position
        self.cash = cash
        self.position = position
        self.trades: list[dict] = []

    def apply(self, side: str, qty: float, price: float, ts: str):
        """记账 (不含下单逻辑, 由 Executor 调用)"""
        if side == "BUY":
            self.cash -= qty * price
            self.position += qty
        elif side == "SELL":
            self.cash += qty * price
            self.position -= qty
        self.trades.append({"ts": ts, "side": side, "qty": qty, "price": price})