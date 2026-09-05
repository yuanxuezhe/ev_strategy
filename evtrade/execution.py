from __future__ import annotations
"""下单抽象: 模拟 / 真实 (自 mysql_analyze_demo.py 原样迁移)"""


# ============ 执行器 (下单抽象: 模拟 / 真实) ============

class Executor:
    """下单接口: 策略只调 trade(signal), 不关心是模拟还是真实委托"""

    def trade(self, signal: str, price: float, ts: str) -> bool:
        raise NotImplementedError


class SimulatedExecutor(Executor):
    """回测模拟下单: 以信号当根 close 成交, 资金/持仓不足则买满/卖完

    scale: 倍投系数。连续同方向信号时, 下一次数量 = 上一次 × scale (首次为
    trade_qty); 方向翻转重置为基础数量。信号即计数 (未成交也计)。1.0 = 关闭。
    与 kernel._execute 语义逐行一致 (差分测试锁定)。
    """

    def __init__(self, account, qty: float, verbose=True, scale: float = 1.0):
        self.account = account
        self.qty = qty
        self.verbose = verbose
        self.scale = scale
        self.last_side = 0
        self.cur_qty = qty

    def trade(self, signal: str, price: float, ts: str) -> bool:
        acc = self.account
        side = 1 if signal == "BUY" else -1
        if side == self.last_side:
            self.cur_qty = self.cur_qty * self.scale
        else:
            self.cur_qty = self.qty
            self.last_side = side
        if signal == "BUY":
            qty = min(self.cur_qty, acc.cash / price) if price > 0 else 0
            if qty <= 0:
                return False
            acc.apply("BUY", qty, price, ts)
            if self.verbose:
                print(f"        >> BUY  {qty:.0f}股 @ {price:.4f}  花费 {qty*price:.2f}  "
                      f"剩余资金 {acc.cash:.2f} 持仓 {acc.position:.0f}", flush=True)
            return True
        elif signal == "SELL":
            qty = min(self.cur_qty, acc.position)
            if qty <= 0:
                return False
            acc.apply("SELL", qty, price, ts)
            if self.verbose:
                print(f"        >> SELL {qty:.0f}股 @ {price:.4f}  收入 {qty*price:.2f}  "
                      f"剩余资金 {acc.cash:.2f} 持仓 {acc.position:.0f}", flush=True)
            return True
        return False


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
