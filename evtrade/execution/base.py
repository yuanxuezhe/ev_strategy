from __future__ import annotations
"""下单抽象: 模拟 / 真实 (自 mysql_analyze_demo.py 原样迁移)

================================================================
✅  可改层模块  ✅
================================================================
本文件定义下单撮合逻辑, 是实盘对接的入口点:

  - SimulatedExecutor.trade (回测模拟)
      内部 buy_pct/sell_pct/all_in/scale/qty 计算与 evtrade.kernel._execute
      是**逐位等价**的 (tests/test_funding.py 锁定)。
      如果只改 SimulatedExecutor 而不改 kernel._execute, 对账会 FAIL。
      改两边 (本文件 + kernel._execute + gpu CUDA 段) 即可。

  - BrokerExecutor.trade (实盘占位)
      **这里是接券商 API 的入口**: 调 broker.place_order,
      收到成交回报后调 self.account.apply(side, qty, price, ts)。
      当前的占位实现 print [LIVE] 后返回 False, 不改账户。

要加的常见功能:
  - 手续费/滑点: 在 apply 前扣 cash
  - 部分成交: account.apply 只接收已成交的部分
  - 撤单/拒单: trade 返回 False + 不调 apply
================================================================
"""


# ============ 执行器 (下单抽象: 模拟 / 真实) ============

class Executor:
    """下单接口: 策略只调 trade(signal), 不关心是模拟还是真实委托"""

    def trade(self, signal: str, price: float, ts: str) -> bool:
        raise NotImplementedError

    def update_price(self, price: float) -> None:
        """更新账户最新价 (权益估值用; 由 Engine 在策略期每根 bar 调用)。

        封装 ``self.account.last_price = price``, 避免引擎穿透执行器内部结构。
        """
        self.account.last_price = price


class SimulatedExecutor(Executor):
    """回测模拟下单: 以信号当根 close 成交, 资金/持仓不足则买满/卖完

    scale: 倍投系数。连续同方向信号时, 下一次数量 = 上一次 × scale (首次为
    trade_qty); 方向翻转重置为基础数量。信号即计数 (未成交也计)。1.0 = 关闭。
    与 kernel._execute 语义逐行一致 (差分测试锁定)。

    资金模式 (阶段 2 新增, 与 kernel 对齐):
      buy_pct  ∈ [0,1]: BUY 时按当前 cash 的比例下注
      sell_pct ∈ [0,1]: SELL 时按当前 position 的比例卖
      all_in=True: 等价于 buy_pct=sell_pct=1.0
      三者均 0 时走旧 fixed-qty 路径 (与历史行为 100% 一致)
    """

    def __init__(self, account, qty: float, verbose=True, scale: float = 1.0,
                 buy_pct: float = 0.0, sell_pct: float = 0.0, all_in: bool = False):
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
