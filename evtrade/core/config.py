from __future__ import annotations
"""常量与默认配置 (DB_URL 支持环境变量覆盖)

可改默认值 (改动安全):
  - INIT_CASH / INIT_POSITION / TRADE_QTY: 默认资金/持仓/单笔数量
  - DB_URL: 环境变量 EVTRADE_DB_URL 可覆盖, 避免凭据入仓

注意: 修改默认值会改变所有未显式传参的回测结果, 需在 kbs/10 同步。
"""

import os

# 数据库连接串: 可用环境变量 EVTRADE_DB_URL 覆盖, 避免凭据硬编码进版本库
DB_URL = os.environ.get(
    "EVTRADE_DB_URL",
    "mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4",
)
TABLE = "minute_bars"

INIT_CASH = 200000.0       # 期初资金 20万
INIT_POSITION = 200000.0   # 期初持仓 20万股
TRADE_QTY = 10000.0        # 每次信号交易股数

