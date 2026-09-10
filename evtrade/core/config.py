from __future__ import annotations
"""常量与默认配置 (DB_URL 支持环境变量覆盖)

可改默认值 (改动安全):
  - INIT_CASH / INIT_POSITION / TRADE_QTY: 默认资金/持仓/单笔数量
  - TF1: 通道轨 EMA 默认周期 (CLI --tf1 可覆盖)
  - DB_URL: 环境变量 EVTRADE_DB_URL 可覆盖, 避免凭据入仓
  - PERIODS: 7 个预定义周期; --period 支持任意 m/h/d, PERIODS 仅做默认

注意: 修改默认值会改变所有未显式传参的回测结果, 需在 kbs/10 同步。
"""

import os
from datetime import timedelta

# 数据库连接串: 可用环境变量 EVTRADE_DB_URL 覆盖, 避免凭据硬编码进版本库
DB_URL = os.environ.get(
    "EVTRADE_DB_URL",
    "mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4",
)
TABLE = "minute_bars"
INTERVAL = 0.1  # 100ms

# 周期: (单位, 数值, stime 起始位 0-indexed, timedelta)
# stime = YYYYMMDDHHmmss: 0-3年 4-5月 6-7日 8-9时 10-11分 12-13秒
PERIODS = {
    "1m":  ("m", 1,  10, timedelta(minutes=1)),
    "5m":  ("m", 5,  10, timedelta(minutes=5)),
    "15m": ("m", 15, 10, timedelta(minutes=15)),
    "30m": ("m", 30, 10, timedelta(minutes=30)),
    "1h":  ("h", 1,  8,  timedelta(hours=1)),
    "4h":  ("h", 4,  8,  timedelta(hours=4)),
    "1d":  ("d", 1,  6,  timedelta(days=1)),
}

TF1 = 21  # 通道轨 EMA 周期

INIT_CASH = 200000.0       # 期初资金 20万
INIT_POSITION = 200000.0   # 期初持仓 20万股
TRADE_QTY = 10000.0        # 每次信号交易股数

