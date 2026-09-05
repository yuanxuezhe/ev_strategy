from __future__ import annotations
"""minute_bars 周期合并 + 通道偏离策略 (回测/实盘统一架构)

原单文件实现已拆分为 evtrade 包 (本文件保留原命令行入口, 行为不变):

分层:
  Feed (行情源)     ──yield Bar──▶ Engine
                                   │
                              BarAggregator (周期合并, 独立于策略)
                                   │ on_bars(周期K线集合)
                                   ▼
                         ChannelDeviationStrategy + ema_channel(指标)
                                   │ signal
                                   ▼
                              Executor (模拟/真实下单) → Account (资金持仓)

切换实盘/回测: 换 Feed + 换 Executor, 其余不变。
默认走 numba 流式内核 (engine=kernel, 与原实现逐笔等价, 快约千倍);
--engine ref 可回到原 Python 引擎。
用法: python mysql_analyze_demo.py --period 5m --start 20250101 --end 20260903 --no-sleep
依赖: pip install pymysql sqlalchemy numpy numba
"""

from evtrade.cli import main

if __name__ == "__main__":
    main()
