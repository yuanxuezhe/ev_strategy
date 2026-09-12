"""core 子包: 主调度层 (2026-09-13 重构)

公开模块:
  aggregator         BarAggregator 增量桶合并
  timeutils          compute_bucket_general (任意周期) + 历法 (标量/向量)
  engine             Engine 逐 bar 路径 (framework 仅驱动 step)
  vectorized_engine  向量化批量引擎 (numpy 桶聚合 + step 循环; framework 仅驱动 step)
  tsbucket           桶级 ts/mark 预计算 (纯 numpy, LRU 缓存)
  sweep              并发参数扫描 + WFO + 鲁棒评分 (按策略 final_state.primary_score)
  batched_sweep      opt-in GPU-batched sweep (batched 仅产 sig)
  data               MySQL 拉数 + 合成数据 + DB_URL/TABLE 常量

2026-09-13 删除: replay / permutation / metrics / config
framework 不再有资金/持仓/撮合/PnL/收益概念; 策略 step 自管。

设备解析 (resolve_device / gpu_available) 在 evtrade.backends。
"""