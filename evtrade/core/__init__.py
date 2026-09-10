"""core 子包: 主调度层

公开模块:
  aggregator         BarAggregator 增量桶合并
  timeutils          compute_bucket_general (任意周期) + 历法 (标量/向量)
  engine             Engine 逐 bar 路径 (实盘/对账)
  vectorized_engine  向量化批量引擎 (numpy 桶聚合 + step 循环)
  tsbucket           桶级 ts/mark 预计算 (纯 numpy, LRU 缓存)
  sweep              并发参数扫描 + 鲁棒评分
  replay             录制回放对账 (vectorized vs Engine)
  permutation        MC 置换检验
  metrics            bars_to_arrays + 30 字段 summarize
  data               MySQL 拉数 + npz 缓存 + 合成数据
  config             全局常量

设备解析 (resolve_device / gpu_available) 在 evtrade.backends。
"""

