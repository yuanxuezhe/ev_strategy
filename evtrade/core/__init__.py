from __future__ import annotations
"""core 子包: 主调度层

公开模块 (DSL/numba 内核已下线):
  aggregator         BarAggregator 增量桶合并 (差分锁定)
  timeutils          compute_bucket_general (任意周期, 差分锁定)
  engine             Engine 逐 bar 路径 (实盘/对账)
  vectorized_engine  CuPy 统一 CPU/GPU 批量引擎
  gpu                CuPy 探测 / GPU info / precompute_ts_mark
  sweep              并发参数扫描 + 鲁棒评分
  replay             录制回放对账 (vectorized vs Engine)
  permutation        MC 置换检验
  metrics            bars_to_arrays + 16 字段 summarize
  capability         gpu_available / select_device
  data               MySQL 拉数 + npz 缓存 + 合成数据
  config             全局常量
"""
