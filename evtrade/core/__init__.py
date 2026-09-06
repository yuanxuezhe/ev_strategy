from __future__ import annotations
"""core 子包: 主调度层

公开模块:
  kernel / kernel_dsl / engine / sweep / replay / permutation / gpu / capability / data

历史: 早期 __init__ re-export 一个 `main` (指向 .runner.main, 实际为
..cli.main 跳板), 2026-09 重构后删掉 .runner, 直接用 `evtrade.cli.main`。
"""
