from __future__ import annotations
"""核心调度层

================================================================
✅  可改层 (core 子包)  ✅
================================================================
core/ 是主调度层, 把 kernel/engine/sweep/replay 串起来。

当前实现:
  - kernel.py     流式 numba 决策内核 (冻结)
  - engine.py     参考引擎 (冻结, 给 replay 对账用)
  - sweep.py      并发参数扫描 + 鲁棒评分 (慎改, kbs/13 锁定公式)
  - replay.py     录制回放对账 (冻结, 实盘一致性验收门)
  - permutation.py MC 置换检验 (慎改, kbs/13 锁定方式)

目前这五个文件还在 evtrade/ 顶层 (与 frozen module 共存),
未来可整体迁移到本子包。当前保留是为了不破坏 56 项差分测试的
import 路径 (cli.py / tests 都从 evtrade 直接 import)。
"""


def main(argv=None):
    """统一 CLI 入口 (委托 evtrade.cli.main)"""
    from ..cli import main as _cli_main
    return _cli_main(argv)
