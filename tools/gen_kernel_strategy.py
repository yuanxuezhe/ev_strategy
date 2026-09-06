"""tools/gen_kernel_strategy.py

校验 kernel.py::_strategy_check 中的 DSL 策略段与
strategies/channel_deviation.py::_CHANNEL_DEVIATION_DSL 的一致性。

背景:
  kernel.py:_strategy_check 函数体被 DSL-STRATEGY-BEGIN/END 标记包围,
  本应是 channel_deviation DSL 渲染后的 st 形式; 但生产路径下,
  kernel_dsl.dsl_kernel('channel_deviation') 特判返回冻结 kernel 本尊,
  build_dsl_kernel 的渲染产物只在 tests/test_kernel_dsl.py 用作 diff 比对。
  也就是说: 同一策略的逻辑存在于三份独立源码 —— DSL docstring、
  冻结 kernel 的手写 st 代码、build_dsl_kernel 渲染产物 —— 任何一份
  修改都可能让差分测试或真实回测结果悄悄漂移。

用法:
  python tools/gen_kernel_strategy.py --check      # 仅校验, exit 0=一致 / 1=漂移

注意:
  本脚本默认 --check-only。即使检测到漂移, 也不应自动 --write 替换:
  历史上 kernel.py 的手写代码与 DSL 渲染产物在 numba 语义上等价但
  文本结构不同 (例如 DSL 把 elif 拆成 else+if, 括号更密), 直接覆盖
  会破坏 72 项差分测试的逐位锁定基线; 真要升级为 '渲染产物写回' 的
  工作流, 需要先建立新基线, 跑通全量差分测试再切。

CI 接入:
  - 把 'python tools/gen_kernel_strategy.py --check' 加到 CI step,
    失败即阻止合并;
  - 漂移时报警, 开发者手工 review + 选择: (a) 改 DSL 让两份对齐, 或
    (b) 改 kernel.py 手写段对齐 DSL。
"""
from __future__ import annotations

import argparse
import difflib
import sys
import textwrap
from pathlib import Path

# 允许从仓库根目录运行
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evtrade.strategies.dsl import (  # noqa: E402
    render_numba_state_body,
)
from evtrade.strategies import get_strategy_class  # noqa: E402

KERNEL_PATH = REPO_ROOT / "evtrade" / "core" / "kernel.py"
SPLICE_BEGIN = "# ==== DSL-STRATEGY-BEGIN"
SPLICE_END = "# ==== DSL-STRATEGY-END"


def _read_kernel_region() -> str:
    """读 kernel.py 中 BEGIN/END 之间的策略段 (含两端标记)"""
    src = KERNEL_PATH.read_text(encoding="utf-8")
    i0 = src.index(SPLICE_BEGIN)
    i1 = src.index(SPLICE_END)
    i0 = src.rindex("\n", 0, i0) + 1
    i1 = src.index("\n", i1) + 1
    return src[i0:i1]


def _render_channel_deviation_body() -> str:
    """DSL 渲染 -> 与 kernel.py 同形的 st 代码 (无 BEGIN/END 标记)"""
    cls = get_strategy_class("channel_deviation")
    body = render_numba_state_body(cls).strip("\n")
    return textwrap.indent(body, "    ") + "\n"


def diff_kernel_vs_dsl() -> tuple[str, str]:
    """返回 (current_region, expected_region) 两个字符串"""
    current = _read_kernel_region()
    expected_inner = _render_channel_deviation_body()
    expected = (
        f"    # ==== DSL-STRATEGY-BEGIN (kernel_dsl.py 按此标记整段替换) ====\n"
        f"{expected_inner}"
        f"    # ==== DSL-STRATEGY-END ====\n"
    )
    return current, expected


def format_diff(current: str, expected: str) -> str:
    return "".join(difflib.unified_diff(
        current.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile="kernel.py (current)",
        tofile="channel_deviation DSL (expected)"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", default=True,
                    help="仅校验, 不写回 (默认行为)")
    args = ap.parse_args(argv)

    current, expected = diff_kernel_vs_dsl()

    if current == expected:
        print("[OK] kernel.py 策略段与 channel_deviation DSL 一致")
        return 0

    print("[DRIFT] kernel.py 策略段与 channel_deviation DSL 不一致:")
    sys.stdout.write(format_diff(current, expected))
    print("\n修法: 手工对齐两份源码 ——")
    print("  - 改 DSL docstring 让渲染产物等于 kernel.py 现状, 或")
    print("  - 改 kernel.py 标记区间让其与 DSL 渲染产物一致。")
    print("不要自动写回: 历史上两份代码逐位锁定, 自动覆盖会破坏 72 项差分测试基线。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
