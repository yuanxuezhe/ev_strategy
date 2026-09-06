"""tools/gen_kernel_strategy.py 行为

覆盖:
  - 工具可被 import 并跑 main(), 返回 int exit code
  - diff_kernel_vs_dsl() 返回 (current, expected) 两个非空字符串
  - format_diff() 在两份不同时产出 unified diff 文本
  - 当前快照: 工具报 DRIFT (历史上两份代码逐位锁定但结构不同;
    自动覆盖会破坏 72 项差分测试, 因此工具仅 check 不写回)
  - 模拟 '强制一致' 时: 把 kernel.py 的 BEGIN/END 区间临时替换为 DSL 渲染
    产物, 工具应当返回 0
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# 把 tools 加进 sys.path
TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import gen_kernel_strategy as gks  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO_ROOT / "evtrade" / "core" / "kernel.py"


# ---------- 基础行为 ----------


def test_diff_kernel_vs_dsl_returns_two_strings():
    current, expected = gks.diff_kernel_vs_dsl()
    assert isinstance(current, str) and current
    assert isinstance(expected, str) and expected


def test_format_diff_on_drift_produces_unified_diff():
    current, expected = gks.diff_kernel_vs_dsl()
    # 当前快照是 DRIFT (渲染产物与手写代码结构不同)
    diff = gks.format_diff(current, expected)
    assert "---" in diff and "+++" in diff


def test_main_returns_int_exit_code(monkeypatch):
    """main() 不带 argv 时从 sys.argv 解析; 在 pytest 上下文里 sys.argv 是
    pytest 参数, 会让 argparse 报 'unrecognized arguments' 退出码 2。
    本测试显式传 argv=['--check'], 隔离 pytest 干扰。"""
    rc = gks.main(argv=["--check"])
    assert isinstance(rc, int)
    # 当前快照是 DRIFT, 应当返回 1
    assert rc == 1


# ---------- 模拟 '一致' 状态: 临时替换 kernel.py 区间后 main() 返回 0 ----------


def test_main_returns_zero_when_kernel_matches(monkeypatch):
    """模拟强制一致: 临时把 kernel.py 的 BEGIN/END 区间替换为 DSL 渲染产物,
    main() 必须返回 0。"""
    # 1. 备份原文件
    original = KERNEL_PATH.read_text(encoding="utf-8")
    # 2. 计算期望区间
    _, expected = gks.diff_kernel_vs_dsl()
    # 3. 替换区间
    src = original
    i0 = src.index(gks.SPLICE_BEGIN)
    i0 = src.rindex("\n", 0, i0) + 1
    i1 = src.index(gks.SPLICE_END)
    i1 = src.index("\n", i1) + 1
    new_src = src[:i0] + expected + src[i1:]
    KERNEL_PATH.write_text(new_src, encoding="utf-8")
    try:
        rc = gks.main(argv=["--check"])
        assert rc == 0, "替换后应当判定一致"
    finally:
        KERNEL_PATH.write_text(original, encoding="utf-8")
        # 双保险: 再跑一次应当回到 DRIFT=1
        assert gks.main(argv=["--check"]) == 1


# ---------- CLI 子进程调用 (跨进程 sanity check) ----------


def test_cli_invocation_exit_code():
    """用 subprocess 跑 'python tools/gen_kernel_strategy.py --check',
    当前快照应当返回 1 (DRIFT)。"""
    proc = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "gen_kernel_strategy.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 1
    assert "DRIFT" in proc.stdout
