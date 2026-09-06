"""死代码清理与 .gitignore 覆盖

覆盖:
  - evtrade/feeds/registry.py 已删 (与 _registry.py 重复且无引用)
  - evtrade/core/runner.py 已删 (仅 core/__init__.py 跳板用, 已重写 __init__)
  - .gitignore 覆盖 root 散落的 results*.csv 与 champion_bars.csv
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_feeds_registry_removed():
    p = REPO_ROOT / "evtrade" / "feeds" / "registry.py"
    assert not p.exists(), (
        f"feeds/registry.py 仍存在, 应当删除 (与 _registry.py 重复且无引用): {p}")


def test_core_runner_removed():
    p = REPO_ROOT / "evtrade" / "core" / "runner.py"
    assert not p.exists(), (
        f"core/runner.py 仍存在, 应当删除 (仅 core/__init__.py 跳板用): {p}")


def test_gitignore_covers_root_csv_outputs():
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    # root 散落的 sweep 输出与 champion bars
    assert "results*.csv" in gi or "results_*.csv" in gi, (
        ".gitignore 应覆盖根目录 results*.csv (sweep 输出)")
    assert "champion_bars.csv" in gi, (
        ".gitignore 应覆盖根目录 champion_bars.csv")


def test_core_init_no_longer_imports_runner():
    """core/__init__.py 不再 re-export `main` (跳板已删)"""
    import evtrade.core as core_mod
    # 不应当再有 __all__ = ['main']
    all_attr = getattr(core_mod, "__all__", None)
    if all_attr is not None:
        assert "main" not in all_attr
