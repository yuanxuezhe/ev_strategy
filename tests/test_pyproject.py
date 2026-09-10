"""pyproject.toml / uv 集成

覆盖:
  - pyproject.toml 存在, [project] 字段必填 (name/version/requires-python)
  - entry point evtrade -> evtrade.cli:main 已注册
  - dev-dependencies 走 [dependency-groups] dev (PEP 735, 替代已废弃的
    tool.uv.dev-dependencies)
  - 核心依赖: numpy/pandas/sqlalchemy/pymysql/torch (2026-09-09 已删 numba)
  - 无 GPU extra: torch 为核心依赖, CPU/GPU 同一 torch 包 (2026-09-10 删 gpu/all extra)
  - 排除目录 (tests/docs/...) 不被打包
  - 若环境有 uv: uv lock --check 通过
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject() -> str:
    return (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


# ---------- 基础字段 ----------


def test_pyproject_exists():
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_name_and_version():
    text = _read_pyproject()
    assert 'name = "evtrade"' in text
    assert 'version = "0.1.0"' in text


def test_requires_python_at_least_38():
    text = _read_pyproject()
    assert 'requires-python = ">=3.8"' in text


def test_entry_point_evtrade_registered():
    text = _read_pyproject()
    # [project.scripts] evtrade = "evtrade.cli:main"
    assert "[project.scripts]" in text
    assert 'evtrade = "evtrade.cli:main"' in text


# ---------- 依赖 ----------


def test_core_dependencies_listed():
    text = _read_pyproject()
    for pkg in ("numpy", "pandas", "sqlalchemy", "pymysql", "torch"):
        assert f'"{pkg}>=' in text, f"缺少核心依赖: {pkg}"


def test_numba_dependency_removed():
    """2026-09-09 统一 CPU/GPU 重构后, numba 依赖已下线 (DSL/numba 内核已删)"""
    text = _read_pyproject()
    # 不再要求 numba (CPU/GPU 走 torch 算子)
    assert '"numba>=' not in text, "numba 已下线, 不应再列为核心依赖"


def test_no_optional_gpu_extra():
    """2026-09-10 simplify-user-surface: 无独立 GPU 安装路径

    torch 在核心依赖; CPU/GPU 是同一 torch 包的不同运行时, 不是两个 extra
    (`uv sync --extra gpu` 与 `uv sync` 无差别 → extra 空转, 已删)。
    """
    text = _read_pyproject()
    assert "[project.optional-dependencies]" not in text, (
        "不应再有 [project.optional-dependencies] (gpu/all extra 已删, torch 为核心依赖)")


def test_cupy_dependency_removed():
    """2026-09-10 pytorch-unified-strategy: cupy 已下线"""
    text = _read_pyproject()
    assert "cupy" not in text, "cupy 已下线, 由 torch 统一接管 CPU/GPU"


# ---------- dev 依赖用 [dependency-groups] (PEP 735, uv 推荐) ----------


def test_dev_dependencies_use_dependency_groups():
    """uv 0.4.7+ 已弃用 tool.uv.dev-dependencies, 改用 [dependency-groups] dev"""
    text = _read_pyproject()
    assert "[dependency-groups]" in text
    assert 'dev = [' in text
    assert "tool.uv" in text  # managed = true 仍保留
    # 旧字段作为活跃配置不应再出现 (注释里提到是允许的, 这里只验结构性)
    import re
    # 抓取 [tool.uv] 段, 不应再有 dev-dependencies key
    m = re.search(r"\[tool\.uv\](.*?)(?=\n\[|\Z)", text, re.S)
    if m:
        # [tool.uv] 段不应再有 dev 依赖字段 (旧字段 dev_dependencies 已废)
        assert "dev_dependencies" not in m.group(1), (
            "[tool.uv] 段不应再有旧的 dev_dependencies 字段")
        assert "dev-dependencies" not in m.group(1), (
            "[tool.uv] 段不应再有旧的 dev-dependencies 字段")


def test_pytest_listed_in_dev():
    text = _read_pyproject()
    assert "pytest" in text


# ---------- 包发现 ----------


def test_setuptools_packages_find_excludes_test_dirs():
    text = _read_pyproject()
    assert "[tool.setuptools.packages.find]" in text
    assert 'exclude = ["tests*"' in text or "exclude = ['tests*'" in text
    # 排除 docs/kbs/scripts/tools/cache (非包内容)
    for excl in ("tests*", "docs*", "kbs*", "scripts*", "tools*", "cache*"):
        assert excl in text


def test_package_data_for_defaults():
    """evtrade.strategies._defaults/*.json 随包安装"""
    text = _read_pyproject()
    assert "[tool.setuptools.package-data]" in text
    assert "_defaults" in text
    assert "*.json" in text


# ---------- pytest 配置 ----------


def test_pytest_config_present():
    text = _read_pyproject()
    assert "[tool.pytest.ini_options]" in text
    assert 'testpaths = ["tests"]' in text
    # gpu marker 已注册
    assert "gpu:" in text


# ---------- uv lock 一致性 (可选, 仅当 uv 可用) ----------


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv 不可用")
def test_uv_lock_check_passes():
    """本机 uv 锁文件 (若存在) 与 pyproject 一致"""
    r = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    # 0=通过 (有 lockfile 且一致); 2=无 lockfile (本仓库不强制)
    if r.returncode == 2 and "Unable to find lockfile" in (r.stderr + r.stdout):
        pytest.skip("uv.lock 不存在 (默认不入库), 跳过")
    assert r.returncode == 0, f"uv lock --check 失败: {r.stderr}"
