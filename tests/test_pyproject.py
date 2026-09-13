"""pyproject.toml / uv 集成

覆盖:
  - pyproject.toml 存在, [project] 字段必填 (name/version/requires-python)
  - entry point evtrade -> evtrade.cli:main 已注册
  - dev-dependencies 走 [dependency-groups] dev (PEP 735, 替代已废弃的
    tool.uv.dev-dependencies)
  - 核心依赖: numpy/pandas/sqlalchemy/pymysql/torch (2026-09-09 已删 numba)
  - GPU 走 scripts/sync-torch-cu.sh (非 pyproject extra):
    pyproject MUST NOT 声明 gpu optional extra, MUST NOT 为 torch 配 cu128
    source/index —— 否则 `uv lock` 会把默认 `uv sync` 的 torch 也塌缩成 cu128
    (2026-09-13 drop-gpu-extra-cpu-default: 替代 add-gpu-extra-pyproject)
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


def _active_pyproject() -> str:
    """只保留非注释行的 pyproject 文本。

    pyproject 里的注释会提及 `cu128` / `torch==2.9.0+cu128` 以说明"为何不再
    用 extra 装 GPU", 故校验"活跃配置"时须剔除纯注释行, 否则会误伤文档注释。
    (TOML 本文件无行内 # 注释, 逐行去 `#` 前缀行即可。)
    """
    lines = _read_pyproject().splitlines()
    return "\n".join(l for l in lines if l.strip() and not l.strip().startswith("#"))


# ---------- 基础字段 ----------


def test_pyproject_exists():
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_name_and_version():
    text = _read_pyproject()
    assert 'name = "evtrade"' in text
    assert 'version = "0.1.0"' in text


def test_requires_python_at_least_310():
    """2026-09-12 add-gpu-extra-pyproject: torch==2.9.0+cu128 仅支持 py>=3.10"""
    text = _read_pyproject()
    assert 'requires-python = ">=3.10"' in text


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


def test_no_gpu_optional_extra():
    """2026-09-13 drop-gpu-extra-cpu-default: pyproject MUST NOT 声明 gpu extra。

    原因: `uv lock` 会把 base(`torch>=2.0`)与所有 extras 一起锁进同一份 lock;
    若声明 `gpu = ["torch==2.9.0+cu128"]`, torch 会被统一塌缩成 cu128, 导致
    默认 `uv sync` (CPU 机器) 也被迫下载 GPU wheel。GPU 改走
    scripts/sync-torch-cu.sh (与 lockfile 无关)。
    """
    active = _active_pyproject()
    # 整个 [project.optional-dependencies] 段要么不存在, 要么不含 gpu extra
    import re
    m = re.search(r"\[project\.optional-dependencies\](.*?)(?=\n\[|\Z)", active, re.S)
    assert m is None or "gpu" not in m.group(1), (
        "gpu optional extra 已下线 (会污染共享 lock 的 torch 解析); GPU 走 sync-torch-cu.sh")
    assert "torch==2.9.0+cu128" not in active, (
        "torch==2.9.0+cu128 不应再作为活跃配置出现 (GPU 由 sync-torch-cu.sh 安装)")


def test_no_cupynum_gpu_extras():
    """spec: 不得出现 cupy/numba 依赖, 也不得出现其它 GPU extra (cudnn/rocm/xpu)"""
    active = _active_pyproject()
    assert "cupy" not in active, "cupy 已下线, 由 torch 统一接管 CPU/GPU"
    assert '"numba>=' not in active, "numba 已下线, 不应再列为核心依赖"
    import re
    m = re.search(r"\[project\.optional-dependencies\](.*?)(?=\n\[|\Z)", active, re.S)
    if m:
        extras = re.findall(r"^(\w+)\s*=\s*\[", m.group(1), re.M)
        assert not (set(extras) & {"cudnn", "rocm", "xpu", "gpu"}), (
            f"不应存在 GPU 安装 extra, 发现: {extras}")


def test_no_torch_cu128_source():
    """spec: torch MUST 从 pypi 默认解析 (CPU wheel); 不得配 cu128 source/index。

    若为 torch 配 `[tool.uv.sources]` 指向 cu128 索引, `uv lock` 会把默认 torch
    也解析到 cu128 (即便加 explicit 索引也无法阻止 base 与 extra 的统一)。
    cu128 由 scripts/sync-torch-cu.sh 在 `uv sync` 后单独安装。
    """
    active = _active_pyproject()
    assert "cu128" not in active, (
        "pyproject 活跃配置不应引用 cu128 索引 (GPU wheel 由 sync-torch-cu.sh 安装)")
    # torch 仍是核心依赖, 走 pypi 默认
    assert '"torch>=' in active, "torch>=2.0 必须保留为核心依赖 (pypi CPU wheel)"


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
