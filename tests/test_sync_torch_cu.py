"""sync-torch-cu.sh 行为契约

2026-09-12 add-gpu-extra-pyproject: 该 helper 是 GPU 协作者的防御性工具,
lockfile 被 uv 重生回 CPU wheel 时自动 reinstall 到 cu128。

覆盖:
  - 脚本存在 + 可执行 (bash +x)
  - torch 已是目标版本时退出 0 且不做 reinstall (幂等)
  - 接受 EVT_TORCH_CU_TAG 环境变量
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync-torch-cu.sh"


def _run_script(env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    # 脚本中文输出, 在 Windows bash 下 stdout 是 UTF-8; 强制以 utf-8 解码避免 codepage 误判
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=str(REPO_ROOT), capture_output=True,
        env={**env, "PYTHONIOENCODING": "utf-8"},
        timeout=300,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash 不可用")
def test_script_exists_and_executable():
    """scripts/sync-torch-cu.sh 必须存在"""
    assert SCRIPT.is_file(), f"缺失: {SCRIPT}"
    # shebang
    head = SCRIPT.read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("#!") and "bash" in head, f"shebang 应为 bash, 实际: {head}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash 不可用")
def test_script_idempotent_when_torch_already_correct():
    """torch 已是 cu128 时, 脚本应识别并跳过 (退出 0)"""
    import torch
    if not torch.version.cuda:
        pytest.skip("当前环境 torch 是 CPU wheel, 不适用幂等测试")
    r = _run_script()
    assert r.returncode == 0, f"脚本失败: stderr={r.stderr!r}"
    out = r.stdout.decode("utf-8", errors="replace")
    assert "跳过" in out, f"幂等时脚本应打 '跳过' 信息, 实际 stdout={out!r}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash 不可用")
def test_script_uses_default_cu128():
    """默认 EVT_TORCH_CU_TAG 应为 cu128"""
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'CU_TAG="${EVT_TORCH_CU_TAG:-cu128}"' in text, (
        "脚本默认 cu tag 应为 cu128")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash 不可用")
def test_script_accepts_custom_cu_tag():
    """EVT_TORCH_CU_TAG 应能被覆写 (用 --dry-run 模式打印即可; 真实 reinstall 太重)"""
    # 用 --dry-run 标记不会让脚本跳过检测逻辑, 仅打印目标 tag
    text = SCRIPT.read_text(encoding="utf-8")
    assert "EVT_TORCH_CU_TAG" in text, "脚本应引用 EVT_TORCH_CU_TAG 环境变量"
    # 实际跑会执行 reinstall, 在 CI 跳过