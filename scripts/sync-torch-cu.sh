#!/usr/bin/env bash
# sync-torch-cu.sh — 把 torch 重写到 cu128 wheel (GPU 协作者一次性)
#
# 适用场景:
#   - lockfile 重生后, uv 把 torch 重装回 CPU wheel (cuda=None)
#   - 新 GPU 机器首次 clone 后想确保 cu128 已装
#   - 想切换 cu tag (cu126 / cu128) 测试其它 CUDA 版本
#
# 用法:
#   bash scripts/sync-torch-cu.sh                    # 默认 cu128
#   EVT_TORCH_CU_TAG=cu126 bash scripts/sync-torch-cu.sh
#
# 环境:
#   - 需要 bash (Windows: Git Bash; CLAUDE.md 已确认)
#   - 需要 uv 在 PATH
#
# 副作用:
#   - 改 .venv 内 torch wheel; 不改 pyproject.toml / uv.lock
#   - 若要持久化到 lockfile, 跑 uv lock --upgrade-package torch==2.9.0+cu128
#     --index-strategy unsafe-best-match  并 commit uv.lock

set -euo pipefail

CU_TAG="${EVT_TORCH_CU_TAG:-cu128}"
TORCH_VERSION="2.9.0+${CU_TAG}"
INDEX_URL="https://download.pytorch.org/whl/${CU_TAG}"

# 定位项目根 (脚本在 scripts/ 下)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[sync-torch-cu] project root: ${PROJECT_ROOT}"
echo "[sync-torch-cu] target: torch==${TORCH_VERSION} from ${INDEX_URL}"

# 选 python 解释器 (优先 .venv/Scripts/python.exe for Windows, 否则 venv 内的 python)
if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "[sync-torch-cu] ERROR: 找不到 python 解释器 (需要 .venv 或系统 python)" >&2
  exit 1
fi

# 探测当前 torch 状态
CURRENT_VERSION=""
CURRENT_CUDA=""
if "${PY}" -c "import torch" 2>/dev/null; then
  CURRENT_VERSION="$("${PY}" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")"
  CURRENT_CUDA="$("${PY}" -c "import torch; print(torch.version.cuda or 'None')" 2>/dev/null || echo "")"
fi
echo "[sync-torch-cu] current torch: ${CURRENT_VERSION:-<not installed>}, cuda=${CURRENT_CUDA:-<n/a>}"

# 决策: 是否需要 reinstall
NEEDS_REINSTALL=0
if [ -z "${CURRENT_VERSION}" ]; then
  NEEDS_REINSTALL=1
  echo "[sync-torch-cu] torch 未装, 准备装 ${TORCH_VERSION}"
elif [ "${CURRENT_CUDA}" = "None" ]; then
  NEEDS_REINSTALL=1
  echo "[sync-torch-cu] torch 是 CPU wheel (cuda=None), 重装到 ${CU_TAG}"
elif [ "${CURRENT_VERSION}" != "${TORCH_VERSION}" ]; then
  NEEDS_REINSTALL=1
  echo "[sync-torch-cu] torch 版本 ${CURRENT_VERSION} != ${TORCH_VERSION}, 重装"
else
  echo "[sync-torch-cu] torch 已是 ${TORCH_VERSION}, 跳过"
fi

if [ "${NEEDS_REINSTALL}" -eq 0 ]; then
  exit 0
fi

# reinstall (uv pip install 支持直接装到 .venv)
uv pip install --reinstall \
  --index-strategy unsafe-best-match \
  "torch==${TORCH_VERSION}" \
  --index-url "${INDEX_URL}"

# 验证
"${PY}" -c "
import torch
print('[sync-torch-cu] AFTER: torch==' + torch.__version__, 'cuda=' + str(torch.version.cuda), 'avail=' + str(torch.cuda.is_available()))
assert torch.version.cuda, 'cuda 字段仍为 None, 重装失败'
print('[sync-torch-cu] OK')
"