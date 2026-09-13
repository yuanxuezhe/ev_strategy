## Context

2026-09-12 的 `add-gpu-extra-pyproject` 引入 `[project.optional-dependencies].gpu =
["torch==2.9.0+cu128"]`, 想给 GPU 机器一条 `uv sync --extra gpu` 的一键安装路径。
但在无 NVIDIA 显卡的 Windows 机器上复现: 默认 `uv sync` 装到 `torch 2.9.0+cu128`
(GPU wheel, 几百 MB), 用户困惑"我没显卡为什么装 cu128"。

## 根因

`uv lock` 的语义是**把 base 依赖与项目声明的所有 extras 一起解析进同一份 `uv.lock`**。
torch 被 base(`torch>=2.0`)与 gpu extra(`torch==2.9.0+cu128`)同时引用。`+cu128` 是
local version, 只存在于 cu128 索引; uv 为让两者都满足, 把 torch **按包名统一**到
`2.9.0+cu128`。于是 base torch 也被带成 cu128 —— 即便 `uv sync` 不带 `--extra gpu`,
装的 base torch 版本号已是 cu128。

关键约束(已实测验证):
1. 删除 gpu extra + cu128 source 后 `uv lock` → torch = pypi `2.14.0`(CPU), lock 内
   0 个 cu128 引用。
2. 保留 gpu extra → torch 统一为 `2.9.0+cu128`。
3. 给 cu128 索引加 `explicit = true` 只解决了"索引泄漏成所有包 fallback"的问题
   (numpy 等), 但**无法**阻止 base 与 extra 的 torch 版本统一 —— 统一发生在包名层面,
   不是索引层面。
4. `uv 0.11.7` 无 `uv lock --extra` / `--no-extra` 开关, 无法只锁 base。
5. `uv.lock` 在 `.gitignore` 里, **不入库**, 各机器本地生成。

结论: 一份共享 lock 无法表达"默认 CPU + 某 extra 才 cu128"两个不同版本。要么默认
CPU(extra 机制不成立), 要么默认 cu128(害 CPU 机器)。

## 方案选择

**选定: GPU 走 `scripts/sync-torch-cu.sh`(uv sync 后就地覆盖), 删除 pyproject gpu extra。**

理由:
- `sync-torch-cu.sh` 已存在且功能完备(探测 CPU wheel → `uv pip install --reinstall`
  到 cu128 → 幂等 → `EVT_TORCH_CU_TAG` 覆写), 且覆盖发生在 **venv 层面, 不改 lockfile**,
  天然不污染共享 lock。
- 删除 extra 后, 默认 `uv sync` 回到 pypi CPU torch, 满足"CPU 机器默认 CPU"的核心诉求。
- `uv.lock` 不入库, 故本机重 lock 不影响协作者; GPU 覆盖是本地动作。

**放弃的备选**:
- 保留 extra + 加 `explicit`: 不解决问题(版本统一在包名层面)。
- 把 `uv.lock` 提交入库: 与 `.gitignore` 现状冲突, 且 GPU/CPU 机器 lock 会互相覆盖。
- 用两个独立环境(venv)区分 GPU/CPU: 超出本项目范围, 复杂度高。

## 风险

- **BREAKING**: 任何依赖 `uv sync --extra gpu` 的脚本 / 文档失效, 需改走脚本。已全量
  sweep 更新 kbs / CLAUDE.md / tests; archive changes 里的历史引用不动(历史文档)。
- GPU 机器需多跑一条命令(`bash scripts/sync-torch-cu.sh`)。可接受 —— 本就是"一次性"。
- `sync-torch-cu.sh` 用 `uv pip install`(venv 层面)绕开 lock, 与项目"lockfile 为单一事实源"
  的精神略有张力; 但 torch 是**运行时后端选择**, 非项目核心依赖契约, 且该脚本在
  `add-gpu-extra-pyproject` 中本就是受支持路径, 风险可控。
