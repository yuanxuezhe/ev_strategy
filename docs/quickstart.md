# 快速开始 (uv)

## 安装 uv

按 [astral-sh/uv](https://github.com/astral-sh/uv) 安装:

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 项目安装

```bash
# 仓库根目录

# 1. CPU 路径 (默认; 不需要 CUDA)
uv sync

# 2. GPU 路径 (torch CUDA wheel, 需 NVIDIA + CUDA runtime)
uv sync --extra gpu

# 3. 装开发工具 (pytest 等)
uv sync --extra dev

# 4. 装 GPU + 开发
uv sync --all-extras
```

`uv sync` 自动:
- 选最新兼容 Python (>= 3.8, 由 pyproject.toml requires-python 决定)
- 建 `.venv` 虚拟环境
- 解析 lockfile (`uv.lock`), 装所有依赖
- 编辑模式装本项目 (`-e .`) → `evtrade` 命令与 `python -m evtrade` 都可用

## 启动

```bash
# 跑 sweep
uv run python -m evtrade sweep \
  --strategy channel_deviation --code 159992.SZ \
  --start 20250101 --end 20260903 \
  --grid low1=1.0,1.5,2.0 --grid low2=0.5,1.0,1.5 \
  --device auto --out sweep_results.csv

# 跑测试
uv run pytest

# 仅 CPU 测试 (默认 GPU 测试 skip)
uv run pytest -m "not gpu"

# 落盘默认参数
uv run python -m evtrade params save channel_deviation \
  --params "low1:1.5;low2:1.0;high1:1.5;high2:0.5"

# 等价的简写 (pyproject.toml [project.scripts] 注册的入口)
uv run evtrade sweep --strategy channel_deviation ...
```

## 故障排查

### `uv sync` 报 requires-python 不满足

```text
error: Package 'foo' requires a different Python: 3.8.x not in '>=3.10'
```

→ uv 选定的 Python 版本与某个依赖要求不符。显式指定 Python:

```bash
uv python install 3.11
uv sync --python 3.11
```

### torch 报 CUDA driver 不匹配

```text
Torch not compiled with CUDA enabled / CUDA path could not be detected.
```

→ `uv sync` 默认装 torch CPU wheel，只跑 CPU。要跑 GPU 需装 CUDA 版 torch：

```bash
uv sync --extra gpu            # 或
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4 示例
```

> CPU / CUDA 是同一个 `torch` 包的两个 wheel，不能并存。换运行时需要重装 torch。
> 装对后 `--device auto` 会自动路由到 CUDA，`--device gpu` 在 CUDA 不可用时打 warning 回退 cpu。

### `uv lock` 不一致 (团队协作时)

```bash
uv lock --check          # 验证 uv.lock 与 pyproject.toml 一致
uv lock                   # 重新生成 uv.lock
```

### 离线 / 公司内网

```bash
uv sync --offline          # 用本地缓存
uv export > requirements.txt  # 锁文件导出 (供 pip 离线装)
```

## 与非 uv 流程兼容

无 uv 也可:

```bash
# pip (在仓库根目录)
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux/macOS
pip install -e .                  # 编辑模式装本项目 + 依赖
python -m evtrade sweep ...
```

`pyproject.toml` 是单一事实源, pip + setuptools 也能用 (装 wheel); uv 是
速度更快、依赖解析更准确的现代选择。

## uv.lock 与团队协作

`uv.lock` 默认**不入仓库** (`.gitignore` 已排除):
- 每位开发者本地 `uv lock` 生成自己的 lockfile, 跨平台 lockfile 差异小
  (uv 锁平台无关的包版本, 仅 hash 与 binary 分开发散)
- CI / 实盘机可直接 `uv sync` 而无需 lockfile (从 `pyproject.toml` 解析)

如需锁定全员一致的版本:

```bash
git add -f uv.lock
git commit -m "构建: 锁定依赖 uv.lock (团队一致)"
```

注意 lockfile 较大 (~ 300 KB), 包含所有平台 hash。
