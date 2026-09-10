# evtrade

策略回测 / 参数扫描 / 行情回放 — PyTorch 统一 CPU/GPU 后端 (VectorizedStrategy.step 契约)。

## 安装 (uv)

```bash
# CPU 路径 (默认; torch CPU wheel 即可)
uv sync

# GPU 路径 (torch CUDA wheel, 需 NVIDIA + CUDA runtime)
uv sync --extra gpu

# 开发 (含 pytest)
uv sync --extra dev
```

`uv` 会自动建虚拟环境、装核心依赖。`pyproject.toml` 列出全部依赖 (numpy / pandas / sqlalchemy / pymysql / torch)。

## 启动

### CLI (三种入口等效)

```bash
uv run python -m evtrade ...
uv run python -m evtrade.cli ...
uv run evtrade ...                      # pyproject.toml [project.scripts] 注册
```

### 子命令

```bash
uv run python -m evtrade backtest --strategy channel_deviation --code 159992.SZ \
  --start 20250101 --end 20260903 --device auto

uv run python -m evtrade sweep --strategy channel_deviation --code 159992.SZ \
  --start 20250101 --end 20260903 \
  --grid low1=1.0,1.5,2.0 --grid low2=0.5,1.0,1.5 \
  --device auto --out sweep_results.csv \
  --save-defaults                        # 跑完自动选最优并落盘 + git commit

uv run python -m evtrade params save channel_deviation \
  --params "low1:1.5;low2:1.0;high1:1.5;high2:0.5"
uv run python -m evtrade params show channel_deviation
uv run python -m evtrade params list

uv run python -m evtrade replay --log bars_log.csvz --against-ref
```

### 测试

```bash
uv run pytest                              # 全部 (GPU 测试默认 skip)
uv run pytest -m "not gpu"                 # 仅 CPU
uv run pytest -m gpu --gpu                 # GPU 路径 (需 torch CUDA)
```

## 工作流：网格 → 选参 → 落盘 → 单次回测/实盘

```bash
# 1. 网格扫描
uv run python -m evtrade sweep --strategy channel_deviation --code 159992.SZ \
  --start 20250101 --end 20260903 \
  --grid low1=1.0,1.5,2.0 --grid low2=0.5,1.0,1.5 \
  --grid high1=1.0,1.5,2.0 --grid high2=0.3,0.5,0.8 \
  --splits 20250901 20260301 --device auto \
  --out sweep_results.csv --save-defaults

# 2. (程序自动选最优并落盘 evtrade/strategies/_defaults/channel_deviation.json,
#    单独 git commit 含中文选择原因)

# 3. 单次回测 (不传参数, 自动读默认)
uv run python -m evtrade backtest --strategy channel_deviation --code 159992.SZ \
  --start 20260101 --end 20260903 --device auto

# 4. 实盘接入 (live 子命令未实现, 占位: 直接调 loader)
uv run python -c "
from evtrade.strategies._defaults_loader import load
print(load('channel_deviation'))
"
```

## 参数解析优先级 (CLI)

`backtest` / `sweep` 子命令解析策略参数时按以下顺序:

1. CLI `--params "k:v;..."` 显式传入 (任意策略, 类型自动推导 int/float/bool/str)
2. `evtrade/strategies/_defaults/<strategy>.json` 落盘默认
3. 空 dict → `VectorizedStrategy.params_spec` 的 `default`

实盘/CI 机器用 `EVTRADE_DEFAULTS_DIR` 环境变量切换落盘目录, 不污染仓库:

```bash
export EVTRADE_DEFAULTS_DIR=/etc/evtrade/defaults
```

## 仓库结构

```
evtrade/
  __init__.py          顶层 re-export + 少量 sys.modules 兼容垫片
  __main__.py          python -m evtrade 入口
  backends.py          get_xp(device) -> torch.device (CPU/GPU 统一后端) + resolve_device
  cli.py               argparse 子命令 (backtest/sweep/replay/params)
  primitives.py        Bar 结构
  core/                engine / vectorized_engine / sweep / replay / tsbucket / data / metrics
                       / timeutils / aggregator / permutation / config / _harness
  strategies/          VectorizedStrategy 唯一基类 + 注册表 + 默认参数 (_defaults/)
  execution/           Executor 抽象 (Simulated) + trade_decision 单一成交决策
  indicators/          ema (*_step 增量 + numpy 批量参考; 仅 EMA)
tests/                 140+ pytest 用例
docs/                  使用说明
kbs/                   中文设计文档 (spec 投影, 15 份)
```

## 详见

- `docs/params-workflow.md` — 四步工作流完整说明
- `docs/quickstart.md` — 快速开始 (含 uv 详细命令)
- `kbs/` — 中文设计文档（spec 投影，15 份，与代码同步）
