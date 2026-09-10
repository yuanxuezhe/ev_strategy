# evtrade

策略回测 / 参数扫描 / 行情回放 — PyTorch 统一 CPU/GPU 后端 (VectorizedStrategy.step 契约)。

## 安装 (uv)

```bash
uv sync                  # 核心依赖 (含 torch; 无独立 GPU extra — CPU/CUDA 是同一个 torch 包的两个 wheel)
uv sync --group dev      # 开发工具 (pytest 等)
```

要跑 GPU：`pip install torch --index-url https://download.pytorch.org/whl/cu124`
（NVIDIA + CUDA runtime；CPU 与 CUDA wheel 不能并存）。详细安装 / 故障排查 /
离线内网 / 非 uv 流程见 `kbs/使用说明.md` §0。

> **CPU/GPU 统一**：后端为 torch 单端（`backends.get_xp` 路由 torch.device），策略代码一份、
> CPU/GPU 同一条路径；`--device {cpu,gpu,auto}`（默认 auto）选择设备，`--device gpu` 无 CUDA
> 抛错、`auto` 降级 cpu + warning。当前回测热路径为 numpy/Python 标量实现（与设备无关），
> `--device` 为统一后端预留。

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
kbs/                   中文设计文档 (spec 投影, 15 份) + 使用说明 (操作手册)
```

## 详见

- `kbs/使用说明.md` — 操作手册（安装/故障排查/选参工作流/命令清单/输出解读/FAQ）
- `kbs/10-配置参数与运行指南.md` — 每个参数的工程含义
- `kbs/13-绩效评估与鲁棒选参框架.md` — 评分/选参方法论（WFO/邻域衰减/置换检验）
- `kbs/` — 中文设计文档（spec 投影，15 份，与代码同步）
