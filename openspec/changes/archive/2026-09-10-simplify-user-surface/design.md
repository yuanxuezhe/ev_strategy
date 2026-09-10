# Design: simplify-user-surface

## 1. 文档合并映射

| 源 | 目标 | 处理 |
|---|---|---|
| `docs/quickstart.md` §安装uv/项目安装/故障排查/离线内网/pip兼容/uv.lock | `kbs/使用说明.md` 第 0 章扩写 | 内容搬移；`uv sync --extra gpu` 相关行删除（B 段） |
| `docs/params-workflow.md` §1-4 四步工作流 | `kbs/使用说明.md` 新章「3. 选参工作流」 | 内容搬移；§5 实盘接入段**重写**（删 `BrokerExecutor`/`scripts/run_live.py` 占位 → 自定义 bar 流 + 自定义 `Executor` 子类的示意） |
| `docs/params-workflow.md` §参数解析优先级 | `kbs/10` §1 末（或 §2 前） | 3 级优先级表搬移 |
| `docs/params-workflow.md` §完整示例 | 根 `README.md` 已有等价工作流段 | 不搬（README 保持精简），删 |
| 根 `README.md` | 保留 | "详见"段去掉 docs/ 指向，只留 kbs/ |
| `kbs/README.md` 索引 | 保留 | 使用说明行补"含 uv 安装/故障排查/选参工作流" |
| `CLAUDE.md` §6 后 / 末尾 | 微调 | 如有 docs/ 引用则改指 kbs/ |

## 2. pyproject 改动

- 删整段 `[project.optional-dependencies]`（gpu/all）+ 其注释块。
- `[tool.uv]` 上方注释 `uv sync --extra gpu: ...` 行删除，改 `uv sync: 装核心依赖（含 torch，CPU 或 CUDA wheel 由平台决定）`。
- `tests/test_pyproject.py`：删 `test_gpu_extra_listed`；加
  `test_no_optional_gpu_extra`（断言 "optional-dependencies" 不在文本中）。
  `test_cupy_dependency_removed` 保留。

## 3. `--device` 文案口径（不扩 scope，只校准）

代码不动。文档统一一句口径：
> 当前回测热路径（策略 `step` + 指标 + metrics）为 numpy/Python 标量实现，与设备无关；
> `--device` 选择 torch 后端设备（`backends.get_xp`），为后续 tensor 热路径预留，
> CPU/GPU 同一条代码路径。

落点：`kbs/15` 开头（已有类似表述，校准措辞）、`kbs/10` §1 一句、根 README 安装段一句。

## 4. 不做

- 不动 `--device` CLI 语义、`backends.py`、`vectorized_engine` 签名。
- 不合并 `engine.py` / `vectorized_engine.py`（reconcile 独立性，见
  consolidate-simplify-core design §1.1）。
- 不迁 hot-path 到 torch。

## 5. 验证门禁

```
uv run pytest -q                                  # 全绿
npx openspec validate --specs                     # 通过
grep -rn "docs/quickstart\|docs/params-workflow\|extra gpu\|BrokerExecutor" \
    README.md kbs/ CLAUDE.md pyproject.toml       # 仅剩历史记载命中:
                                                  #   kbs/README.md:12 + kbs/使用说明.md:557 (consolidate 历史行)
                                                  #   CLAUDE.md:94 (本门禁定义自身) — 均为刻意保留
grep -n "optional-dependencies" pyproject.toml    # 0 命中
ls docs/                                          # No such file or directory
python -m evtrade backtest --help                 # 无报错（示例 flag 现行）
```

> `BrokerExecutor` 在 consolidate-simplify-core 中已删代码，上述 3 处命中为"删除了什么"的
> 历史记录与门禁定义本身，不在本 change 清理范围。
```
