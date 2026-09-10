# Tasks: simplify-user-surface

## 1. pyproject GPU 声明去冗余
- [x] 1.1 删 `pyproject.toml` `[project.optional-dependencies]`（gpu/all）+ 注释；校准 `[tool.uv]` 上方 uv sync 注释
- [x] 1.2 `tests/test_pyproject.py`：删 `test_gpu_extra_listed`；加 `test_no_optional_gpu_extra`

## 2. 文档合并（docs/ → kbs/）
- [x] 2.1 `kbs/使用说明.md` 第 0 章并入 quickstart 内容（uv 安装/故障排查/离线内网/pip 兼容/uv.lock），删 `--extra gpu` 行
- [x] 2.2 `kbs/使用说明.md` 新增「选参工作流」章（params-workflow §1-4）；实盘接入段重写（自定义 bar 流 + Executor，删 BrokerExecutor 占位）
- [x] 2.3 参数解析优先级段并入 `kbs/10`
- [x] 2.4 删 `docs/` 目录；根 README「详见」去 docs/ 指向；`kbs/README` 索引行更新
- [x] 2.5 `--device` 统一文案口径落点：kbs/15、kbs/10、根 README 各一句

## 3. 验证
- [x] 3.1 `uv run pytest -q` 全绿
- [x] 3.2 design §5 grep 门禁全 0 命中 + `ls docs/` 不存在
- [x] 3.3 主 spec 合入 delta；`npx openspec validate --specs`
- [x] 3.4 `openspec archive simplify-user-surface`；commit（带 change 名）+ push
