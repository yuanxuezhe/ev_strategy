# evtrade-architecture

> 本变更的 delta spec。本变更不修改任何现有需求（openspec/specs/evtrade-architecture/spec.md 的 8 条 Requirement 保持不变），仅新增 1 条代码卫生约束，记录本次清理上升为架构原则。

## ADDED Requirements

### Requirement: Code hygiene — unused imports and non-underscore internal helpers MUST NOT accumulate
`evtrade/` 包内的源码文件 MUST NOT 留下未使用的 import 与未加下划线前缀的纯内部辅助函数。"未使用"指该项目内部（含 `tests/`、`scripts/`）无 import / 直接调用 / 字符串反射调用；"纯内部辅助函数"指仅在本文件内部被调用、无项目内外部 caller 的公开名。dev reload 工具（`*_cache` 类）、公共扩展 API（注册表 getter 等）、向后兼容 shim (`evtrade/__init__.py:95-114` 的 `sys.modules.setdefault` 层)、抽象基类的 `NotImplementedError` 占位等 MUST 保留。

#### Scenario: 死 import 已清零
- **WHEN** 静态扫描 `evtrade/**/*.py`（排除 `__pycache__`）中的 import 语句
- **THEN** 所有非 `from __future__ import` / typing / 显式 re-export shim 的 import 必须在文件内或项目内有引用方（`pytest` / `scripts/` 算项目内）

#### Scenario: 内部辅助函数以下划线标注
- **WHEN** 一个公开名（非下划线前缀）的函数 / 类仅在定义文件内部被调用，且项目内无任何外部 caller
- **THEN** 该符号 SHOULD 重命名为下划线前缀（如 `helper` → `_helper`），除非属于 dev reload 工具 / 公共扩展 API / 向后兼容 shim / 抽象基类占位等豁免类别
