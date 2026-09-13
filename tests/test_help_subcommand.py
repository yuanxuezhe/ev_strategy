"""help 子命令 + desc 字段

锁定:
  - `params_spec` 三策略全部字段都有 desc 字段 (commit 1 落地后)
  - `_resolve_params` 不读 desc (零行为影响)
  - desc 是可选字段: 缺 desc 仍能 help 出来
  - CLI `python -m evtrade help` 列出所有策略
  - CLI `help --strategy <name>` 输出对齐表格含 desc
  - CLI `help --strategy unknown` exit 非 0 + 列出可用
  - CLI `help --strategy A --strategy B` 输出两段
"""
from __future__ import annotations

import subprocess
import sys

from evtrade.cli import help_main
from evtrade.strategies import (
    _STRATEGIES,
    _STRATEGY_SUMMARIES,
    available_strategies,
    get_strategy,
    get_strategy_class,
    get_strategy_param_spec,
)


# ============ 1. 内部 Python API ============

def test_help_lists_all_registered_strategies():
    """available_strategies 含三策略"""
    names = available_strategies()
    for n in ("channel_deviation", "ma_crossover", "filtered_mr"):
        assert n in names, f"{n} 未注册; available={names}"


def test_each_strategy_has_params_spec_with_desc_field():
    """三策略所有 params_spec 字段都有 desc 字段"""
    for name in available_strategies():
        spec = get_strategy_param_spec(name)
        assert spec, f"{name} params_spec 空"
        for k, schema in spec.items():
            assert "desc" in schema, (
                f"{name}.{k} 缺 desc 字段; spec={schema}")
            assert isinstance(schema["desc"], str), (
                f"{name}.{k}.desc 非字符串: {schema['desc']!r}")
            assert len(schema["desc"]) >= 5, (
                f"{name}.{k}.desc 长度太短 ({len(schema['desc'])}): "
                f"{schema['desc']!r}")


def test_strategy_summaries_cover_registered():
    """_STRATEGY_SUMMARIES 含所有已注册策略的简短描述"""
    for name in available_strategies():
        assert name in _STRATEGY_SUMMARIES, (
            f"{name} 缺 _STRATEGY_SUMMARIES 简短描述")
        s = _STRATEGY_SUMMARIES[name]
        assert isinstance(s, str) and len(s) >= 5, (
            f"{name} 简短描述异常: {s!r}")


def test_desc_field_does_not_affect_resolve_params():
    """desc 是 help 展示用, _resolve_params 不读 → 行为不变"""
    for name in available_strategies():
        s = get_strategy(name)
        assert s.params, f"{name} 实例化失败"


def test_help_works_with_partial_desc():
    """缺 desc 的字段 help 仍能列出 ('(无描述)' 占位); 验证 desc 是可选字段"""
    cls = _STRATEGIES["channel_deviation"]
    orig_spec = cls.params_spec
    spec_copy = {k: dict(v) for k, v in orig_spec.items()}
    spec_copy["low1"].pop("desc", None)
    cls.params_spec = spec_copy
    try:
        result = help_main(["--strategy", "channel_deviation"])
        assert result is None, f"help_main 返 {result}, 期望 None (成功)"
    finally:
        cls.params_spec = orig_spec


# ============ 2. CLI 端 (subprocess) ============

def _run(*args):
    """跑 python -m evtrade help ... 返 (returncode, stdout, stderr)"""
    r = subprocess.run(
        [sys.executable, "-m", "evtrade", "help", *args],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode, r.stdout, r.stderr


def test_cli_help_lists_strategies():
    rc, out, _ = _run()
    assert rc == 0, f"exit={rc}; out={out!r}"
    for name in ("channel_deviation", "ma_crossover", "filtered_mr"):
        assert name in out, f"列表模式缺 {name}; out={out!r}"


def test_cli_help_strategy_detail_shows_table():
    rc, out, _ = _run("--strategy", "channel_deviation")
    assert rc == 0, f"exit={rc}; out={out!r}"
    # 表头
    assert "channel_deviation" in out
    assert "参数" in out
    # 至少含部分参数名
    for k in ("low1", "low2", "high1", "high2", "tf1"):
        assert k in out, f"缺参数 {k}; out={out!r}"
    # 含 type / 默认值 / 范围 / 描述 字样
    assert "类型" in out
    assert "默认" in out
    assert "范围" in out
    # desc 内容应出现在输出
    assert "下轨" in out
    # 锁存约束提示
    assert "validators" in out or "校验器" in out


def test_cli_help_unknown_strategy_exits_nonzero():
    rc, out, _ = _run("--strategy", "does_not_exist")
    assert rc != 0, f"未知名策略应返非 0, 实际 {rc}; out={out!r}"
    assert "未知" in out, f"缺错误提示 '未知'; out={out!r}"
    # 错误信息应列出已注册策略
    assert "channel_deviation" in out
    assert "ma_crossover" in out


def test_cli_help_multiple_strategies():
    rc, out, _ = _run("--strategy", "channel_deviation",
                      "--strategy", "ma_crossover")
    assert rc == 0, f"exit={rc}; out={out!r}"
    # 两段都出现
    assert out.count("channel_deviation") >= 1
    assert out.count("ma_crossover") >= 1


def test_cli_root_help_lists_help_subcommand():
    """python -m evtrade --help 列出 4 子命令 (含 help)"""
    rc, out, _ = subprocess.run(
        [sys.executable, "-m", "evtrade", "--help"],
        capture_output=True, text=True, timeout=30,
    ).returncode, subprocess.run(
        [sys.executable, "-m", "evtrade", "--help"],
        capture_output=True, text=True, timeout=30,
    ).stdout, ""
    assert rc == 0
    assert "help" in out
    for sub in ("backtest", "sweep", "params"):
        assert sub in out, f"root help 缺子命令 {sub}; out={out!r}"


# ============ 3. _format 辅助 ============

def test_fmt_helpers():
    from evtrade.cli import _fmt_default, _fmt_range, _fmt_type
    assert _fmt_type({"type": int}) == "int"
    assert _fmt_type({"type": float}) == "float"
    assert _fmt_type({}) == "any"
    assert _fmt_default({"default": 1.5}) == "1.5"
    assert _fmt_default({"default": "1h"}) == "'1h'"
    assert _fmt_default({}) == "(必填)"
    assert _fmt_range({}) == "any"
    assert _fmt_range({"min": 0, "max": 100}) == "[0, 100]"
    assert _fmt_range({"min": 1.0}) == "[1.0, None]"
