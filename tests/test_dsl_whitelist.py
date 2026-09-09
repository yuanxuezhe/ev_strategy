"""DSL 白名单校验与三端一致调用支持 (min/max/abs)

覆盖:
  - _validate 接受 ast.Call 到白名单函数 (之前 _FORBIDDEN_NODES 把它短路了)
  - _validate 末尾兜底拒绝未知节点 (IfExp / Subscript / Starred / keyword / ...)
  - _unparse_expr / render_numba_body 接受 min/max/abs
  - _c99_expr / render_cuda_body 把 min/max/abs 编译成 ternary / 负号三元
  - make_python_runner 在 exec 前 parse+validate (安全一致)
"""
from __future__ import annotations

import ast
import textwrap

import pytest

from evtrade.strategies import dsl as dsl_mod
from evtrade.strategies.dsl import (
    CompileError,
    _CALL_WHITELIST,
    _validate,
    compile_all,
    make_python_runner,
    render_cuda_body,
    render_numba_body,
)


# ---------- 工具: 用临时策略类的 docstring 喂 DSL ----------

class _Strategy:
    """docstring 由测试运行时注入, 见 _run_dsl"""
    compute_signal = lambda self, ctx: 0  # noqa: E731


def _make_cls_with_dsl(body: str):
    """返回一个类, compute_signal.__doc__ == body

    DSL 工厂要求 state_spec (空 dict 表示无持久状态, 测试用无状态策略)。
    """
    cls = type("_S", (), {
        "compute_signal": staticmethod(lambda ctx: 0),
        "state_spec": {},   # DSL 必填字段; 测试用空 (无持久状态)
        "params_spec": {},
    })
    cls.compute_signal.__doc__ = body
    return cls


# ---------- _validate 行为 ----------

def test_validate_accepts_whitelisted_calls():
    for fn in ("min", "max", "abs"):
        tree = ast.parse(f"def _f(ctx):\n    return {fn}(ctx.cur_high, ctx.cur_low)\n")
        _validate(tree)  # 不抛


def test_validate_rejects_unknown_call():
    tree = ast.parse("def _f(ctx):\n    return math.sin(ctx.cur_high)\n")
    with pytest.raises(CompileError, match="不支持调用"):
        _validate(tree)


def test_validate_rejects_keyword_call_even_for_whitelisted_fn():
    tree = ast.parse("def _f(ctx):\n    return min(a=ctx.cur_high, b=ctx.cur_low)\n")
    with pytest.raises(CompileError, match="不支持调用"):
        _validate(tree)


@pytest.mark.parametrize("node_src", [
    "ctx.cur_high[0]",                 # ast.Subscript
    "a if ctx.cur_high > 0 else b",    # ast.IfExp
])
def test_validate_rejects_unknown_nodes(node_src):
    tree = ast.parse(f"def _f(ctx):\n    return {node_src}\n")
    with pytest.raises(CompileError):
        _validate(tree)


def test_validate_rejects_augassign_in_statement_position():
    """增强赋值 (AugAssign) 在语句位置也是 DSL 不允许的"""
    tree = ast.parse("def _f(ctx):\n    ctx.cur_high += 1\n    return 0\n")
    with pytest.raises(CompileError):
        _validate(tree)


# ---------- _unparse_expr / render_numba_body 接受 min/max/abs ----------

@pytest.mark.parametrize("expr,expect_substr", [
    ("min(ctx.cur_high, ctx.cur_low)", "min("),
    ("max(ctx.cur_high, ctx.cur_low)", "max("),
    ("abs(ctx.cur_high - ctx.cur_low)", "abs("),
])
def test_render_numba_body_whitelisted_calls(expr, expect_substr):
    cls = _make_cls_with_dsl(f"return {expr}\n")
    out = render_numba_body(cls)
    assert expect_substr in out


def test_render_numba_body_rejects_unknown_call():
    cls = _make_cls_with_dsl("return math.sin(ctx.cur_high)\n")
    with pytest.raises(CompileError):
        render_numba_body(cls)


# ---------- _c99_expr / render_cuda_body 三端一致 ----------

@pytest.mark.parametrize("expr,expect_substr", [
    ("min(ctx.cur_high, ctx.cur_low)", "<"),       # min(a,b) -> (a<b?a:b)
    ("max(ctx.cur_high, ctx.cur_low)", ">"),       # max(a,b) -> (a>b?a:b)
    ("abs(ctx.cur_high - ctx.cur_low)", "< 0 ?"),  # abs(x) -> (x<0?-x:x)
])
def test_render_cuda_body_whitelisted_calls(expr, expect_substr):
    cls = _make_cls_with_dsl(f"return {expr}\n")
    out = render_cuda_body(cls)
    assert expect_substr in out


def test_render_cuda_body_min_with_three_args():
    """min/max/abs 现在允许 ≥2 参数 (白名单只校验函数名, 不限 arity); CUDA
    渲染如实保留为 `min(a, b, c);` —— C++ std::min 是 2 元, 3 元编译会失败,
    但 DSL 校验层面不拦截 (报错责任交给 CUDA 编译阶段)。
    旧版本曾要求 CompileError, 现改为记录 arity 校验的当前行为。"""
    cls = _make_cls_with_dsl("return min(ctx.cur_high, ctx.cur_low, ctx.cur_close)\n")
    out = render_cuda_body(cls)
    assert "min(cur_high, cur_low, cur_close)" in out


# ---------- make_python_runner 安全一致: parse+validate 后才 exec ----------

def test_make_python_runner_executes_whitelisted_call():
    cls = _make_cls_with_dsl("return abs(ctx.cur_high - ctx.cur_low)\n")
    runner = make_python_runner(cls)

    class _Ctx:
        cur_high = 5.0
        cur_low = 2.0
    assert runner(_Ctx) == 3.0


def test_make_python_runner_rejects_unknown_call_via_validate():
    cls = _make_cls_with_dsl("return math.sin(ctx.cur_high)\n")
    # 现在 make_python_runner 先 validate, 应当抛 CompileError 而不是 exec 任意代码
    with pytest.raises(CompileError):
        make_python_runner(cls)


def test_make_python_runner_rejects_unknown_node_via_validate():
    cls = _make_cls_with_dsl("return ctx.cur_high[0]\n")
    with pytest.raises(CompileError):
        make_python_runner(cls)


# ---------- compile_all 三端一致 ----------

def test_compile_all_renders_min_on_all_targets():
    cls = _make_cls_with_dsl("return min(ctx.cur_high, ctx.cur_low)\n")
    out = compile_all(cls)
    assert "min(" in out["numba"]
    assert "<" in out["cuda"]   # ternary (a<b?a:b)
    # python 端是原 docstring body
    assert "min(ctx.cur_high, ctx.cur_low)" in out["python"]


# ---------- 白名单集合: Python 内置 3 项 + indicators 子包增量 API (2026-09 解耦后) ----------

def test_call_whitelist_contents():
    """DSL 可调函数白名单: Python 内置 (min/max/abs) + indicators 子包全部增量 API
    (ema/atr/rsi/sma/boll 的 push + current; ema_channel_push + ema_channel_current)。
    三端 (Python exec / numba @njit / CUDA __device__) 同源渲染, 任何新增指标
    必须同时扩展 evtrade.indicators 子包与本白名单, 三端自动一致 (kbs/11 §5)。"""
    from evtrade.strategies import dsl as dsl_mod
    expected = frozenset({
        # Python builtins
        "min", "max", "abs",
        # evtrade.indicators.ema
        "ema_push", "ema_current", "ema_channel_push", "ema_channel_current",
        # evtrade.indicators.atr
        "atr_push", "atr_current",
        # evtrade.indicators.rsi
        "rsi_push", "rsi_current",
        # evtrade.indicators.boll
        "sma_push", "sma_current", "boll_push", "boll_current",
    })
    assert dsl_mod._CALL_WHITELIST == expected
