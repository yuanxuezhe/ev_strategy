"""state_spec 通用化测试: 框架支持任意持久状态字段

锁定 (Phase 2 Commit 4 终验):
  - 零状态策略 (state_spec = {}): Python + numba + CUDA 三端可用,
    ctx 上无策略专属字段, framework 不再注入任何 baked-in 字段
  - 不同状态策略: 自定义字段名 (非 channel_deviation 的 low_hit 等),
    framework 工厂化投影到 Python / numba / CUDA 三端
  - make_dsl_ctx / build_ctx_to_kernel_map / build_cuda_sig_fields /
    build_cuda_device_header 工厂正确按 state_spec 生成产物
  - 缺 state_spec 时编译期抛 CompileError (DSL 策略必须声明)
"""
from __future__ import annotations

import numpy as np
import pytest

from evtrade.strategies import (StrategyBase, register_strategy,
                                 get_strategy_class,
                                 get_strategy_state_spec,
                                 make_dsl_ctx, build_ctx_to_kernel_map,
                                 build_cuda_sig_fields, build_cuda_device_header)
from evtrade.strategies.dsl import (CompileError, render_numba_body, render_cuda_body,
                                    build_cuda_state_decls,
                                    build_cuda_strategy_check_call)


# ============ 临时策略类 (本测试独享, 不污染 strategies 包) ============

@register_strategy("zero_state_demo")
class ZeroStateStrategy(StrategyBase):
    """零状态策略: state_spec = {}, DSL body 只读框架字段 (cur_ts/up/dw)"""

    params_spec = {"threshold": {"default": 1.0, "type": float}}

    state_spec = {}

    def compute_signal(self, ctx):
        """DSL body: 仅用框架字段"""
        pass  # DSL 注入

    def check(self, cur, up, dw):
        if not up or not dw:
            return None, {}
        from .dsl import dsl_check
        return {1: "BUY", -1: "SELL"}.get(
            dsl_check(self, make_dsl_ctx(self.__class__), cur, up, dw)
        ), {}


ZeroStateStrategy.compute_signal.__doc__ = """\
if ctx.cur_ts > 0 and ctx.up > ctx.p0:
    return 1
if ctx.cur_ts > 0 and ctx.dw < -ctx.p0:
    return -1
return 0
"""


@register_strategy("diff_state_demo")
class DiffStateStrategy(StrategyBase):
    """不同状态策略: state_spec 与 channel_deviation 完全不同

    用任意字段名 (非 channel_deviation 的 low_hit/high_hit/...) 验证
    framework 工厂化按策略类声明投影, 无 baked-in 字段。
    """

    params_spec = {"limit": {"default": 2.5, "type": float}}

    state_spec = {
        "counter": {"type": int, "default": 0},
        "touched_up": {"type": bool, "default": False},
        "last_level": {"type": float, "default": 0.0},
    }

    def compute_signal(self, ctx):
        """DSL body: 操作用自定义状态字段"""
        pass  # DSL 注入

    def check(self, cur, up, dw):
        if not up or not dw:
            return None, {}
        from .dsl import dsl_check
        return {1: "BUY", -1: "SELL"}.get(
            dsl_check(self, make_dsl_ctx(self.__class__), cur, up, dw)
        ), {}


DiffStateStrategy.compute_signal.__doc__ = """\
ctx.counter = ctx.counter + 1
if ctx.cur_high > ctx.last_level + ctx.p0:
    ctx.touched_up = True
if ctx.touched_up and ctx.cur_low < ctx.last_level - ctx.p0:
    ctx.touched_up = False
    ctx.last_level = ctx.cur_low
    return -1
ctx.last_level = ctx.cur_low
return 0
"""


# ============ 工厂函数单测 ============

def test_get_strategy_state_spec_returns_declared_dict():
    """get_strategy_state_spec 返回策略类声明的 state_spec (引用相等)"""
    spec = get_strategy_state_spec("diff_state_demo")
    assert "counter" in spec and "touched_up" in spec and "last_level" in spec
    assert spec["counter"]["type"] is int
    assert spec["touched_up"]["type"] is bool
    assert spec["last_level"]["type"] is float
    assert spec["counter"]["default"] == 0
    assert spec["touched_up"]["default"] is False
    assert spec["last_level"]["default"] == 0.0


def test_get_strategy_state_spec_empty_for_zero_state():
    """零状态策略返回空 dict (非 None)"""
    spec = get_strategy_state_spec("zero_state_demo")
    assert spec == {}


def test_make_dsl_ctx_zero_state_has_only_framework_fields():
    """make_dsl_ctx(零状态策略): ctx 只有框架字段 (cur_ts/up/dw/p0..p15),
    无任何策略专属字段"""
    ctx = make_dsl_ctx(get_strategy_class("zero_state_demo"))
    # 框架字段
    for f in ("cur_ts", "cur_open", "cur_high", "cur_low", "cur_close",
              "cur_volume", "up", "dw"):
        assert hasattr(ctx, f), f"缺框架字段 {f}"
    # 参数寄存器
    for i in range(16):
        assert hasattr(ctx, f"p{i}")
    # 零状态: 无策略专属字段 (即不应有 channel_deviation 之类残留)
    assert not hasattr(ctx, "low_hit")
    assert not hasattr(ctx, "high_hit")
    assert not hasattr(ctx, "lock_ts")
    assert not hasattr(ctx, "low_acted")
    assert not hasattr(ctx, "high_acted")


def test_make_dsl_ctx_diff_state_has_declared_fields_with_defaults():
    """make_dsl_ctx(不同状态策略): ctx 含 state_spec 字段 (初值 = default)"""
    ctx = make_dsl_ctx(get_strategy_class("diff_state_demo"))
    assert ctx.counter == 0
    assert ctx.touched_up is False
    assert ctx.last_level == 0.0
    # 框架字段也在
    assert hasattr(ctx, "cur_ts")
    assert hasattr(ctx, "up")


def test_build_ctx_to_kernel_map_zero_state_only_framework():
    """build_ctx_to_kernel_map(零状态): 只含框架字段 + p0..p15, 无策略字段"""
    cls = get_strategy_class("zero_state_demo")
    m = build_ctx_to_kernel_map(cls)
    for f in ("cur_ts", "cur_open", "cur_high", "cur_low", "cur_close",
              "cur_volume", "up", "dw"):
        assert m[f] == f
    for i in range(16):
        assert m[f"p{i}"] == f"p{i}"
    # 不含 channel_deviation 历史字段
    for old in ("low_hit", "high_hit", "lock_ts", "low_acted", "high_acted"):
        assert old not in m, f"残留 channel_deviation 字段 {old}"


def test_build_ctx_to_kernel_map_diff_state_identity():
    """build_ctx_to_kernel_map(不同状态): 单名空间 identity 映射"""
    cls = get_strategy_class("diff_state_demo")
    m = build_ctx_to_kernel_map(cls)
    assert m["counter"] == "counter"
    assert m["touched_up"] == "touched_up"
    assert m["last_level"] == "last_level"


def test_build_cuda_sig_fields_zero_state():
    """CUDA 签名字段 (零状态): 只含框架 + p0..p7"""
    cls = get_strategy_class("zero_state_demo")
    sig = build_cuda_sig_fields(cls)
    for f in ("cur_ts", "cur_open", "cur_high", "cur_low", "cur_close",
              "cur_volume", "up", "dw"):
        assert f in sig
    for i in range(8):
        assert f"p{i}" in sig


def test_build_cuda_sig_fields_diff_state():
    """CUDA 签名字段 (不同状态): 含 state_spec 字段"""
    cls = get_strategy_class("diff_state_demo")
    sig = build_cuda_sig_fields(cls)
    assert "counter" in sig
    assert "touched_up" in sig
    assert "last_level" in sig


def test_build_cuda_device_header_diff_state_has_state_args():
    """CUDA device header (不同状态): state_spec 字段以 int/long long/double 出现

    Python int → long long (CUDA 64-bit), bool → int (C 无 bool 用 1/0),
    float → double。
    """
    cls = get_strategy_class("diff_state_demo")
    header = build_cuda_device_header(cls)
    assert "long long &counter" in header
    assert "int &touched_up" in header
    assert "double &last_level" in header


# ============ CUDA 寄存器声明 + 调用 arg 工厂 (build_cuda_state_decls / build_cuda_strategy_check_call) ============

def test_build_cuda_state_decls_zero_state_empty():
    """零状态策略: 寄存器声明为空字符串"""
    cls = get_strategy_class("zero_state_demo")
    decls = build_cuda_state_decls(cls)
    assert decls == ""


def test_build_cuda_state_decls_zero_state_no_residual_field():
    """零状态策略: 不应残留任何 baked-in 字段声明"""
    cls = get_strategy_class("zero_state_demo")
    decls = build_cuda_state_decls(cls)
    for old in ("low_hit", "high_hit", "lock_ts", "low_acted", "high_acted"):
        assert old not in decls, f"零状态残留声明 {old}"


def test_build_cuda_strategy_check_call_zero_state_empty():
    """零状态策略: 调用处 state arg 列表为空字符串"""
    cls = get_strategy_class("zero_state_demo")
    call_args = build_cuda_strategy_check_call(cls)
    assert call_args == ""


def test_build_cuda_state_decls_diff_state_contains_all_fields():
    """不同状态策略: 寄存器声明含所有 state_spec 字段 (按顺序, 按类型)"""
    cls = get_strategy_class("diff_state_demo")
    decls = build_cuda_state_decls(cls)
    # 按 state_spec 顺序: counter (int) → touched_up (bool) → last_level (float)
    assert "long long counter = 0;" in decls
    assert "int touched_up = 0;" in decls          # bool default False → 0
    assert "double last_level = 0.0;" in decls


def test_build_cuda_state_decls_diff_state_field_order_preserved():
    """state_spec 字段顺序保留 (与 build_cuda_device_header 顺序一致)"""
    cls = get_strategy_class("diff_state_demo")
    decls = build_cuda_state_decls(cls)
    counter_pos = decls.index("counter")
    touched_pos = decls.index("touched_up")
    last_pos = decls.index("last_level")
    assert counter_pos < touched_pos < last_pos


def test_build_cuda_strategy_check_call_diff_state_in_order():
    """不同状态策略: 调用处 arg 列表与 state_spec 顺序一致 (逗号分隔)"""
    cls = get_strategy_class("diff_state_demo")
    call_args = build_cuda_strategy_check_call(cls)
    assert call_args == "counter, touched_up, last_level"


def test_build_cuda_state_decls_call_args_match_signature_order():
    """build_cuda_strategy_check_call 输出顺序与 build_cuda_device_header
    state 部分一致 (CUDA 编译时 strategy_check 调用必须按签名顺序传参)"""
    cls = get_strategy_class("diff_state_demo")
    header = build_cuda_device_header(cls)
    call_args = build_cuda_strategy_check_call(cls)
    # header 里 state 部分用 "&name" 形式 (CUDA 引用), call_args 用裸名
    # 从 header 里按出现顺序抽出 state 字段名 (在 framework args 之前)
    framework_marker = "const long long &cur_ts"
    state_section = header.split(framework_marker)[0]
    # 顺序扫描 state 字段 (按 state_spec 顺序)
    state_field_order_header = []
    for name in ("counter", "touched_up", "last_level"):
        if f"&{name}" in state_section:
            state_field_order_header.append(name)
    assert call_args == ", ".join(state_field_order_header)
    assert call_args == "counter, touched_up, last_level"


def test_build_cuda_state_decls_handles_bool_default_true():
    """bool default = True → 字面量 1 (CUDA 无 bool)"""
    from evtrade.strategies.base import _STRATEGIES, register_strategy

    @register_strategy("_bool_true_default")
    class _BoolTrue(StrategyBase):
        params_spec = {}
        state_spec = {"flag": {"type": bool, "default": True}}
        def compute_signal(self, ctx):
            return 0

    try:
        decls = build_cuda_state_decls(_BoolTrue)
        assert "int flag = 1;" in decls
    finally:
        _STRATEGIES.pop("_bool_true_default", None)


def test_build_cuda_state_decls_handles_negative_int_default():
    """int default = -5 → 字面量 -5 (负数正常输出)"""
    from evtrade.strategies.base import _STRATEGIES, register_strategy

    @register_strategy("_neg_int_default")
    class _NegInt(StrategyBase):
        params_spec = {}
        state_spec = {"counter": {"type": int, "default": -5}}
        def compute_signal(self, ctx):
            return 0

    try:
        decls = build_cuda_state_decls(_NegInt)
        assert "long long counter = -5;" in decls
    finally:
        _STRATEGIES.pop("_neg_int_default", None)


def test_build_cuda_state_decls_handles_float_default():
    """float default = 0.5 → 字面量 0.5"""
    from evtrade.strategies.base import _STRATEGIES, register_strategy

    @register_strategy("_float_default")
    class _Float(StrategyBase):
        params_spec = {}
        state_spec = {"level": {"type": float, "default": 0.5}}
        def compute_signal(self, ctx):
            return 0

    try:
        decls = build_cuda_state_decls(_Float)
        assert "double level = 0.5;" in decls
    finally:
        _STRATEGIES.pop("_float_default", None)


def test_build_cuda_state_decls_handles_nan_float_default():
    """float default = NaN → __longlong_as_double bit cast (与 kernel 同口径)"""
    from evtrade.strategies.base import _STRATEGIES, register_strategy

    @register_strategy("_nan_default")
    class _Nan(StrategyBase):
        params_spec = {}
        state_spec = {"nan_field": {"type": float, "default": float("nan")}}
        def compute_signal(self, ctx):
            return 0

    try:
        decls = build_cuda_state_decls(_Nan)
        assert "double nan_field = __longlong_as_double" in decls
        assert "0x7ff8000000000000ULL" in decls
    finally:
        _STRATEGIES.pop("_nan_default", None)


# ============ 缺 state_spec 应编译期抛错 ============

def test_require_state_spec_raises_for_missing():
    """DSL 工厂函数对未声明 state_spec 的策略立即抛 CompileError"""

    class NoSpec(StrategyBase):
        # 故意不声明 state_spec (沿用 StrategyBase 默认)
        params_spec = {}
        def compute_signal(self, ctx):
            return 0

    # 实例方法都要求传入策略类; 缺 state_spec 时编译期抛错
    with pytest.raises(CompileError, match="state_spec"):
        build_ctx_to_kernel_map(NoSpec)
    with pytest.raises(CompileError, match="state_spec"):
        build_cuda_sig_fields(NoSpec)
    with pytest.raises(CompileError, match="state_spec"):
        build_cuda_device_header(NoSpec)
    with pytest.raises(CompileError, match="state_spec"):
        make_dsl_ctx(NoSpec)


def test_strategy_base_state_spec_default_empty():
    """StrategyBase.state_spec 默认为空 dict (而非 None)"""
    assert StrategyBase.state_spec == {}


# ============ 三端渲染可正常生成 ============

def test_render_numba_body_zero_state():
    """零状态策略 numba body 渲染成功"""
    cls = get_strategy_class("zero_state_demo")
    src = render_numba_body(cls)
    assert "ctx.up" in src or ".up" in src
    assert "ctx.cur_ts" in src or "st.cur_ts" in src


def test_render_cuda_body_diff_state():
    """不同状态策略 CUDA body 渲染成功 (含 state 字段引用)"""
    cls = get_strategy_class("diff_state_demo")
    src = render_cuda_body(cls)
    # state_spec 字段在 CUDA 端通过 build_ctx_to_kernel_map 投影 (identity)
    # body 里应出现这些字段 (作为局部变量或 ctx 投影名)
    assert "counter" in src
    assert "touched_up" in src
    assert "last_level" in src


# ============ numba 内核路径: build_dsl_kernel 可用 ============

def test_dsl_kernel_diff_state_compiles():
    """dsl_kernel(不同状态策略) 编译成功 + KernelState 含 state_spec 字段"""
    from evtrade.core.kernel_dsl import dsl_kernel
    kmod = dsl_kernel("diff_state_demo")
    # jitclass 内省 (numba 暴露 _numba_type_): 用 duck-type 验证字段已注入
    KS = kmod.KernelState
    # 通过构造 + 读字段验证 (numba jitclass 属性可正常访问)
    st = KS(period_seconds=300, warmup_until=0, tf1=21,
            init_cash=200000.0, init_position=0.0, trade_qty=10000.0,
            scale=1.0, buy_pct=0.0, sell_pct=0.0, all_in=False,
            record_trades=False, trade_cap=0,
            p0=2.5)
    assert st.counter == 0
    assert st.touched_up == False  # noqa: E712  (numba bool 真值)
    assert st.last_level == 0.0


def test_dsl_kernel_zero_state_compiles():
    """dsl_kernel(零状态策略) 编译成功 + KernelState 不含 state_spec 字段"""
    from evtrade.core.kernel_dsl import dsl_kernel
    kmod = dsl_kernel("zero_state_demo")
    KS = kmod.KernelState
    st = KS(period_seconds=300, warmup_until=0, tf1=21,
            init_cash=200000.0, init_position=0.0, trade_qty=10000.0,
            scale=1.0, buy_pct=0.0, sell_pct=0.0, all_in=False,
            record_trades=False, trade_cap=0,
            p0=1.0)
    # 零状态: 不应出现 channel_deviation 历史字段 (numba 也不会让访问未声明字段)
    # 用 hasattr 测试框架层不残留任何 baked-in 状态字段
    for old in ("low_hit", "high_hit", "lock_ts", "low_acted", "high_acted"):
        assert not hasattr(st, old), f"零状态 KernelState 残留字段 {old}"


# ============ 端到端: 实际跑一次 numba 内核 ============

def test_run_one_dsl_diff_state_end_to_end():
    """run_one_dsl(不同状态策略) 端到端跑通 (DSL body 写自定义字段)"""
    from evtrade.core.kernel import bars_to_arrays
    from evtrade.core.kernel_dsl import run_one_dsl
    from evtrade.data import synthetic_bars
    bars_list = synthetic_bars(days=10, start_ymd="20241101", seed=42)
    bars_arr = bars_to_arrays(bars_list)
    result = run_one_dsl(
        bars_arr, period="5m", warmup_until=0,
        init_cash=200000.0, init_position=0.0, trade_qty=10000.0,
        strategy_name="diff_state_demo",
        strategy_params={"limit": 2.5},
    )
    # 必须返回 summarize dict (有 final_price 等字段)
    assert "final_price" in result
    assert "n_trades" in result
    assert result["n_trades"] >= 0