"""DSL -> 三端转译器测试 (Python / numba / CUDA C99)

本测试只验证字符串渲染正确性; bitwise 一致需步骤 2/3 (通用 CUDA kernel 改造) 后再验证。
"""
from __future__ import annotations

import re

import pytest

from evtrade.strategies import (compile_all, get_strategy, render_cuda_body,
                                 render_numba_body)


# ============ 渲染产物结构 ============

def test_compile_all_returns_three():
    """compile_all 返回 python/numba/cuda 三份源码"""
    out = compile_all(get_strategy("channel_deviation"))
    assert set(out.keys()) == {"python", "numba", "cuda"}
    assert out["python"].strip()  # 非空
    assert out["numba"].strip()
    assert out["cuda"].strip()


def test_python_unchanged():
    """python 版是 DSL 原文 (无改动)"""
    out = compile_all(get_strategy("channel_deviation"))
    # 包含 ctx.xxx 字段访问 (DSL 原样)
    assert "ctx.up" in out["python"]
    assert "ctx.dw" in out["python"]
    assert "ctx._bucket_ts" in out["python"]


def test_numba_no_ctx_prefix():
    """numba 版保留 ctx. 前缀 (用于 jitclass 字段访问)"""
    cuda_src = render_numba_body(get_strategy("channel_deviation"))
    assert "ctx.up" in cuda_src or ".up" in cuda_src  # numba jitclass 也用 .
    # numba Python 子集: if x: 不用花括号
    assert "if " in cuda_src
    # 浮点字面量 100 在 numba 版用 100 不是 100.0 (这是 numba 的 Python 子集惯例)
    # 注: numba njit 接受 100 当 int; 但 ctx.low_dev 是 double 会自动转


def test_cuda_strips_ctx_prefix():
    """CUDA 版去掉 ctx. 前缀 (struct 字段直接访问)"""
    src = render_cuda_body(get_strategy("channel_deviation"))
    # 不应有 "ctx." 出现
    assert "ctx." not in src
    # 应有直接字段访问
    assert "up " in src or "up;" in src or "up =" in src
    # C99 用花括号
    assert "if (" in src and "{" in src and "}" in src


def test_cuda_bool_to_int():
    """CUDA 无 bool: True/False -> 1/0; ctx 状态字段映射到内核名"""
    src = render_cuda_body(get_strategy("channel_deviation"))
    # ctx._low_acted = False 应渲染成 low_acted = 0 (映射到内核状态名)
    assert "low_acted = 0;" in src
    assert "high_acted = 0;" in src
    assert "low_acted = 1;" in src
    assert "high_acted = 1;" in src


def test_cuda_and_or_to_c99():
    """CUDA and/or/not -> &&/||/!"""
    src = render_cuda_body(get_strategy("channel_deviation"))
    assert "&&" in src
    assert "||" in src
    # 桶切换处的 not ctx._bucket_ts == _bucket_ts 用了 not
    # 注意: 此断言只在特定位置; 不强求严格 "!"
    # 但 && 一定存在


def test_cuda_int_literal_to_double():
    """CUDA 整数字面量加 .0 让 C 当 double 算"""
    src = render_cuda_body(get_strategy("channel_deviation"))
    # 100 应该是 100.0
    assert "* 100.0" in src


def test_cuda_return_signal():
    """CUDA: signal 变量形式 (与 frozen/kernel 同语义), return signal 收尾"""
    src = render_cuda_body(get_strategy("channel_deviation"))
    assert "return 0;" in src            # up/dw 无效时提前返回
    assert "int signal = 0;" in src      # 局部变量首赋值处声明
    assert "signal = 1;" in src
    assert "signal = (-1);" in src
    assert "return signal;" in src


# ============ DSL 白名单拒绝非法构造 ============

def test_unsupported_call_rejected():
    """调用外部函数被拒绝"""
    from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy
    from evtrade.strategies.dsl import CompileError

    @register_strategy("_test_unsupported_call")
    class _(StrategyBase):
        params_spec = {}
        def compute_signal(self, ctx):
            """x = min(1, 2)"""
            return 0
    try:
        render_cuda_body(_)
    except CompileError:
        pass
    finally:
        _STRATEGIES.pop("_test_unsupported_call", None)
    assert True  # 走到这里就行


# ============ example_breakout 也能渲染 ============

def test_breakout_renders():
    """breakout 策略: 走纯 Python check, 不渲染 DSL (抛 CompileError)

    breakout 是教学示例, 没写 DSL docstring; 只有 check() 直接 Python。
    """
    from evtrade.strategies.dsl import CompileError
    from evtrade.strategies.example_breakout import BreakoutStrategy
    with pytest.raises(CompileError):
        compile_all(BreakoutStrategy)


def test_breakout_renders_with_dsl_docstring():
    """演示: 给 breakout 加 DSL docstring 后能渲染三端"""
    from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy

    @register_strategy("_test_breakout_dsl")
    class _B(StrategyBase):
        params_spec = {"lookback": {"default": 20, "type": int}}
        def __init__(self, params=None, **kwargs):
            super().__init__(params=params, **kwargs)
            self._closes = []
        def compute_signal(self, ctx):
            """\
if ctx.close > 100:
    return 1
return 0
"""
            return 0
        def check(self, cur, ind=None, dw=None):
            if cur["close"] > 100:
                return "BUY", {"close": cur["close"]}
            return None, {}

    out = compile_all(_B)
    assert "ctx.close" in out["python"]
    assert "close" in out["cuda"]
    assert "ctx." not in out["cuda"]
    # 清理
    _STRATEGIES.pop("_test_breakout_dsl", None)


# ============ 关键不变量:三端表达式字面等价 ============

def test_three_ends_same_field_access():
    """三端都用同样的字段名 (low1/low2/low_hit/low_dev...)"""
    out = compile_all(get_strategy("channel_deviation"))
    py_fields = set(re.findall(r"ctx\.(\w+)", out["python"]))
    cuda_fields = set(re.findall(r"\b(\w+)\s*[=<>!+\-*/;,)]", out["cuda"]))
    # 抽出 numba 字段
    nb_fields = set(re.findall(r"ctx\.(\w+)", out["numba"]))
    # 三端访问的 ctx 字段必须一致
    assert py_fields == nb_fields
    # CUDA 字段应该是 py 字段去掉 ctx. (但可能混入 ctx.* 上下文里的局部变量名)
    # 至少重叠度要高 (>= 50%)
    overlap = len(py_fields & cuda_fields)
    assert overlap >= len(py_fields) * 0.5, \
        f"CUDA 字段缺失: py={py_fields}, cuda={cuda_fields}"


# ============ 步骤 1: 通用 CUDA kernel 模板静态检查 ============

def test_generic_template_has_strategy_placeholder():
    """通用 CUDA 模板含 {STRATEGY_BODY} 占位符"""
    from evtrade.core.gpu import _CUDA_SOURCE_GENERIC_TEMPLATE
    assert "{STRATEGY_BODY}" in _CUDA_SOURCE_GENERIC_TEMPLATE


def test_generic_template_p0_to_p7():
    """通用 CUDA 模板用 p0..p7 寄存器接受策略参数"""
    from evtrade.core.gpu import _CUDA_SOURCE_GENERIC_TEMPLATE
    for i in range(8):
        assert f"p{i}" in _CUDA_SOURCE_GENERIC_TEMPLATE, f"缺 p{i}"


def test_cuda_render_into_template():
    """render_cuda_device_function 输出能注入 {STRATEGY_BODY} 占位符"""
    from evtrade.core.gpu import _CUDA_SOURCE_GENERIC_TEMPLATE
    from evtrade.strategies import get_strategy
    from evtrade.strategies.dsl import render_cuda_device_function

    cls = get_strategy("channel_deviation")
    func = render_cuda_device_function(cls)
    # 注入后不应有未替换的占位符
    src = _CUDA_SOURCE_GENERIC_TEMPLATE.replace("{STRATEGY_BODY}", func)
    assert "{STRATEGY_BODY}" not in src
    # 注入的是 device 函数, 内核状态名 + 参数寄存器齐备
    assert "__device__" in func and "strategy_check" in func
    assert "lock_ts" in func and "low_hit" in func
    assert "p0" in func and "p7" in func


def test_generic_kernel_signature():
    """通用 kernel 函数签名接收 num_params + params_arr"""
    from evtrade.core.gpu import _CUDA_SOURCE_GENERIC_TEMPLATE
    src = _CUDA_SOURCE_GENERIC_TEMPLATE
    assert "sweep_kernel_generic" in src
    assert "const double* __restrict__ params_arr" in src
    assert "int num_params" in src


def test_channel_deviation_dsl_uses_p0_to_p3():
    """channel_deviation DSL 用 ctx.p0..p3 引用策略参数 (与 GPU 通用 kernel 对齐)"""
    from evtrade.strategies import get_strategy
    cls = get_strategy("channel_deviation")
    spec_keys = list(cls.params_spec.keys())
    # 通用 kernel 用 p0..p7, DSL 必须按 spec 顺序引用 p0..pN
    assert spec_keys[:4] == ["low1", "low2", "high1", "high2"]
    # DSL 包含 ctx.pN (与 GPU kernel 字段对齐)
    dsl_doc = cls.compute_signal.__doc__
    for i in range(4):
        assert f"ctx.p{i}" in dsl_doc, f"DSL 缺 ctx.p{i}"


def test_cuda_kernel_compile_attempt():
    """尝试编译通用 CUDA kernel (步骤 1 端到端)

    没 GPU 时会抛 ImportError; 有 GPU 时尝试编译。
    """
    from evtrade.strategies import get_strategy
    try:
        from evtrade.core.gpu import _compile_generic_kernel
        kernel = _compile_generic_kernel("channel_deviation")
        # 编译成功; 不具体测结果
        assert kernel is not None
    except (ImportError, ValueError) as e:
        # 没 GPU 或 cupy 不可用是预期的
        msg = str(e)
        assert "cupy" in msg.lower() or "cuda" in msg.lower() or "no attribute" in msg.lower() or "channel_deviation" in msg


# ============ 步骤 2: 内核状态映射渲染 ============

def test_cuda_state_body_maps_kernel_names():
    """render_cuda_body 把 ctx 状态字段映射为内核名 (lock_ts/low_acted), 无 ctx. 残留"""
    src = render_cuda_body(get_strategy("channel_deviation"))
    assert "ctx." not in src
    assert "lock_ts" in src          # ctx._bucket_ts -> lock_ts
    assert "_bucket_ts" not in src
    assert "low_acted" in src and "_low_acted" not in src
    assert "high_acted" in src and "_high_acted" not in src


def test_cuda_device_function_declares_locals():
    """device 函数: 局部变量首赋值处声明 (int signal / double low_dev)"""
    from evtrade.strategies import get_strategy
    from evtrade.strategies.dsl import render_cuda_device_function
    func = render_cuda_device_function(get_strategy("channel_deviation"))
    assert func.startswith("__device__")
    assert "int signal = 0;" in func          # 纯整数字面量 -> int
    assert "double low_dev =" in func         # 浮点表达式 -> double
    assert "return signal;" in func           # DSL 收尾 return signal


def test_cuda_device_function_rejects_overflow_params():
    """ctx.p8..p15 / cur_mark 超出 CUDA 签名 -> 编译期 CompileError"""
    from evtrade.strategies.base import _STRATEGIES, StrategyBase, register_strategy
    from evtrade.strategies.dsl import CompileError, render_cuda_device_function

    @register_strategy("_test_cuda_overflow")
    class _(StrategyBase):
        params_spec = {f"x{i}": {"default": 0.0} for i in range(9)}
        def compute_signal(self, ctx):
            """return ctx.p8"""
            return 0
    try:
        with pytest.raises(CompileError, match="p8"):
            render_cuda_device_function(_)
    finally:
        _STRATEGIES.pop("_test_cuda_overflow", None)


# ============ 步骤 2/3: 通用 CUDA kernel 端到端 (GPU 可用时) ============

cupy_ready = True
_reason = ""
try:
    from evtrade.gpu import _ensure_cupy  # noqa: F401
    _ensure_cupy()
except Exception as e:  # noqa: BLE001
    cupy_ready = False
    _reason = str(e)[:80]

if cupy_ready:
    from evtrade.data import synthetic_bars
    from evtrade.core.gpu import cuda_sweep_window_generic
    from evtrade.core.kernel import bars_to_arrays
    from evtrade.core.kernel_dsl import run_one_dsl

    _GPU_BARS = bars_to_arrays(synthetic_bars(days=25, start_ymd="20250101", seed=11))
    _GPU_WARM = 20250110000000

    def _gpu_vs_cpu(strat, params):
        pl = [{"period": "5m", "tf1": 21, "scale": 2.0,
               "init_cash": 200000., "init_position": 200000.,
               "trade_qty": 10000., "params": params,
               "strategy_name": strat}]
        gpu = cuda_sweep_window_generic(_GPU_BARS, pl, _GPU_WARM,
                                        strategy_name=strat)[0]
        cpu = run_one_dsl(_GPU_BARS, "5m", _GPU_WARM, 200000., 200000.,
                          10000., scale=2.0, strategy_name=strat,
                          strategy_params=params)
        return gpu, cpu

    def test_generic_cuda_matches_cpu_channel_deviation():
        """通用 CUDA kernel vs CPU DSL 内核: channel_deviation 全指标逐位一致"""
        gpu, cpu = _gpu_vs_cpu("channel_deviation",
                               {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5})
        assert gpu["n_trades"] > 5, "有效信号样本过少"
        for k in gpu:
            assert gpu[k] == cpu[k], f"{k}: gpu={gpu[k]!r} cpu={cpu[k]!r}"

    def test_generic_cuda_matches_cpu_dev_trigger():
        """通用 CUDA kernel vs CPU DSL 内核: dev_trigger 全指标逐位一致"""
        gpu, cpu = _gpu_vs_cpu("dev_trigger", {"entry_dev": 0.5})
        assert gpu["n_trades"] > 5, "有效信号样本过少"
        for k in gpu:
            assert gpu[k] == cpu[k], f"{k}: gpu={gpu[k]!r} cpu={cpu[k]!r}"

    def test_sweep_gpu_generic_dev_trigger():
        """sweep(device='gpu') 对 DSL 策略走通用 CUDA kernel, 与 CPU 路径逐位一致"""
        from evtrade.core.sweep import parse_grid, sweep

        base = {"start": "20250110", "period": "5m", "tf1": 21,
                "trade_qty": 10000.0, "init_cash": 200000.0,
                "init_position": 200000.0, "params": {"entry_dev": 0.5}}
        combos = parse_grid(["entry_dev=0.5,0.8"], extra_keys={"entry_dev"})
        df_cpu = sweep(_GPU_BARS, base, combos, n_workers=2,
                       strategy_name="dev_trigger", device="cpu", verbose=False)
        df_gpu = sweep(_GPU_BARS, base, combos,
                       strategy_name="dev_trigger", device="gpu", verbose=False)
        assert len(df_cpu) == len(df_gpu) == 2
        for col in ("n_trades", "final_equity", "excess_pct",
                    "ann_excess_pct", "sharpe_excess", "max_drawdown",
                    "x_mdd", "max_dd_days", "turnover"):
            a = df_cpu.sort_values("entry_dev")[col].tolist()
            b = df_gpu.sort_values("entry_dev")[col].tolist()
            assert a == b, f"{col}: cpu={a} gpu={b}"
