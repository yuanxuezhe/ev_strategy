from __future__ import annotations
"""kernel.py 读 DSL 渲染: 按策略特化的 numba 内核模块 (步骤 2)

================================================================
✅  主调度层新增模块 (不改动冻结文件的语义)  ✅
================================================================
机制 —— 一份内核源, N 个策略特化:
  kernel.py 的 _strategy_check 函数体位于 DSL-STRATEGY-BEGIN/END 标记之间,
  是 channel_deviation DSL 的手写 st 形式 (与 frozen/strategy.py 逐位锁定)。
  build_dsl_kernel(name) 把该策略的 DSL 经 strategies.dsl.render_numba_state_body
  渲染成同形 st 代码, 整段替换两标记之间的内容, exec 出一个独立内核模块
  (jitclass / step / run_backtest / summarize 全套, numba 首次调用时编译)。

  dsl_kernel(name):
    - channel_deviation -> 冻结 kernel 本尊 (零额外编译; 72 项差分测试不受影响)
    - 其他 DSL 策略     -> build_dsl_kernel 的缓存产物

语义保证 (三端同源):
  特化模块与冻结 kernel 的唯一差异是 _strategy_check 函数体, 其余逐字节相同;
  DSL 渲染保持表达式字面顺序, 浮点路径与参考引擎一致。
  tests/test_kernel_dsl.py: 特化(channel_deviation) 与冻结 kernel bitwise 一致。

ctx 字段契约 (DSL -> 内核, 唯一事实源在 strategies/dsl.py::_CTX_TO_KERNEL):
  ctx.p0..p15        -> st.p0..st.p15    (按策略 params_spec 声明顺序)
  ctx.cur_ts/high/low/close/open/volume -> st.cur_*
  ctx.up / ctx.dw    -> _strategy_check 的函数参数 (裸名)
  ctx.low_hit / high_hit        -> st.low_hit / st.high_hit
  ctx._low_acted / _high_acted  -> st.low_acted / st.high_acted
  ctx._bucket_ts     -> st.lock_ts
  其余 ctx.x / 裸名 x -> 函数局部变量 (首次赋值前不可读)

GPU 对应端: gpu.py::_CUDA_SOURCE_GENERIC_TEMPLATE + render_cuda_device_function,
由 cuda_sweep_window_generic(..., strategy_name=...) 使用。
================================================================
"""
import functools
import inspect
import sys
import textwrap
import types

import numpy as np

# kernel.py 中整段替换的标记前缀 (与 kernel._strategy_check 内的注释一致)
_SPLICE_BEGIN = "# ==== DSL-STRATEGY-BEGIN"
_SPLICE_END = "# ==== DSL-STRATEGY-END"

_EMPTY_SIG = np.empty(0, np.int8)
_EMPTY_F = np.empty(0, np.float64)


def strategy_has_dsl(strategy_name: str) -> bool:
    """策略是否带可渲染的 DSL compute_signal docstring (未知策略抛 ValueError)"""
    from ..strategies import get_strategy
    from ..strategies.dsl import CompileError, render_numba_body
    try:
        render_numba_body(get_strategy(strategy_name))
        return True
    except CompileError:
        return False


@functools.lru_cache(maxsize=None)
def build_dsl_kernel(strategy_name: str):
    """DSL -> 该策略专用的 numba 内核模块 (splice kernel.py 策略段后 exec)

    与冻结 kernel 的唯一差异: _strategy_check 函数体来自该策略 DSL 渲染。
    本函数只做字符串拼接 + exec (numba 编译惰性, 首次回测时触发);
    同策略进程内只构建一次 (lru_cache)。
    """
    from . import kernel
    from ..strategies import get_strategy
    from ..strategies.dsl import render_numba_state_body

    body = render_numba_state_body(get_strategy(strategy_name)).strip("\n")
    src = inspect.getsource(kernel)
    i0 = src.index(_SPLICE_BEGIN)
    i1 = src.index(_SPLICE_END)
    i0 = src.rindex("\n", 0, i0) + 1                    # BEGIN 注释行行首
    i1 = src.index("\n", i1) + 1                        # END 注释行行尾之后
    src = src[:i0] + textwrap.indent(body, "    ") + "\n" + src[i1:]
    # exec 源没有真实文件可作 numba 磁盘缓存键 -> 去掉 cache=True
    src = src.replace("@njit(cache=True)", "@njit()")

    mod = types.ModuleType(f"evtrade.core._kernel_dsl[{strategy_name}]")
    mod.__file__ = kernel.__file__
    mod.__package__ = "evtrade.core"     # 供 kernel.resolve_period_seconds 的相对导入
    # numba jitclass 注册时会按 __module__ 回查 sys.modules, 必须先挂进去
    sys.modules[mod.__name__] = mod
    exec(compile(src, f"<kernel_dsl:{strategy_name}>", "exec"), mod.__dict__)
    return mod


@functools.lru_cache(maxsize=None)
def dsl_kernel(strategy_name: str):
    """策略 -> numba 内核模块; channel_deviation 返回冻结 kernel 本尊"""
    if strategy_name == "channel_deviation":
        from . import kernel
        return kernel
    return build_dsl_kernel(strategy_name)


def make_state_general(strategy_name: str, period: str, warmup_until: int,
                       tf1: int = 21, init_cash: float = 200000.0,
                       init_position: float = 200000.0,
                       trade_qty: float = 10000.0, scale: float = 1.0,
                       buy_pct: float = 0.0, sell_pct: float = 0.0,
                       all_in: bool = False, strategy_params: dict | None = None,
                       record_trades: bool = False, trade_cap: int = 0):
    """按 params_spec 声明顺序把策略参数填进内核 p0..pN 并构造状态

    tf1 (通道 EMA 周期) 是引擎级参数, 不属于 params_spec。
    参数 > 16 个 -> ValueError (内核 p0..p15 上限)。
    """
    from ..strategies import get_strategy_param_spec
    spec = get_strategy_param_spec(strategy_name)
    if len(spec) > 16:
        raise ValueError(
            f"策略 {strategy_name} 有 {len(spec)} 个参数, 超过内核上限 16 (p0..p15)")
    sp = dict(strategy_params or {})
    pvals = [float(sp.get(k, schema.get("default", 0.0)))
             for k, schema in spec.items()]
    return dsl_kernel(strategy_name).make_state(
        period=period, warmup_until=warmup_until, tf1=int(tf1),
        low1=0.0, low2=0.0, high1=0.0, high2=0.0,
        init_cash=init_cash, init_position=init_position,
        trade_qty=trade_qty, scale=scale,
        buy_pct=buy_pct, sell_pct=sell_pct, all_in=all_in,
        record_trades=record_trades, trade_cap=trade_cap,
        **{f"p{i}": v for i, v in enumerate(pvals)})


def run_one_dsl(bars: dict, period: str, warmup_until: int,
                init_cash: float, init_position: float, trade_qty: float,
                tf1: int = 21, scale: float = 1.0,
                buy_pct: float = 0.0, sell_pct: float = 0.0,
                all_in: bool = False,
                strategy_name: str = "channel_deviation",
                strategy_params: dict | None = None) -> dict:
    """任意 DSL 策略单窗回测 (numba 内核; 指标口径 = kernel.summarize)

    sweep 对非 channel_deviation 的 DSL 策略走这里 (run_one 的通用版)。
    """
    st = make_state_general(strategy_name, period, warmup_until, tf1=tf1,
                            init_cash=init_cash, init_position=init_position,
                            trade_qty=trade_qty, scale=scale,
                            buy_pct=buy_pct, sell_pct=sell_pct, all_in=all_in,
                            strategy_params=strategy_params)
    kmod = dsl_kernel(strategy_name)
    kmod.run_backtest(st, bars["stime"], bars["open"], bars["high"],
                      bars["low"], bars["close"], bars["volume"],
                      _EMPTY_SIG, _EMPTY_F, _EMPTY_F)
    return kmod.summarize(st)
