from __future__ import annotations
"""kernel.py 读 DSL 渲染: 按策略特化的 numba 内核模块 (步骤 2)

================================================================
✅  主调度层  ✅
================================================================
机制 —— 一份内核源, N 个策略特化:
  kernel.py 的 _strategy_check 函数体位于 DSL-STRATEGY-BEGIN/END 标记之间
  (空模板, 占位 pass)。build_dsl_kernel(name) 把该策略的 DSL 经
  strategies.dsl.render_numba_state_body 渲染成 st 代码, 整段替换两标记之间的
  内容, exec 出一个独立内核模块 (jitclass / step / run_backtest / summarize
  全套, numba 首次调用时编译)。

  dsl_kernel(name) = build_dsl_kernel(name), **所有策略同路径**
  (2026-09 重构后已无策略特殊路径)。

语义保证 (三端同源):
  特化模块 = kernel.py 源码 + DSL 渲染产物, 表达式字面顺序与 DSL docstring 一致;
  浮点路径与参考引擎一致 (差分测试锁定)。
  tests/test_dsl_spliced_channel_deviation_bitwise 等价锁定:
    特化模块 = Python ref 引擎 (ChannelDeviationStrategy.check)。

ctx 字段契约 (DSL -> 内核, 唯一事实源在 strategies/dsl.py::build_ctx_to_kernel_map):
  ctx.p0..p15        -> st.p0..st.p15    (按策略 params_spec 声明顺序)
  ctx.cur_ts/high/low/close/open/volume -> st.cur_*
  ctx.up / ctx.dw    -> _strategy_check 的函数参数 (裸名)
  state_spec 字段   -> st.<name>          (单名空间; 策略类声明, 框架层无 baked-in 字段)
  其余 ctx.x / 裸名 x -> 函数局部变量 (首次赋值前不可读)

splice 注入 (Phase 2 Commit 3):
  本模块同时替换两段:
    1. kernel._STATE_SPEC + KernelState 类 (KERNEL-STATE-CLASS 区间): 按策略 state_spec
       注入对应 jitclass 字段 (__init__ 末尾按 default 初始化); numba 首次调用
       时按新 spec 重建 jitclass, 后续走 cache。
    2. kernel._strategy_check 函数体 (DSL-STRATEGY 区间): DSL body 渲染产物。

GPU 对应端: gpu.py::_CUDA_SOURCE_GENERIC_TEMPLATE + render_cuda_device_function,
由 cuda_sweep_window_generic(..., strategy_name=...) 使用。
================================================================
"""
import hashlib
import inspect
import sys
import textwrap
import threading
import types

import numpy as np

# kernel.py 中整段替换的标记前缀 (与 kernel._strategy_check 内的注释一致)
_SPLICE_BEGIN = "# ==== DSL-STRATEGY-BEGIN"
_SPLICE_END = "# ==== DSL-STRATEGY-END"

# kernel.py 中 KernelState jitclass 块的整段替换标记 (按 strategy_name 注入
# state_spec 字段); 与 _SPLICE_BEGIN/END 配对使用。
_STATE_CLASS_BEGIN = "# ==== KERNEL-STATE-CLASS-BEGIN"
_STATE_CLASS_END = "# ==== KERNEL-STATE-CLASS-END"

_EMPTY_SIG = np.empty(0, np.int8)
_EMPTY_F = np.empty(0, np.float64)

# 渲染器版本: 任一变更 (render_numba_state_body / 字段映射 / 拼接逻辑)
# 都应 bump 此版本号, 让旧缓存产物失效。
RENDERER_VERSION = "v1"


def _source_hash(strategy_cls) -> str:
    """compute_signal.__doc__ 的 sha1 (用于 cache key, 让 DSL 修改能失效缓存)"""
    method = getattr(strategy_cls, "compute_signal", None)
    doc = getattr(method, "__doc__", None) or ""
    return hashlib.sha1(doc.encode("utf-8")).hexdigest()


def strategy_has_dsl(strategy_name: str) -> bool:
    """策略是否带可渲染的 DSL compute_signal docstring (未知策略抛 ValueError)

    仅做编译期探针: 取类不实例化, 避免触发策略 __init__ 副作用
    内的 make_python_runner/exec。
    """
    from ..strategies import get_strategy_class
    from ..strategies.dsl import CompileError, render_numba_body
    try:
        render_numba_body(get_strategy_class(strategy_name))
        return True
    except CompileError:
        return False


# 缓存: key=(strategy_name, source_hash, RENDERER_VERSION), value=module
# 用 dict 替换 functools.lru_cache:
#   - 三元 key 支持 DSL 改动 / 渲染器版本升级时正确失效缓存
#   - 可暴露 invalidate_dsl_cache() 给测试与 dev reload
#   - 避免 lru_cache 在并发首 miss 时多次重复 splice+exec 竞态 sys.modules
_KERNEL_DSL_CACHE: dict = {}


def _invalidate_cache(cache: dict, strategy_name: str | None = None) -> int:
    """通用缓存清除: 按 strategy_name 过滤 keys 并 pop, None 时清空全部。

    kernel_dsl / gpu 的缓存失效共用此实现 (两处原先各写一份相同逻辑)。
    返回清除的条目数。
    """
    if strategy_name is None:
        n = len(cache)
        cache.clear()
        return n
    keys = [k for k in cache if k[0] == strategy_name]
    for k in keys:
        cache.pop(k, None)
    return len(keys)


def invalidate_dsl_cache(strategy_name: str | None = None) -> int:
    """清除 DSL 内核缓存; strategy_name=None 时清空全部

    返回清除的条目数, 方便测试断言与日志。
    """
    return _invalidate_cache(_KERNEL_DSL_CACHE, strategy_name)


def _build_dsl_kernel_impl(strategy_name: str, source_hash: str):
    """实际 splice + exec; 由 build_dsl_kernel 持有单飞锁调用"""
    from . import kernel
    from ..strategies import get_strategy_class, get_strategy_state_spec
    from ..strategies.dsl import render_numba_state_body

    cls = get_strategy_class(strategy_name)
    state_spec = get_strategy_state_spec(strategy_name)

    # 1) 注入 DSL 策略段 (DSL body 渲染产物 -> _strategy_check 函数体)
    body = render_numba_state_body(cls).strip("\n")
    src = inspect.getsource(kernel)
    i0 = src.index(_SPLICE_BEGIN)
    i1 = src.index(_SPLICE_END)
    i0 = src.rindex("\n", 0, i0) + 1                    # BEGIN 注释行行首
    i1 = src.index("\n", i1) + 1                        # END 注释行行尾之后
    # 注入白名单 indicator 函数 import (DSL body 调 ema_channel_push 等;
    # 详见 strategies/dsl._INDICATOR_INJECT_NUMBA); 必须放在模块顶层 (import
    # 不得嵌进函数体, 否则 col 0 行落在 def docstring 后会触发 IndentationError)。
    # 做法: 找到模块第一个 `@njit` 装饰行, 把 imports 插到它之前 (模块级, 在
    # docstring + 其它 imports 之后, 在 numba 装饰函数之前)。
    from ..strategies.dsl import _INDICATOR_INJECT_NUMBA
    inject = _INDICATOR_INJECT_NUMBA + "\n"
    # 找第一个以 @njit 开头的行
    lines = src.split("\n")
    first_njit = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("@njit"):
            first_njit = idx
            break
    if first_njit is None:
        raise RuntimeError("kernel.py 中找不到 @njit 装饰行; 无法注入 indicator imports")
    inject_pos = sum(len(l) + 1 for l in lines[:first_njit])
    src = src[:inject_pos] + inject + src[inject_pos:]
    # 然后替换 BEGIN/END 之间为带缩进的 DSL body
    # BEGIN/END 位置在注入 imports 后整体下移, 需要重新计算
    i0 = src.index(_SPLICE_BEGIN)
    i1 = src.index(_SPLICE_END)
    i0 = src.rindex("\n", 0, i0) + 1
    i1 = src.index("\n", i1) + 1
    src = src[:i0] + textwrap.indent(body, "    ") + "\n" + src[i1:]

    # 2) 注入 KernelState jitclass (按 state_spec 添加字段)
    state_src = kernel._build_kernel_state_source(strategy_name)
    j0 = src.index(_STATE_CLASS_BEGIN)
    j1 = src.index(_STATE_CLASS_END)
    j0 = src.rindex("\n", 0, j0) + 1
    j1 = src.index("\n", j1) + 1
    src = src[:j0] + textwrap.indent(state_src, "") + src[j1:]

    # exec 源没有真实文件可作 numba 磁盘缓存键 -> 去掉 cache=True
    src = src.replace("@njit(cache=True)", "@njit()")

    mod_name = f"evtrade.core._kernel_dsl[{strategy_name}:{source_hash[:8]}:{RENDERER_VERSION}]"
    mod = types.ModuleType(mod_name)
    mod.__file__ = kernel.__file__
    mod.__package__ = "evtrade.core"     # 供 kernel.resolve_period_seconds 的相对导入
    # numba jitclass 注册时会按 __module__ 回查 sys.modules, 必须先挂进去
    sys.modules[mod.__name__] = mod
    exec(compile(src, f"<kernel_dsl:{strategy_name}>", "exec"), mod.__dict__)
    # 注入 make_state 便捷 shim: 直接调本模块 KernelState jitclass (与历史
    # kernel.py::make_state 等价)。保留供 _KMOD.make_state 测试 / 旧脚本;
    # 新代码请用 make_state_general(..., strategy_params=...)。
    KS = mod.KernelState
    # state_spec 字段默认: 不传 (走 __init__ 内 hardcoded default)
    def make_state(period="5m", warmup_until=0, tf1=21,
                   init_cash=200000.0, init_position=200000.0,
                   trade_qty=10000.0, scale=1.0,
                   buy_pct=0.0, sell_pct=0.0, all_in=False,
                   record_trades=False, trade_cap=0,
                   p0=0.0, p1=0.0, p2=0.0, p3=0.0,
                   p4=0.0, p5=0.0, p6=0.0, p7=0.0,
                   p8=0.0, p9=0.0, p10=0.0, p11=0.0,
                   p12=0.0, p13=0.0, p14=0.0, p15=0.0):
        from .kernel import resolve_period_seconds
        if all_in:
            buy_pct = max(buy_pct, 1.0)
            sell_pct = max(sell_pct, 1.0)
        return KS(resolve_period_seconds(period), warmup_until, tf1,
                  init_cash, init_position, trade_qty, scale,
                  float(buy_pct), float(sell_pct), bool(all_in),
                  bool(record_trades), int(trade_cap),
                  float(p0), float(p1), float(p2), float(p3),
                  float(p4), float(p5), float(p6), float(p7),
                  float(p8), float(p9), float(p10), float(p11),
                  float(p12), float(p13), float(p14), float(p15))
    mod.make_state = make_state
    return mod


# 单飞锁: 并发首 miss 时, 同一 key 只有第一个线程进 _build_dsl_kernel_impl,
# 其它线程拿到它的结果, 避免重复 splice+exec 与 sys.modules 竞态。
_BUILD_LOCK = threading.RLock()


def build_dsl_kernel(strategy_name: str):
    """DSL -> 该策略专用的 numba 内核模块 (splice kernel.py 策略段后 exec)

    缓存键: (strategy_name, source_hash, RENDERER_VERSION)。
    任一变更 (DSL docstring 修改 / 渲染器逻辑升级) 会自动失效旧缓存。
    """
    from ..strategies import get_strategy_class
    cls = get_strategy_class(strategy_name)
    src_hash = _source_hash(cls)
    key = (strategy_name, src_hash, RENDERER_VERSION)
    mod = _KERNEL_DSL_CACHE.get(key)
    if mod is not None:
        return mod
    # 单飞: 同 key 的并发首 miss 串行化
    with _BUILD_LOCK:
        mod = _KERNEL_DSL_CACHE.get(key)
        if mod is not None:
            return mod
        mod = _build_dsl_kernel_impl(strategy_name, src_hash)
        _KERNEL_DSL_CACHE[key] = mod
        return mod


def dsl_kernel(strategy_name: str):
    """策略 -> numba 内核模块 (所有策略同路径, 经 build_dsl_kernel 渲染 + 缓存)

    等价于 build_dsl_kernel(strategy_name), 保留此入口供向后兼容 (历史 API
    即 dsl_kernel, 而不是 build_dsl_kernel)。
    """
    return build_dsl_kernel(strategy_name)


def make_state_general(strategy_name: str, period: str, warmup_until: int,
                       tf1: int = 21, init_cash: float = 200000.0,
                       init_position: float = 200000.0,
                       trade_qty: float = 10000.0, scale: float = 1.0,
                       buy_pct: float = 0.0, sell_pct: float = 0.0,
                       all_in: bool = False, strategy_params: dict | None = None,
                       record_trades: bool = False, trade_cap: int = 0):
    """按 params_spec 声明顺序把策略参数填进内核 p0..pN 并构造状态

    tf1 (通道 EMA 周期) 已下沉为策略 params (channel_deviation 等的 params_spec 自声明
    tf1); 若调用方同时传 `tf1` 关键字和 `strategy_params`, 且 strategy_params 含 tf1,
    则 strategy_params 优先 (显式覆盖); 否则用本函数的 tf1 兜底填进 p4。
    参数 > 16 个 -> ValueError (内核 p0..p15 上限)。
    """
    from ..strategies import get_strategy_param_spec
    spec = get_strategy_param_spec(strategy_name)
    if len(spec) > 16:
        raise ValueError(
            f"策略 {strategy_name} 有 {len(spec)} 个参数, 超过内核上限 16 (p0..p15)")
    sp = dict(strategy_params or {})
    # tf1 兜底: 若 strategy_params 没声明 tf1, 用本函数 tf1 入参数兜底 (历史 tf1 是
    # 引擎级关键字, 仍允许直接传 — 不显式声明就回落到函数入参, 避免 silently 落到
    # params_spec 的 default 21, 造成指标周期与调用方预期不符)。
    if "tf1" in spec and "tf1" not in sp:
        sp["tf1"] = tf1
    pvals = [float(sp.get(k, schema.get("default", 0.0)))
             for k, schema in spec.items()]
    return dsl_kernel(strategy_name).make_state(
        period=period, warmup_until=warmup_until, tf1=int(tf1),
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
                strategy_name: str | None = None,
                strategy_params: dict | None = None) -> dict:
    """任意 DSL 策略单窗回测 (numba 内核; 指标口径 = kernel.summarize)

    strategy_name: 必填 (策略 key)。sweep 对所有 DSL 策略走这里 (历史曾有
    特定策略专用的 run_one shim, 已并入 sweep.run_one_from_dict 统一入口)。
    """
    if not strategy_name:
        raise ValueError("run_one_dsl: strategy_name is required")
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
