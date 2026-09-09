from __future__ import annotations
"""ChannelDeviationStrategy: 通道偏离回撤策略 (DSL 版; 指标自维护)

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
DSL 版 (走参考引擎 + numba kernel + CUDA, 三端同源)。
唯一 ChannelDeviationStrategy; frozen/strategy.py 已删除 (2026-09 重构)。

参数通过 params_spec + params dict 传递, 不再靠固定 __init__ 关键字。

2026-09-09: framework 不再持有任何指标字段 (详见 change
`2026-09-09-decouple-indicators-from-framework`); EMA 通道增量状态
由策略 state_spec 自声明 (up_st/dw_st), DSL body 第一段调
`evtrade.indicators.ema_channel_push` / `ema_channel_current` 自维护。
三端 (Python/numba/CUDA) 浮点路径与原 _ema_push / _ema_current 逐位一致
(测试锁定)。
"""
from .base import StrategyBase, register_strategy


# DSL 主逻辑 (白名单: 算术/比较/布尔/标量赋值/return + 白名单 indicator 调用)
# EMA 通道状态的"推入"由 framework 在桶切换时统一处理 (kernel.step() 在
# 桶切换时调 _ema_push 把旧桶的 high/low 写入 state_spec 字段 up_st_* /
# dw_st_*, 与原 _sync_ema 等价); DSL body 这里只"读" — ema_current 从
# 6 标量 state 取当前轨 (未就绪返回 0.0)。这保证三端 (Python/numba/CUDA)
# 推入时机严格一致 (与原 frozen/strategy.py 行为逐位锁定)。
#
# 持久状态字段 (state_spec 投影): low_hit, high_hit, lock_ts, low_acted,
# high_acted, up_st_*, dw_st_*, prev_ts, has_prev。字段名与 numba jitclass
# / CUDA device 函数 单名空间对齐。
_CHANNEL_DEVIATION_DSL = """
# === 第一段: 通道值 (由 framework kernel 在桶切换时 push; DSL 只读) ===
ctx.up = ema_current(ctx.up_sum, ctx.up_count, ctx.up_ema, ctx.cur_high, ctx.p4)
ctx.dw = ema_current(ctx.dw_sum, ctx.dw_count, ctx.dw_ema, ctx.cur_low, ctx.p4)
# === 第二段: 桶切换清锁 + 偏离 + 触发/置位 ===
if ctx.up == 0 or ctx.dw == 0:
    return 0
if ctx.cur_ts != ctx.lock_ts:
    ctx.lock_ts = ctx.cur_ts
    ctx.low_acted = False
    ctx.high_acted = False
ctx.low_dev = (ctx.dw - ctx.cur_low) / ctx.dw * 100
ctx.high_dev = (ctx.cur_high - ctx.up) / ctx.up * 100
ctx.low_dev_h = (ctx.dw - ctx.cur_high) / ctx.dw * 100
ctx.high_dev_l = (ctx.cur_low - ctx.up) / ctx.up * 100
# 参数访问: 按 params_spec 顺序 p0=low1, p1=low2, p2=high1, p3=high2, p4=tf1
signal = 0
if ctx.low_hit and ctx.low_dev_h < ctx.p1 and not ctx.low_acted:
    signal = 1
    ctx.low_hit = False
    ctx.low_acted = True
elif ctx.high_hit and ctx.high_dev_l < ctx.p3 and not ctx.high_acted:
    signal = -1
    ctx.high_hit = False
    ctx.high_acted = True
if ctx.low_dev > ctx.p0 and not ctx.low_acted:
    ctx.low_hit = True
    ctx.low_acted = True
if ctx.high_dev > ctx.p2 and not ctx.high_acted:
    ctx.high_hit = True
    ctx.high_acted = True
return signal
"""


@register_strategy("channel_deviation")
class ChannelDeviationStrategy(StrategyBase):
    """通道偏离回撤策略 (有状态)

    params (dict 或 kwargs):
      tf1:   通道轨 EMA 周期 (策略私有; DSL body 调 ema_push/current 时用 p4)
      low1:  下轨极端偏离阈值 (%)
      low2:  下轨回撤确认阈值 (%)
      high1: 上轨极端偏离阈值 (%)
      high2: 上轨回撤确认阈值 (%)

    state_spec (DSL 持久状态, framework 投影到 numba jitclass / CUDA device 函数):
      low_hit / high_hit:    是否在当前桶触发极端偏离
      lock_ts:               上一次锁定的桶 ts (桶切换时清零)
      low_acted / high_acted: 本桶内是否已对低/高信号下过单 (避免重复触发)
      up_st / dw_st:         EMA 通道增量状态 (3-tuple (sum, count, ema));
                              DSL body 第一段 ema_push/current 自维护
    """

    params_spec = {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "low2":  {"default": 1.0, "type": float, "min": 0.0, "max": 100.0},
        "high1": {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "high2": {"default": 0.5, "type": float, "min": 0.0, "max": 100.0},
        "tf1":   {"default": 21,  "type": int,   "min": 2,   "max": 1000},
    }

    state_spec = {
        "low_hit":        {"type": bool, "default": False},
        "high_hit":       {"type": bool, "default": False},
        "lock_ts":        {"type": int,  "default": 0},
        "low_acted":      {"type": bool, "default": False},
        "high_acted":     {"type": bool, "default": False},
        # 桶切换检测 (DSL body 用): prev_ts = 上一根 bar 的桶 ts;
        # has_prev = 本会话已见过至少一根 bar (避免首根 bar 把空 state 推空)
        "prev_ts":        {"type": int,  "default": 0},
        "has_prev":       {"type": bool, "default": False},
    }

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params=params, **kwargs)
        # ctx 跨桶持久 (state_spec 字段跨 bar 持续); 状态字段初值由 make_dsl_ctx
        # 按 state_spec.default 注入, 不再手工初始化 self.<字段>。
        from .dsl import make_dsl_ctx, make_python_runner
        self._ctx = make_dsl_ctx(type(self))
        # runner 走类级缓存 (make_python_runner 内部按 (cls, source_hash) 复用);
        # 避免每次 __init__ 都 exec 一次原 DSL body。
        self._runner = make_python_runner(type(self), "compute_signal")

    def compute_signal(self, ctx):
        """DSL 主逻辑 (channel_deviation)"""
        pass  # DSL 注入

    def check(self, cur, indicators=None):
        """统一签名: check(cur, indicators: dict) -> (signal|None, info)

        framework 永远传 indicators={} (Engine.on_bars 签名, 见 09 号文档 §2);
        DSL body 第一段自维护指标, 本方法只填 cur_*/pN, 不再接收 positional up/dw。
        历史兼容 (cur, up, dw) 三位置参数形式已删除 (2026-09-09; framework
        完全解耦 EMA)。
        """
        ctx = self._ctx
        ctx.cur_ts = cur["ts"]
        ctx.cur_open = float(cur.get("open", 0.0))
        ctx.cur_high = float(cur["high"])
        ctx.cur_low = float(cur["low"])
        ctx.cur_close = float(cur["close"])
        ctx.cur_volume = float(cur.get("volume", 0.0))
        for i, k in enumerate(self.params_spec):
            setattr(ctx, f"p{i}", float(self.params[k]))
        sig_int = int(self._runner(ctx))
        # low_dev* / up / dw 是 DSL body 里挂在 ctx 上的临时属性, 由 _runner 写入
        signal = {1: "BUY", -1: "SELL"}.get(sig_int)
        # 注: 早期 return (up/dw 尚未就绪) 时 low_dev* 字段未设, getattr 兜底 0.0
        # up/dw 在 ema_current 未就绪时 = 0.0, 但 per-bar 轨迹应与 kernel step 输
        # 出对齐 (kernel 用 _ema_current 未就绪返回 NaN); 把 0.0 转 NaN 便于差分
        # 测试用 equal_nan=True 对位。
        _NAN = float("nan")
        _up = getattr(ctx, "up", 0.0)
        _dw = getattr(ctx, "dw", 0.0)
        if _up == 0.0:
            _up = _NAN
        if _dw == 0.0:
            _dw = _NAN
        info = {"up": _up,
                "dw": _dw,
                "low_dev": getattr(ctx, "low_dev", 0.0),
                "high_dev": getattr(ctx, "high_dev", 0.0),
                "low_dev_h": getattr(ctx, "low_dev_h", 0.0),
                "high_dev_l": getattr(ctx, "high_dev_l", 0.0),
                "low_hit_prev": getattr(ctx, "low_hit", False),
                "high_hit_prev": getattr(ctx, "high_hit", False)}
        return signal, info

    def format_signal_line(self, cur, signal, info):
        """verbose 引擎的信号行 (原 Engine.on_bars 内联格式, 原样迁入)

        framework 不假定任何指标字段; channel_deviation 把 up/dw 与偏离值
        都放在 info dict 里, 由本方法自行取用 (hook 签名无 positional 指标)。
        """
        from ..primitives import fmt
        prefix = f"{signal} >>> " if signal else "             "
        return (f"{prefix}[{cur['ts']}] {cur['code']} | O:{cur['open']} H:{cur['high']} "
                f"L:{cur['low']} C:{cur['close']} | vol:{cur['volume']} x{cur['count']} | "
                f"UP={fmt(info.get('up'))} DW={fmt(info.get('dw'))} | "
                f"low_dev(L/DW)={fmt(info.get('low_dev'))}% "
                f"high_dev(H/UP)={fmt(info.get('high_dev'))}% | "
                f"low_dev_h(H/DW)={fmt(info.get('low_dev_h'))}% "
                f"high_dev_l(L/UP)={fmt(info.get('high_dev_l'))}% | "
                f"low_hit_prev={fmt(info.get('low_hit_prev'))} "
                f"high_hit_prev={fmt(info.get('high_hit_prev'))}")

    def get_extra_bucket_columns(self, *, tab: dict, per_bar: dict) -> dict:
        """策略额外列: 4 个偏离百分比 (与 DSL body 同式)

        framework.bucket_table() 只提供 OHLCV + 信号轨迹 (不输出指标);
        per_bar 由 framework 统一打包 (core.kernel.bundle_per_bar / replay),
        只含 ts/o/h/l/c/v (无 up/dw); 策略 hook 内部按需调
        evtrade.indicators.ema.ema_channel 批量算 per-bar up/dw, 再算偏离。
        """
        import numpy as np
        from ..indicators.ema import ema_channel
        h = per_bar.get("h")
        l = per_bar.get("l")
        if h is None or l is None:
            return {}
        tf1 = int(self.params.get("tf1", 21))
        up_full, dw_full = ema_channel(h, l, tf1)
        per_bar_dev = _compute_deviation_columns(up_full, dw_full, h, l)
        last_idx = np.cumsum(tab["count"]) - 1
        return {k: v[last_idx] for k, v in per_bar_dev.items()}

    def get_extra_signal_columns(self, *, sig, per_bar: dict) -> dict:
        """策略信号轨迹额外列: per-bar EMA 通道 up/dw (hook 内部批量算)

        framework --signals-out 默认仅写 (stime, signal); channel_deviation
        额外暴露 per-bar 通道值, 便于离线复盘脚本画图。per_bar 由 framework
        统一打包 (replay_kernel 返回值), 只含 OHLCV; 策略按需从 per_bar["h"]
        /["l"] 调 evtrade.indicators.ema.ema_channel 批量算。
        """
        import numpy as np
        from ..indicators.ema import ema_channel
        h = per_bar.get("h")
        l = per_bar.get("l")
        if h is None or l is None:
            return {}
        tf1 = int(self.params.get("tf1", 21))
        up_full, dw_full = ema_channel(h, l, tf1)
        # 把 None (前 tf1-1 根) 替换成 NaN, 便于 numpy/csv 消费
        up_arr = np.array([np.nan if v is None else v for v in up_full], dtype=np.float64)
        dw_arr = np.array([np.nan if v is None else v for v in dw_full], dtype=np.float64)
        return {"up": up_arr, "dw": dw_arr}

    def get_extra_signal_columns(self, *, sig, per_bar: dict) -> dict:
        """策略信号轨迹额外列: per-bar EMA 通道 up/dw (hook 内部批量算)

        framework --signals-out 默认仅写 (stime, signal); channel_deviation
        额外暴露 per-bar 通道值, 便于离线复盘脚本画图。per_bar 由 framework
        统一打包 (replay_kernel 返回值), 只含 OHLCV; 策略按需从 per_bar["h"]
        /["l"] 调 evtrade.indicators.ema.ema_channel 批量算。
        """
        import numpy as np
        from ..indicators.ema import ema_channel
        h = per_bar.get("h")
        l = per_bar.get("l")
        if h is None or l is None:
            return {}
        tf1 = int(self.params.get("tf1", 21))
        up_full, dw_full = ema_channel(h, l, tf1)
        # 把 None (前 tf1-1 根) 替换成 NaN, 便于 numpy/csv 消费
        up_arr = np.array([np.nan if v is None else v for v in up_full], dtype=np.float64)
        dw_arr = np.array([np.nan if v is None else v for v in dw_full], dtype=np.float64)
        return {"up": up_arr, "dw": dw_arr}


ChannelDeviationStrategy.compute_signal.__doc__ = _CHANNEL_DEVIATION_DSL


def _compute_deviation_columns(up, dw, h, l) -> dict:
    """四个偏离值 (策略层公式, 与 DSL body 同式)

    输入: 全轨迹或桶级数组 (等长)
    输出: dict 含 low_dev / high_dev / low_dev_h / high_dev_l, 长度同输入。
          up/dw 无效 (None/NaN 或 0) 处偏离值为 NaN。

    这是策略专属的"指标计算"函数 (channel_deviation 的语义: 通道上下轨对
    当根 bar high/low 的偏离百分比), 不应放在 framework.bucket_table 里;
    调用方 (CLI 展示 / 复盘工具 / 测试) 需要时自行调用本函数。
    """
    import numpy as np
    def _arr(x):
        return np.array([np.nan if v is None else v for v in x], dtype=np.float64) \
            if not isinstance(x, np.ndarray) else x
    upa = _arr(up)
    dwa = _arr(dw)
    ha = _arr(h)
    la = _arr(l)
    valid = np.isfinite(upa) & np.isfinite(dwa) & (upa != 0.0) & (dwa != 0.0)
    upv = np.where(valid, upa, np.nan)
    dwv = np.where(valid, dwa, np.nan)
    return {
        "low_dev":    (dwv - la) / dwv * 100.0,
        "high_dev":   (ha - upv) / upv * 100.0,
        "low_dev_h":  (dwv - ha) / dwv * 100.0,
        "high_dev_l": (la - upv) / upv * 100.0,
    }


# 注册 sweep 网格允许的参数名 (框架层 GRID_KEYS 只放引擎级, 策略参数自注册)
from ..core.sweep import register_grid_keys
register_grid_keys(set(ChannelDeviationStrategy.params_spec.keys()))
