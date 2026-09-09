from __future__ import annotations
"""统一策略基类 (CPU+GPU 单一份契约)

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
DSL 删除后, evtrade 只剩这一个策略基类:
  - 唯一抽象方法: compute_signals(xp, bars, params) -> xp.ndarray[int8]
  - framework 包装 compute_signals_for_one_bar 给 Engine.on_bars / 直播用
  - 子类可覆写 compute_signals_for_one_bar 维护 instance-level 增量 state
    (e.g. channel_deviation 的桶切换 FSM)
  - framework 不持有任何 DSL / numba / CUDA 渲染逻辑

bars 契约 (由 vectorized_engine._aggregate_buckets_xp 聚合后传入):
  {"ts":   int64[N_bucket],   # 桶时间戳 (14 位整数)
   "o":    float64[N],        # 每桶 open (首根 1m bar)
   "h":    float64[N],        # 每桶 high (桶内 1m bar 最高)
   "l":    float64[N],        # 每桶 low  (桶内 1m bar 最低)
   "c":    float64[N],        # 每桶 close (桶内最后一根 1m bar)
   "v":    float64[N],        # 每桶 volume (桶内 1m bar 成交量累加)
   "mark": int8[N],           # 1=策略期 (stime >= warmup), 0=预热期
   "n_bars": int}             # 每桶含 1m bar 根数 (供调试)

返回: sig (int8[N]), 每桶一个信号: 0=无 / 1=BUY / -1=SELL

策略注册: @register_strategy("name") (放在类声明前一行)。
params 解析: 子类声明 params_spec (dict of {type, default, min, max}),
            __init__ 自动按 spec 校验/填默认。
"""
from typing import Any


# ============ 注册表 (唯一来源) ============
_STRATEGIES: dict[str, type["VectorizedStrategy"]] = {}


def register_strategy(name: str):
    """类装饰器: 注册策略到 _STRATEGIES 表
    用法:
        @register_strategy("ma_crossover")
        class MACrossoverStrategy(VectorizedStrategy):
            params_spec = {...}
            def compute_signals(self, xp, bars, params):
                ...
    """
    def deco(cls):
        if name in _STRATEGIES and _STRATEGIES[name] is not cls:
            raise ValueError(f"策略名 {name!r} 已注册为 {_STRATEGIES[name].__name__}")
        cls.strategy_key = name
        _STRATEGIES[name] = cls
        return cls
    return deco


def get_strategy(name: str, params: dict | None = None, **kwargs) -> "VectorizedStrategy":
    """按 key 构造策略实例 (CPU/GPU/ref 三路径统一入口)

    params: dict 或 None (策略参数)
    kwargs: 备用, 自动并入 params (向后兼容 get_strategy('foo', k=v) 风格)
    """
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}; 可用: {sorted(_STRATEGIES)}")
    merged = dict(params or {})
    merged.update(kwargs)
    return _STRATEGIES[name](params=merged)


def get_strategy_class(name: str) -> type["VectorizedStrategy"]:
    """按 key 取策略类 (无需实例化; 用于 type check / sweep grid 注册)"""
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}; 可用: {sorted(_STRATEGIES)}")
    return _STRATEGIES[name]


def get_strategy_param_spec(name: str) -> dict[str, dict[str, Any]]:
    """按 key 取 params_spec (用于 sweep grid / CLI 校验)"""
    return getattr(_STRATEGIES[name], "params_spec", {}) or {}


def available_strategies() -> list[str]:
    """返回已注册策略 key 列表"""
    return sorted(_STRATEGIES)


# ============ 唯一基类 ============


class VectorizedStrategy:
    """统一策略基类 (DSL/ref kernel 已下线, 全框架只剩这一份契约)

    子类必须:
      - 声明 params_spec: dict[str, dict[str, Any]]  (可空 {})
      - 实现 compute_signals(xp, bars, params) -> xp.ndarray[int8]
      - 类上加 @register_strategy("name")
      - (可选) 覆写 compute_signals_for_one_bar 维护 instance 增量 state

    默认 compute_signals_for_one_bar 包装 compute_signals:
      单 bar -> 单元素数组 -> compute_signals -> [0]。
    子类覆写该方法维护 instance state (避免每根 O(n) 重算)。
    """

    params_spec: dict[str, dict[str, Any]] = {}
    strategy_key: str = ""

    def __init__(self, params: dict[str, Any] | None = None, **kwargs):
        merged = dict(params or {})
        for k, v in kwargs.items():
            merged.setdefault(k, v)
        self.params = self._resolve_params(merged)
        for k, v in self.params.items():
            setattr(self, k, v)

    @classmethod
    def _resolve_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """按 params_spec 校验/填默认 (框架唯一来源)"""
        spec = cls.params_spec or {}
        unknown = set(params) - set(spec)
        if unknown:
            raise ValueError(
                f"{cls.__name__} 收到未声明的参数 {sorted(unknown)}; "
                f"已知参数: {sorted(spec)}"
            )
        out: dict[str, Any] = {}
        for k, schema in spec.items():
            v = params.get(k, schema.get("default"))
            if v is None:
                raise ValueError(f"{cls.__name__}.{k} 缺默认值; params={params}")
            t = schema.get("type")
            if t is not None and not isinstance(v, t):
                if t is float and isinstance(v, int):
                    v = float(v)
                elif t is int and isinstance(v, float) and v.is_integer():
                    v = int(v)
                else:
                    raise ValueError(
                        f"{cls.__name__}.{k} 期望 {t.__name__}, 收到 "
                        f"{type(v).__name__}={v!r}"
                    )
            mn = schema.get("min")
            mx = schema.get("max")
            if mn is not None and v < mn:
                raise ValueError(f"{cls.__name__}.{k}={v} 小于 min={mn}")
            if mx is not None and v > mx:
                raise ValueError(f"{cls.__name__}.{k}={v} 大于 max={mx}")
            out[k] = v
        return out

    def compute_signals(self, xp, bars: dict, params: dict):
        """子类必须实现: 数组算子 -> 信号数组 (int8)

        xp: numpy 或 cupy 模块 (vectorized_engine 传入)
        bars: 聚合后桶数组 dict (见模块 docstring 契约)
        params: 已解析参数 dict (= self.params)
        返回: xp.ndarray[int8], 长度 = len(bars["ts"])
        """
        raise NotImplementedError

    def compute_signals_for_one_bar(self, xp, bar: dict, params: dict) -> int:
        """framework 包装: 单 bar -> compute_signals[0]

        子类若需要维护 instance-level 增量 state (如桶切换 FSM),
        可覆写本方法; 默认实现走批量 compute_signals。
        """
        single = {k: xp.asarray([v]) for k, v in bar.items()}
        out = self.compute_signals(xp, single, params)
        return int(out[0])

    def format_signal_line(self, ts: int, sig: int, info: dict | None = None) -> str:
        """策略展示 hook: 自定义信号行打印; 默认显示 ts/sig。

        info 字典: 由 strategy.check() / compute_signals_for_one_bar 累积的可选元数据
                  (e.g. 当前通道上轨/下轨/通道宽度)。框架不假设 info 键集。
        """
        side = {1: "BUY", -1: "SELL"}.get(sig, "")
        return f"{ts} sig={sig:+d} {side}".rstrip()

    def get_extra_bucket_columns(self) -> list[str]:
        """策略展示 hook: 桶表追加列名 (framework bucket_table 拼接用)"""
        return []

    def get_extra_signal_columns(self) -> list[str]:
        """策略展示 hook: 信号行追加列名"""
        return []
