from __future__ import annotations
"""向量化策略基类 (CuPy 统一 CPU/GPU 路径)

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
与 StrategyBase (逐 bar check) 平行的第二条策略契约:
  - 数组进数组出: compute_signals(xp, bars, params) -> sig (xp.ndarray[int8])
  - 用 xp (numpy 或 cupy) 写数组算子, 一份代码跑 CPU/GPU
  - 不走 DSL / numba / CUDA 渲染; framework 0 渲染逻辑

适用: 可向量化的策略 (MA 交叉 / 突破 / 协方差等标准数组运算)。
不适用: 逐 bar 状态机 (channel_deviation 的 if/elif 锁存) —— 那些走 DSL/numba。

bars 契约 (由 vectorized_engine 聚合后传入):
  {"ts":   int64[N_bucket],   # 桶时间戳 (14 位整数)
   "o":    float64[N],        # 每桶 open (首根 1m bar)
   "h":    float64[N],        # 每桶 high (桶内 1m bar 最高)
   "l":    float64[N],        # 每桶 low  (桶内 1m bar 最低)
   "c":    float64[N],        # 每桶 close (桶内最后一根 1m bar)
   "v":    float64[N],        # 每桶 volume (桶内 1m bar 成交量累加)
   "mark": int8[N],           # 1=策略期 (stime >= warmup), 0=预热期
   "n_bars": int}             # 每桶含 1m bar 根数 (供调试)

返回: sig (int8[N]), 每桶一个信号: 0=无 / 1=BUY / -1=SELL
"""
from typing import Any

from .base import _STRATEGIES, register_strategy


class VectorizedStrategy:
    """向量化策略基类 (xp=np|cp 自动切后端)

    子类必须:
      - 声明类属性 params_spec (dict, schema 与 StrategyBase 相同)
      - 实现 compute_signals(xp, bars, params) -> xp.ndarray[int8]
      - 类上加 @register_strategy("name")

    参数解析复用 StrategyBase.__init__ 的 params_spec 逻辑 (本类不持有逐 bar
    状态, 但 params 仍按 params_spec 校验/填默认)。
    """

    params_spec: dict[str, dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any] | None = None, **kwargs):
        merged = dict(params or {})
        for k, v in kwargs.items():
            merged.setdefault(k, v)
        self.params = self._resolve_params(merged)
        for k, v in self.params.items():
            setattr(self, k, v)

    @classmethod
    def _resolve_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """与 StrategyBase._resolve_params 同式 (params_spec 填默认 + 校验)"""
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
        """子类实现: 数组算子 -> 信号数组 (int8)

        xp: numpy 或 cupy 模块 (由 vectorized_engine 传入)
        bars: 聚合后的桶数组 dict (见模块 docstring 契约)
        params: 已解析的参数 dict (与 self.params 同)
        返回: xp.ndarray[int8], 长度 = len(bars["ts"])
        """
        raise NotImplementedError


def get_vectorized_strategy(name: str, params: dict | None = None) -> VectorizedStrategy:
    """按 key 构造向量化策略实例 (与 get_strategy 平行; 向量化策略走这个入口)"""
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}; 可用: {list(_STRATEGIES)}")
    cls = _STRATEGIES[name]
    if not (isinstance(cls, type) and issubclass(cls, VectorizedStrategy)):
        raise ValueError(
            f"策略 {name!r} 不是 VectorizedStrategy 子类; "
            f"vectorized 引擎只支持向量化策略 (继承 VectorizedStrategy)"
        )
    return cls(params=params)
