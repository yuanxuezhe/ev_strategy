from __future__ import annotations
"""VectorizedStrategy 唯一基类

策略唯一抽象方法 = step(state, bar, params) -> (state, int);
state 由 engine 持有 (dataclass), 策略无 instance attr。

bars 契约 (由 vectorized_engine._aggregate_buckets_xp 聚合后传入):
  {"ts":   int64[N_bucket],   # 桶时间戳 (14 位整数)
   "o":    float64[N],        # 每桶 open (首根 1m bar)
   "h":    float64[N],        # 每桶 high
   "l":    float64[N],        # 每桶 low
   "c":    float64[N],        # 每桶 close
   "v":    float64[N],        # 每桶 volume
   "mark": int8[N],           # 1=策略期 (stime >= warmup), 0=预热期
   "n_bars": int}             # 每桶含 1m bar 根数

step 返回: (state, int); int ∈ {-1, 0, 1} (SELL/无/BUY)
引擎循环调 step, state 由 engine 持有跨调用持续。

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
            def step(self, state, bar, params):
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
    """按 key 构造策略实例

    params: dict 或 None (策略参数)
    kwargs: 自动并入 params (允许 get_strategy('foo', k=v) 风格)
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
    """统一策略基类

    子类必须:
      - 声明 params_spec: dict[str, dict[str, Any]]  (可空 {})
      - 实现 step(self, state, bar, params) -> (state, int)
      - 类上加 @register_strategy("name")
      - (可选) 覆写 init_state(self, params) -> state  返回 state 初值 (默认 None = 无状态)

    算法与执行模型分层:
      - 策略仅做"算法": 拿到 state + 单桶 bar -> 算指标 -> FSM -> 返回 (new_state, sig)
      - 引擎 (Vectorized Engine / Engine.on_bars) 持有 state, 循环调 step
      - 策略 MUST NOT 在 step 内出现批量循环 / xp 模块引用 / mark 之外过滤
      - 策略 MUST NOT 持有 instance-level 持久状态 (无 self._fsm 等)
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

    def init_state(self, params: dict[str, Any]) -> Any:
        """返回 state 初值; 无状态策略默认返 None

        引擎 (vectorized_engine / engine) 在 strategy 实例化时调一次,
        把 state 存到 engine 层 (策略无感)。

        stateful 策略覆写此方法返回 @dataclass 实例。
        """
        return None

    def step(self, state: Any, bar: dict, params: dict) -> tuple[Any, int]:
        """策略唯一入口: 拿到 state + 单桶 bar -> 返回 (new_state, signal)

        bar:  {"ts", "o", "h", "l", "c", "v", "mark"} (mark=0 预热段)
        params: 已解析参数 dict (= self.params)
        返回: (new_state, signal), signal ∈ {-1, 0, 1} (SELL/无/BUY)

        策略 MUST:
          - 若 bar["mark"] == 0: return state, 0  (预热段)
          - 仅做"算法": 算指标 + FSM, 不持有 instance state
          - 不出现 for i in range(n) 批量循环
        """
        raise NotImplementedError

    def format_signal_line(self, ts: int, sig: int, info: dict | None = None) -> str:
        """策略展示 hook: 自定义信号行打印; 默认显示 ts/sig。

        info 字典: 由 strategy.step() 累积的可选元数据
                  (e.g. 当前通道上轨/下轨/通道宽度)。框架不假设 info 键集。
        """
        from ..primitives import sig_to_side
        side = sig_to_side(sig)
        return f"{ts} sig={sig:+d} {side}".rstrip()

    def get_extra_bucket_columns(self) -> list[str]:
        """策略展示 hook: 桶表追加列名"""
        return []

    def get_extra_signal_columns(self) -> list[str]:
        """策略展示 hook: 信号行追加列名"""
        return []
