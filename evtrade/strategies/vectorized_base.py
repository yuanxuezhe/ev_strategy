"""VectorizedStrategy 唯一基类

策略唯一抽象方法 = step(state, bar, params) -> (state, int);
state 由 engine 持有 (dataclass), 策略无 instance attr。

bar 契约 (vectorized_engine._aggregate_buckets 传入):
  {"ts", "o", "h", "l", "c", "v", "mark"}
  mark ∈ {0, 1}: 0=预热段 (策略不产信号), 1=策略期

step 返回: (state, signal); signal ∈ {-1, 0, 1} (SELL/无/BUY)
"""
from typing import Any


# ============ 注册表 ============

_STRATEGIES: dict[str, type["VectorizedStrategy"]] = {}


def register_strategy(name: str):
    """类装饰器: 注册策略到 _STRATEGIES 表"""
    def deco(cls):
        if name in _STRATEGIES and _STRATEGIES[name] is not cls:
            raise ValueError(f"策略名 {name!r} 已注册为 {_STRATEGIES[name].__name__}")
        cls.strategy_key = name
        _STRATEGIES[name] = cls
        return cls
    return deco


def get_strategy(name: str, params: dict | None = None, **kwargs) -> "VectorizedStrategy":
    """按 key 构造策略实例 (kwargs 自动并入 params)"""
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}; 可用: {sorted(_STRATEGIES)}")
    merged = dict(params or {})
    merged.update(kwargs)
    return _STRATEGIES[name](params=merged)


def get_strategy_class(name: str) -> type["VectorizedStrategy"]:
    """按 key 取策略类 (无需实例化)"""
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}; 可用: {sorted(_STRATEGIES)}")
    return _STRATEGIES[name]


def get_strategy_param_spec(name: str) -> dict[str, dict[str, Any]]:
    """按 key 取 params_spec (sweep grid / CLI 校验用)"""
    return getattr(_STRATEGIES[name], "params_spec", {}) or {}


def available_strategies() -> list[str]:
    """已注册策略 key 列表 (按字母序)"""
    return sorted(_STRATEGIES)


# ============ 唯一基类 ============


class VectorizedStrategy:
    """统一策略基类

    子类必须:
      - 声明 params_spec: dict[str, dict[str, Any]]  (可空 {})
      - 实现 step(self, state, bar, params) -> (state, int)
      - 类上加 @register_strategy("name")
      - (可选) 覆写 init_state(self, params) -> state  返回 state 初值 (默认 None)

    引擎 (Vectorized Engine / Engine.on_bars) 持有 state, 循环调 step。
    策略 MUST NOT 在 step 内出现批量循环 / 持有 instance-level 持久状态。
    """

    params_spec: dict[str, dict[str, Any]] = {}
    strategy_key: str = ""

    def __init__(self, params: dict[str, Any] | None = None, **kwargs):
        merged = dict(params or {})
        merged.update(kwargs)
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
        """state 初值; 无状态策略默认 None。stateful 策略覆写返回 @dataclass 实例"""
        return None

    def step(self, state: Any, bar: dict, params: dict) -> tuple[Any, int]:
        """策略唯一入口: state + 单桶 bar -> (new_state, signal)"""
        raise NotImplementedError

    def format_signal_line(self, ts: int, sig: int, info: dict | None = None) -> str:
        """策略展示 hook: 自定义信号行打印; 默认显示 ts/sig/side"""
        from ..primitives import sig_to_side
        side = sig_to_side(sig)
        return f"{ts} sig={sig:+d} {side}".rstrip()
