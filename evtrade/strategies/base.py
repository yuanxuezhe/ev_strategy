from __future__ import annotations
"""Strategy 基类 + 策略注册表 + 通用参数 dict

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
策略的契约: check(cur, indicators) -> (signal, info)
  signal ∈ {"BUY", "SELL", None}
  info   是 dict, 用于打印/复盘

策略参数通用 dict 形式 (2026-09-06):
  - 策略类声明 params_spec: {key: {"default": ..., "type": float/int/bool, "min"/"max": ...}}
  - __init__(params: dict | None, **kwargs) 接收统一 dict, 用 self.params[k] 访问
  - get_strategy(name, params={...}) 或 get_strategy(name, k1=v1, k2=v2) 都可用
  - CLI 通过 --params "k1:v1;k2:v2" 传入, sweep 通过 --grid "k1=v1,v2;k2=v3,v4" 笛卡尔积

新增策略:
  1. 在 strategies/ 子目录写一个文件 (例 my_strategy.py)
  2. 继承 StrategyBase, 声明类属性 params_spec
  3. 类上加 @register_strategy("my_strategy")
  4. CLI 自动可用 --strategy my_strategy --params "..."

当前实现限制 (2026-09-06):
  - 策略仅走参考引擎路径
  - numba 内核 + GPU kernel 仍硬编码 channel_deviation (low1/low2/high1/high2)
  - 未来: 做 AST 转译器, 让策略主逻辑一份 Python 自动派生 numba/CUDA
================================================================
"""
from typing import Any, Callable, Type


class StrategyBase:
    """所有策略的基类

    子类必须:
      - 设置类属性 name (str)
      - 声明类属性 params_spec (dict): 参数 schema
      - 实现 check(cur, indicators) -> (signal, info)

    params_spec 格式:
      {
        "low1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "tf1":   {"default": 21,  "type": int,   "min": 2,    "max": 1000},
        ...
      }
      至少声明 default (可不写 type / min / max)。

    __init__(params: dict | None = None, **kwargs) 自动:
      - 合并 params 与 kwargs (kwargs 优先级低)
      - 用 params_spec 填默认值
      - 校验 type / min / max (有则校验)
      - 把结果存到 self.params (dict) 与 self.<key> 属性
    """

    name: str = ""
    params_spec: dict[str, dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any] | None = None, **kwargs):
        merged = dict(params or {})
        for k, v in kwargs.items():
            merged.setdefault(k, v)
        self.params = self._resolve_params(merged)
        # 把每个参数也设为实例属性, 方便 self.low1 / self.tf1 直接访问
        for k, v in self.params.items():
            setattr(self, k, v)

    @classmethod
    def _resolve_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """用 params_spec 填默认 + 校验; 返回扁平 dict (所有 key 都有值)"""
        spec = cls.params_spec or {}
        # 警告: 未声明的参数 (避免静默拼写错误)
        unknown = set(params) - set(spec)
        if unknown:
            raise ValueError(
                f"{cls.__name__} 收到未声明的参数 {sorted(unknown)}; "
                f"已知参数: {sorted(spec)}"
            )
        # 填默认 + 校验
        out: dict[str, Any] = {}
        for k, schema in spec.items():
            v = params.get(k, schema.get("default"))
            if v is None:
                raise ValueError(f"{cls.__name__}.{k} 缺默认值; params={params}")
            # 类型校验 / 转换
            t = schema.get("type")
            if t is not None and not isinstance(v, t):
                if t is float and isinstance(v, int):
                    v = float(v)
                elif t is int and isinstance(v, float) and v.is_integer():
                    v = int(v)
                else:
                    raise TypeError(
                        f"{cls.__name__}.{k} 期望 {t.__name__}, 收到 {type(v).__name__}: {v!r}"
                    )
            # 范围校验
            if "min" in schema and v < schema["min"]:
                raise ValueError(f"{cls.__name__}.{k}={v} 小于最小值 {schema['min']}")
            if "max" in schema and v > schema["max"]:
                raise ValueError(f"{cls.__name__}.{k}={v} 大于最大值 {schema['max']}")
            out[k] = v
        return out

    def check(self, cur: dict, indicators: dict):
        """返回 (signal, info); signal ∈ {"BUY", "SELL", None}"""
        raise NotImplementedError

    def get_param(self, key: str, default=None):
        """从 self.params 取参数, 缺则 default"""
        return self.params.get(key, default)


# ============ 策略注册表 ============

_STRATEGIES: dict[str, Type[StrategyBase]] = {}


def register_strategy(name: str) -> Callable[[Type[StrategyBase]], Type[StrategyBase]]:
    """@register_strategy("xxx") 装饰器"""
    def deco(cls: Type[StrategyBase]) -> Type[StrategyBase]:
        if not name:
            raise ValueError("@register_strategy 需要显式 name")
        if name in _STRATEGIES:
            raise ValueError(f"策略名 {name!r} 已注册为 {_STRATEGIES[name].__name__}")
        cls.name = name
        _STRATEGIES[name] = cls
        return cls
    return deco


def get_strategy(name: str, params: dict[str, Any] | None = None, **kwargs) -> StrategyBase:
    """按 key 构造策略实例

    两种调用形式都支持:
      get_strategy("channel_deviation", params={"low1": 1.5, "low2": 1.0})
      get_strategy("channel_deviation", low1=1.5, low2=1.0)        # 旧式 kwargs
      get_strategy("channel_deviation", params={...}, extra="x")    # 混合
    """
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}; 可用: {list(_STRATEGIES)}")
    return _STRATEGIES[name](params=params, **kwargs)


def get_strategy_param_spec(name: str) -> dict[str, dict[str, Any]]:
    """查策略的 params_spec (CLI / jupyter 用)"""
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}")
    return _STRATEGIES[name].params_spec


def available_strategies() -> list[str]:
    return sorted(_STRATEGIES.keys())
