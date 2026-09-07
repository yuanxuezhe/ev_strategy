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

DSL 策略状态字段 (2026-09-07): state_spec
  - DSL 策略在三类之间共享的"持久状态字段" (跨 bar 持续, 桶切换可重置),
    由策略声明, framework 投影到 Python ctx / numba jitclass / CUDA device 函数
  - schema 与 params_spec 平行: {name: {"type": Python 原生类型 (bool/int/float),
                                       "default": 标量初值}}
  - 框架会自动:
      * build_ctx_to_kernel_map(Cls)   把 state_spec 字段名映射到 kernel 状态
      * build_cuda_sig_fields(Cls)     校验 DSL body 只引用签名内的字段
      * build_cuda_device_header(Cls)  生成 __device__ 函数签名
      * make_dsl_ctx(Cls)              构造 Python 端 ctx 实例 (字段初值 = default)
  - 新增 DSL 策略: 在 state_spec 声明自己需要的持久字段; 不需要持久状态的策略
    可以 state_spec = {} (空字典) 或省略。

新增策略:
  1. 在 strategies/ 子目录写一个文件 (例 my_strategy.py)
  2. 继承 StrategyBase, 声明类属性 params_spec (必须) 与 state_spec (DSL 策略必须,
     非 DSL 策略可省略)
  3. 类上加 @register_strategy("my_strategy")
  4. CLI 自动可用 --strategy my_strategy --params "..."

当前实现限制 (2026-09-06):
  - 带 DSL docstring 的策略: 三端同源 (参考引擎 + numba 内核 + CUDA, 见 dsl.py
    与 core/kernel_dsl.py); 无 DSL 的策略仅参考引擎路径 (慢约 500x)
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
        "key1":  {"default": 1.5, "type": float, "min": 0.0, "max": 100.0},
        "key2":  {"default": 21,  "type": int,   "min": 2,    "max": 1000},
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
    # DSL 策略跨桶持久的状态字段 (framework 投影到 Python ctx / numba jitclass /
    # CUDA device 函数); 非 DSL 策略可省略 (留空 dict)。schema:
    #   {"field_name": {"type": bool | int | float, "default": <标量初值>}}
    # 框架层 type 映射: bool → numba.boolean, int → numba.int64, float → numba.float64
    # (在 dsl.py / kernel.py / gpu.py 的工厂函数中统一消费)
    state_spec: dict[str, dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any] | None = None, **kwargs):
        merged = dict(params or {})
        for k, v in kwargs.items():
            merged.setdefault(k, v)
        self.params = self._resolve_params(merged)
        # 把每个参数也设为实例属性, 方便 self.<key> 直接访问
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

    def format_signal_line(self, cur: dict, signal, info: dict) -> str:
        """verbose 引擎的信号 bar 打印行 (Engine 只负责 print, 不假设 info 的键)

        framework 不假定任何特定指标; 策略覆写本方法时, 所需字段从 info dict
        自己取 (e.g. up/dw、偏离值、过滤标志等)。默认实现仅打 OHLCV + 信号。
        """
        return (f"{signal} >>> [{cur['ts']}] {cur['code']} | O:{cur['open']} "
                f"H:{cur['high']} L:{cur['low']} C:{cur['close']} | "
                f"vol:{cur['volume']} x{cur['count']}")

    def get_extra_bucket_columns(self, *, tab: dict, per_bar: dict) -> dict:
        """策略在 framework 桶表之上追加的展示列 (CLI --show-bars / --bars-out 用)

        tab 是 framework.bucket_table() 的输出 (OHLCV + sig + n_sig,
        不含策略专属指标); per_bar 是 caller 提供的 per-bar 数组 dict
        (e.g. {"up": ..., "dw": ..., "h": ..., "l": ...} 来自内核 /
        策略模块, 由 framework 的 bundle_per_bar 统一封装, 调用方不
        直接命名指标字段)。

        策略覆写此方法追加自己的指标列 (numpy 数组, 长度与 tab['ts'] 一致,
        已按 tab['count'] 桶聚合)。默认空 dict, 即仅展示 framework 字段;
        framework 不假定任何策略有额外列。CLI/复盘工具按 key->value 形式
        迭代展示, 不感知具体列名。
        """
        return {}

    def get_extra_signal_columns(self, *, sig, per_bar: dict) -> dict:
        """策略在 framework 信号轨迹之上追加的 per-bar 列 (CLI --signals-out 用)

        framework 默认 --signals-out CSV 仅写 (stime, signal); 策略可覆写
        本方法追加 per-bar 列 (e.g. 通道 up/dw、信号评分、过滤标志等)。
        per_bar 是 caller 提供的 per-bar 数组 dict (框架 bundle 形式,
        具体键名策略自己解; framework 不假定), 长度与 sig 相同。
        """
        return {}

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


def get_strategy_class(name: str) -> type[StrategyBase]:
    """按 key 拿注册过的策略类 (不实例化)

    用于编译期探针 / DSL 渲染 / 类元数据查询 —— 任何不需要 self 的地方都
    应优先使用本函数, 避免 get_strategy() 实例化触发的副作用
    (channel_deviation.__init__ 等会在构造时执行 make_python_runner/exec)。
    未知策略抛 ValueError。
    """
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}; 可用: {list(_STRATEGIES)}")
    return _STRATEGIES[name]


def get_strategy_param_spec(name: str) -> dict[str, dict[str, Any]]:
    """查策略的 params_spec (CLI / jupyter 用)"""
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}")
    return _STRATEGIES[name].params_spec


def get_strategy_state_spec(name: str) -> dict[str, dict[str, Any]]:
    """查策略的 state_spec (DSL framework 工厂函数用; 编译期/渲染期统一来源)

    state_spec 为空 dict 表示该策略无跨桶持久状态字段 (DSL body 里只能写
    局部变量, 不能写 ctx.<持久字段>)。

    字段 schema: {name: {"type": bool | int | float, "default": <标量>}}
      - type 必须是 Python 原生 bool/int/float (与 numba boolean/int64/float64 一一映射)
      - default 必须与 type 一致
    """
    if name not in _STRATEGIES:
        raise ValueError(f"未知策略 {name!r}")
    return _STRATEGIES[name].state_spec or {}


def available_strategies() -> list[str]:
    return sorted(_STRATEGIES.keys())
