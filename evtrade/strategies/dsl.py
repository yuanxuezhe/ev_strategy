from __future__ import annotations
"""策略 DSL: 同一份主逻辑 -> Python + numba njit (CUDA 后续扩展)

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
设计:
  策略作者在 compute_signal(ctx) 方法的 docstring 里写 DSL 代码,
  表达式只允许标量算术/比较/布尔/属性赋值/return signal。

  DSL 编译三端共用一次 parse+validate, 再各自渲染:
    - make_python_runner:    原 docstring body 直接 exec, 不经渲染 (避免漂移)
    - render_numba_body:     手写 AST unparser -> numba @njit 函数体
    - render_cuda_body:      手写 C99 渲染器 -> CUDA 函数体 (调试视图)
    - render_cuda_device_function: AST -> __device__ strategy_check 整函数
                                  (gpu.py 通用模板注入)
    - compile_all:           一次返回上述三份渲染产物 (调试 / 验证用)

约束 (whitelist):
  ✅ 算术: + - * /
  ✅ 比较: < > <= >= == !=
  ✅ 布尔: and or not
  ✅ 常量: 数字 / True / False
  ✅ 标量赋值: ctx.x = value, x = value
  ✅ 属性读写: ctx.x, self.x (限制字段)
  ✅ return signal
  ❌ 字符串 / 列表 / 字典 / 调用外部函数 / 循环 (除受控 for)
  ⚠️  调用仅允许白名单内建: min / max / abs (三端一致)
  ❌ 异常处理 / with / yield / lambda

不满足 -> 编译期抛 CompileError, 列出不支持的节点。

物理保证 bitwise 一致:
  同一份 AST 节点 -> 同一份字符串 -> 同一份机器码 (编译顺序由编译器固定);
  Python 与 numba 共享同一字面表达式, 浮点路径完全相同。

DSL → 三端转译 (2026-09-06 步骤 1):
  - make_python_runner: 直接 exec, 不重新渲染 (避免漂移)
  - render_numba_body:  ast.unparse 函数体, 给 numba @njit 用
  - render_cuda_body:   ast.unparse + C99 化 (类型转换 + 浮点字面量), 给 CUDA kernel 用
  - compile_all:       一次返回三份源码 (调试 / 验证用)

内核状态映射 (步骤 2, kernel.py 读 DSL 渲染):
  - _CTX_TO_KERNEL:              ctx 字段 -> 内核状态名契约 (单一事实源)
  - render_numba_state_body:     ctx.X -> st.<内核名>, kernel_dsl.splice 用
  - render_cuda_device_function: DSL -> __device__ strategy_check
                                 (gpu.py 通用模板注入; return 语义三端一致)
  - DSLCtx + dsl_check:          Python 端通用 ctx, 新策略零样板接入参考引擎
================================================================
"""
import ast
import re
import textwrap


class CompileError(Exception):
    """DSL 不满足白名单时抛出"""


# numba/CUDA 不支持的关键字 / 类型
# 注: ast.Call 不在此处 —— _validate 单独处理 (仅允许白名单 min/max/abs)
_FORBIDDEN_NODES = (
    ast.List, ast.Dict, ast.Set, ast.Tuple,
    ast.Lambda,
    ast.Yield, ast.YieldFrom,
    ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith,
    ast.Starred,
    ast.JoinedStr,      # f-string
    # 字符串节点 (除常量)
)


# DSL 允许的函数调用白名单 (仅此三者, 三端一致)
_CALL_WHITELIST = frozenset({"min", "max", "abs"})


def _validate(tree: ast.Module) -> None:
    """白名单校验; 不通过抛 CompileError

    所有未识别的 ast 节点会被末尾的兜底分支拒绝, 而不是被静默放行
    (静默放行会导致 Python exec 执行它们, 而 numba/CUDA 渲染器稍后报错,
    错误信息错位, 调试困难)。
    """
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise CompileError(f"DSL 不支持节点: {type(node).__name__} "
                               f"(行 {getattr(node, 'lineno', '?')})")
        if isinstance(node, ast.Call):
            # 仅允许白名单内建; 不接受关键字参数 (min/max/abs 都是单参数或双位置参数)
            if (isinstance(node.func, ast.Name)
                    and node.func.id in _CALL_WHITELIST
                    and not node.keywords):
                continue
            raise CompileError(
                f"DSL 不支持调用: "
                f"{ast.unparse(node.func) if hasattr(ast, 'unparse') else type(node.func).__name__} "
                f"(仅允许 min/max/abs, 且必须位置参数)"
            )
        if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
            continue
        if isinstance(node, ast.Constant):
            continue
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            continue
        if isinstance(node, ast.Assign):
            # 仅允许 ctx.x = v 形式 (单目标, Attribute 或 Name)
            if len(node.targets) != 1:
                raise CompileError("DSL 不允许多元赋值")
            continue
        if isinstance(node, ast.Attribute):
            continue
        if isinstance(node, ast.Name):
            continue
        if isinstance(node, ast.Return):
            continue
        if isinstance(node, ast.If):
            continue
        if isinstance(node, ast.Expr):
            continue
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.arguments, ast.arg)):
            continue
        # ctx=Load/Store/Del 是 Name/Attribute 的访问方式标记, 不是独立节点
        if isinstance(node, (ast.Load, ast.Store, ast.Del)):
            continue
        # BinOp/Compare/BoolOp 的 op 字段是 operator 子类 (ast.Add/Sub/...), 不是独立节点
        if isinstance(node, (ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
            continue
        # 兜底: 未识别的 ast 节点 (IfExp / Subscript / Slice / Starred / Match* /
        # keyword / AugAssign 等) 一律拒绝, 避免 Python exec 静默执行它们
        raise CompileError(f"DSL 不支持节点: {type(node).__name__} "
                           f"(行 {getattr(node, 'lineno', '?')})")


def _extract_dsl_body(strategy_class, method_name: str = "compute_signal") -> str:
    """从策略类的 compute_signal 方法的 docstring 提取 DSL 代码 (实例或类均可)"""
    cls = strategy_class if isinstance(strategy_class, type) else type(strategy_class)
    method = getattr(cls, method_name, None)
    if method is None or method.__doc__ is None:
        raise CompileError(f"{cls.__name__}.{method_name} 缺少 docstring")
    return textwrap.dedent(method.__doc__)


def make_python_runner(strategy_class, method_name: str = "compute_signal"):
    """返回 fn(ctx) -> signal

    Python 直接解释执行 DSL, 不经 AST -> 字符串 -> 编译的往返,
    以保留原始浮点表达式顺序 (避免任何重渲染漂移)。

    为安全起见, exec 之前先 parse+validate 一遍: 验证不通过立刻抛 CompileError,
    验证通过后才 exec 原 body (原 body 与解析树等价, 无重渲染漂移)。
    """
    body = _extract_dsl_body(strategy_class, method_name)
    wrapped = f"def _f(ctx):\n{textwrap.indent(body, '    ')}"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as e:
        raise CompileError(f"DSL 语法错误: {e}") from e
    _validate(tree)
    namespace: dict = {}
    # 用 exec 直接定义 _f 函数 (保留原 body 不经渲染, 避免漂移)
    exec("def _f(ctx):\n" + textwrap.indent(body, "    "), namespace)
    return namespace["_f"]


def render_numba_body(strategy_class, method_name: str = "compute_signal") -> str:
    """把 DSL 体渲染成 numba njit 函数体

    关键约束: 表达式字面顺序与 Python 版完全一致; 字段名一致
    (ctx 在 Python 版是普通对象, 在 numba 版是 jitclass,
     通过 .x 访问字段的语法不变)。
    """
    body = _extract_dsl_body(strategy_class, method_name)
    wrapped = f"def _f(ctx):\n{textwrap.indent(body, '    ')}"
    tree = ast.parse(wrapped)
    _validate(tree)
    body_nodes = tree.body[0].body  # 跳过 def _f
    return "\n".join(_unparse_stmt(n) for n in body_nodes)


def _indent_block(src: str, pad: str = "    ") -> list[str]:
    """多行子块的整块缩进 (嵌套 if 的续行也要带上外层缩进)"""
    return [pad + ln if ln.strip() else "" for ln in src.split("\n")]


def _unparse_stmt(node) -> str:
    """简单 stmt/expr unparse (兼容 Python 3.8, 无 ast.unparse)"""
    if isinstance(node, ast.If):
        cond = _unparse_expr(node.test)
        lines = [f"if {cond}:"]
        for s in node.body:
            lines.extend(_indent_block(_unparse_stmt(s)))
        if node.orelse:
            lines.append("else:")
            for s in node.orelse:
                lines.extend(_indent_block(_unparse_stmt(s)))
        return "\n".join(lines)
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise CompileError("DSL 不允许多元赋值")
        return f"{_unparse_expr(node.targets[0])} = {_unparse_expr(node.value)}"
    if isinstance(node, ast.Return):
        return f"return {_unparse_expr(node.value)}"
    if isinstance(node, ast.Expr):
        return _unparse_expr(node.value)
    raise CompileError(f"无法 unparse {type(node).__name__}: {ast.dump(node)}")


def _unparse_expr(node) -> str:
    """expr unparse (Python 风格, 给 numba 用)"""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        # ctx.xxx -> ctx.xxx (numba 用 . 访问 jitclass)
        return f"{_unparse_expr(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        # 白名单 (min/max/abs) 已由 _validate 校验; 这里只负责文本生成
        if not (isinstance(node.func, ast.Name)
                and node.func.id in _CALL_WHITELIST):
            raise CompileError(f"无法 unparse expr Call {ast.dump(node)}")
        args = ", ".join(_unparse_expr(a) for a in node.args)
        return f"{node.func.id}({args})"
    if isinstance(node, ast.BinOp):
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
                  ast.Div: "/", ast.Mod: "%", ast.Pow: "**"}
        # Python 3.8 兼容: BinOp 子类名是小写 (ast.mult)
        if ast.Mult not in op_map and hasattr(ast, "mult"):
            op_map = {ast.Add: "+", ast.Sub: "-", ast.mult: "*",
                      ast.Div: "/", ast.Mod: "%", ast.Pow: "**"}
        op = op_map.get(type(node.op))
        if op is None:
            raise CompileError(f"不支持算子 {type(node.op).__name__}")
        return f"({_unparse_expr(node.left)} {op} {_unparse_expr(node.right)})"
    if isinstance(node, ast.UnaryOp):
        op_map = {ast.UAdd: "+", ast.USub: "-", ast.Not: "not "}
        op = op_map.get(type(node.op))
        if op is None:
            raise CompileError(f"不支持一元算子 {type(node.op).__name__}")
        return f"({op}{_unparse_expr(node.operand)})"
    if isinstance(node, ast.Compare):
        if len(node.ops) > 1:
            raise CompileError("不支持链式比较")
        op_map = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
                  ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
        if ast.LtE not in op_map and hasattr(ast, "ltE"):
            op_map = {ast.Lt: "<", ast.ltE: "<=", ast.Gt: ">",
                      ast.gtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
        op = op_map.get(type(node.ops[0]))
        if op is None:
            raise CompileError(f"不支持比较 {type(node.ops[0]).__name__}")
        return f"({_unparse_expr(node.left)} {op} {_unparse_expr(node.comparators[0])})"
    if isinstance(node, ast.BoolOp):
        kw = " and " if isinstance(node.op, ast.And) else " or "
        return "(" + kw.join(_unparse_expr(v) for v in node.values) + ")"
    raise CompileError(f"无法 unparse expr {type(node).__name__}: {ast.dump(node)}")


def render_cuda_body(strategy_class, method_name: str = "compute_signal") -> str:
    """把 DSL 体渲染成 CUDA C99 函数体 (调试视图; 实际注入用 render_cuda_device_function)

    输出与 render_numba_body 字段名 + 表达式字面完全一致, 只做语法转换:
      - ctx.xxx   -> 内核状态名 (映射表 _CTX_TO_KERNEL; 未映射 -> 局部变量名)
      - True/False -> 1/0  (C99 没有 bool)
      - and/or/not -> &&/||/! (C99)
      - 整数字面量在算术表达式里加 .0 当 double 算
      - 局部变量 (low_dev/signal 等) 首赋值处自动 int/double 声明
    """
    body = _extract_dsl_body(strategy_class, method_name)
    wrapped = f"def _f(ctx):\n{textwrap.indent(body, '    ')}"
    tree = ast.parse(wrapped)
    _validate(tree)
    return _render_cuda_state_body(tree)


def _render_cuda_state_body(tree: ast.Module) -> str:
    """AST -> C99 函数体 (两遍: 局部变量 int/double 推断 -> 逐语句渲染)

    语义:
      - ctx.xxx 按映射表 _CTX_TO_KERNEL 改成内核状态名; 未映射的 ctx.x 与裸名 x
        是局部变量, 首赋值处声明 (纯整数字面量 -> int, 否则 double)
      - True/False -> 1/0, and/or/not -> &&/||/!, 整数字面量加 .0 (比较右值除外)
      - 嵌套 if 的花括号/缩进由 _c99_stmt_state 递归处理
    """
    fn = tree.body[0]

    # 第一遍: 收集局部变量全部赋值的 RHS -> int/double 推断
    rhs_by_name: dict = {}

    def _collect(n) -> None:
        if isinstance(n, ast.If):
            for s in n.body:
                _collect(s)
            for s in n.orelse:
                _collect(s)
        elif isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Attribute):
                field = _c99_attr(t)
            elif isinstance(t, ast.Name):
                field = t.id
            else:
                return
            if field not in _CUDA_SIG_FIELDS:
                rhs_by_name.setdefault(field, []).append(_c99_expr(n.value))

    for stmt in fn.body:
        _collect(stmt)
    decl_kinds = {
        name: ("int" if all(re.fullmatch(r"-?\d+", re.sub(r"[()]", "", v))
                            for v in rhss) else "double")
        for name, rhss in rhs_by_name.items()
    }

    # 第二遍: 逐语句渲染 (内核字段已在签名里, 视为已声明)
    declared = set(_CUDA_SIG_FIELDS)
    lines = [_c99_stmt_state(s, declared, decl_kinds) for s in fn.body]
    return "\n".join(line for line in lines if line)


def _c99_attr(node: ast.Attribute) -> str:
    """ctx.xxx -> 内核状态名 (映射表 _CTX_TO_KERNEL; 未映射字段视为局部变量)"""
    if isinstance(node.value, ast.Name) and node.value.id == "ctx":
        return _CTX_TO_KERNEL.get(node.attr, node.attr)
    raise CompileError("CUDA DSL 仅支持 ctx.xxx 形式访问")


def _c99_expr(node) -> str:
    """表达式节点 -> C99 字符串 (递归)"""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, int):
            # 整数字面量: 加 .0 让 C99 当 double 算
            return f"{v}.0" if v not in (0, 1, -1) else f"{v}"
        if isinstance(v, float):
            # 保留原浮点字面量
            return repr(v)
        raise CompileError(f"CUDA DSL 不支持字面量 {v!r}")
    if isinstance(node, ast.Name):
        if node.id in ("True", "False", "None"):
            return "1" if node.id == "True" else "0"
        return node.id
    if isinstance(node, ast.Attribute):
        return _c99_attr(node)
    if isinstance(node, ast.Call):
        # 白名单 (min/max/abs); _validate 已拒绝关键字参数
        if not (isinstance(node.func, ast.Name)
                and node.func.id in _CALL_WHITELIST):
            raise CompileError(f"CUDA DSL 不支持调用 {type(node.func).__name__}")
        if node.func.id == "abs" and len(node.args) == 1:
            x = _c99_expr(node.args[0])
            return f"(({x}) < 0 ? -({x}) : ({x}))"
        if node.func.id in ("min", "max") and len(node.args) == 2:
            a = _c99_expr(node.args[0])
            b = _c99_expr(node.args[1])
            op = "<" if node.func.id == "min" else ">"
            return f"(({a}) {op} ({b}) ? ({a}) : ({b}))"
        raise CompileError(f"CUDA DSL 仅支持 min(a,b) / max(a,b) / abs(a)")
    if isinstance(node, ast.BinOp):
        op = node.op
        op_map = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.Mod: "%",
        }
        if type(op) not in op_map:
            raise CompileError(f"CUDA DSL 不支持算子 {type(op).__name__}")
        return f"({_c99_expr(node.left)} {op_map[type(op)]} {_c99_expr(node.right)})"
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return f"(!{_c99_expr(node.operand)})"
        if isinstance(node.op, ast.USub):
            return f"(-{_c99_expr(node.operand)})"
        raise CompileError(f"CUDA DSL 不支持一元算子 {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        # 单比较链: a < b < c 拆成 (a<b) && (b<c)
        if len(node.ops) > 1:
            raise CompileError("CUDA DSL 不支持链式比较 (a < b < c)")
        op_map = {
            ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
            ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
        }
        op = type(node.ops[0])
        if op not in op_map:
            raise CompileError(f"CUDA DSL 不支持比较 {op.__name__}")
        return f"({_c99_expr(node.left)} {op_map[op]} {_c99_expr(node.comparators[0])})"
    if isinstance(node, ast.BoolOp):
        kw = "&&" if isinstance(node.op, ast.And) else "||"
        return f"({kw.join(_c99_expr(v) for v in node.values)})"
    raise CompileError(f"CUDA DSL 不支持表达式节点 {type(node).__name__}: {ast.unparse(node)}")


def compile_all(strategy_class, method_name: str = "compute_signal") -> dict:
    """便利入口: 返回 {python, numba, cuda} 三份源码 (调试 / 验证用)"""
    return {
        "python": _extract_dsl_body(strategy_class, method_name),
        "numba": render_numba_body(strategy_class, method_name),
        "cuda": render_cuda_body(strategy_class, method_name),
    }


# ====================================================================
# 内核状态映射 (步骤 2): DSL ctx 字段 -> 内核侧名字
# ====================================================================
# key   = DSL 里写的 ctx.<name>
# value = 内核侧名字 (numba 端渲染成 st.<name>, CUDA 端为裸名);
#         up/dw 是 _strategy_check / strategy_check 的函数参数 (两端都为裸名)。
# 不在表内的 ctx.<name> 与裸名 x 都是策略局部变量 (内核端首赋值处声明,
# 首次赋值前不可读)。
_CTX_TO_KERNEL = {
    "cur_ts": "cur_ts", "cur_open": "cur_open", "cur_high": "cur_high",
    "cur_low": "cur_low", "cur_close": "cur_close", "cur_volume": "cur_volume",
    "up": "up", "dw": "dw",
    "low_hit": "low_hit", "high_hit": "high_hit",
    "_low_acted": "low_acted", "_high_acted": "high_acted",
    "_bucket_ts": "lock_ts",
}
for _i in range(16):
    _CTX_TO_KERNEL[f"p{_i}"] = f"p{_i}"

# numba 端不加 st. 前缀的字段 (_strategy_check 的函数参数)
_KERNEL_BARE = frozenset({"up", "dw"})

# CUDA device 函数签名实际存在的字段 (模板寄存器/状态有限, 超出即拒绝)
_CUDA_SIG_FIELDS = frozenset(
    {"low_hit", "high_hit", "low_acted", "high_acted", "lock_ts",
     "cur_ts", "cur_open", "cur_high", "cur_low", "cur_close", "cur_volume",
     "up", "dw"}
    | {f"p{i}" for i in range(8)}
)


def render_numba_state_body(strategy_class, method_name: str = "compute_signal") -> str:
    """DSL -> kernel._strategy_check 函数体 (st 形式, kernel_dsl.splice 用)

    render_numba_body 的产物再做 ctx -> 内核状态改名:
      ctx.up/ctx.dw   -> up / dw      (函数参数)
      ctx.<表内字段>  -> st.<内核名>  (jitclass 字段)
      其余 ctx.x / 裸名 x -> 局部变量 (名字不变)
    表达式字面顺序保持不变 (浮点路径与 Python 端一致)。
    """
    body = render_numba_body(strategy_class, method_name)

    def _sub(m: "re.Match") -> str:
        name = m.group(1)
        k = _CTX_TO_KERNEL.get(name)
        if k is None:
            return name              # 未映射 ctx 字段 = 局部变量
        if k in _KERNEL_BARE:
            return k
        return f"st.{k}"

    return re.sub(r"\bctx\.([A-Za-z_]\w*)", _sub, body)


# CUDA device 函数: 状态字段按引用传入 (可写), 行情/参数为 const 引用。
# return 语义与 DSL 完全一致 (device 函数返回即跳过本桶剩余策略逻辑)。
_CUDA_DEVICE_HEADER = """\
__device__ __forceinline__ int strategy_check(
    int &low_hit, int &high_hit, int &low_acted, int &high_acted,
    long long &lock_ts,
    const long long &cur_ts,
    const double &cur_open, const double &cur_high,
    const double &cur_low, const double &cur_close, const double &cur_volume,
    const double &up, const double &dw,
    const double &p0, const double &p1, const double &p2, const double &p3,
    const double &p4, const double &p5, const double &p6, const double &p7)
{"""


def _c99_stmt_state(node, declared: set, decl_kinds: dict) -> str:
    """语句 -> C99 (内核状态映射版): 局部变量首赋值处按 decl_kinds 声明"""
    if isinstance(node, ast.If):
        cond = _c99_expr(node.test)
        lines = [f"if ({cond}) {{"]
        for s in node.body:
            lines.append("  " + _c99_stmt_state(s, declared, decl_kinds))
        lines.append("}")
        if node.orelse:
            lines.append("else {")
            for s in node.orelse:
                lines.append("  " + _c99_stmt_state(s, declared, decl_kinds))
            lines.append("}")
        return "\n".join(lines)
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise CompileError("CUDA DSL 不允许多元赋值")
        target = node.targets[0]
        val = _c99_expr(node.value)
        if isinstance(target, ast.Attribute):
            field = _c99_attr(target)
        elif isinstance(target, ast.Name):
            field = target.id
        else:
            raise CompileError("CUDA DSL 仅支持 ctx.x = v 或 x = v 赋值")
        if field in declared:
            return f"{field} = {val};"
        declared.add(field)
        return f"{decl_kinds.get(field, 'double')} {field} = {val};"
    if isinstance(node, ast.Return):
        return f"return {_c99_expr(node.value)};"
    raise CompileError(f"CUDA DSL 不支持语句 {type(node).__name__}")


def render_cuda_device_function(strategy_class, method_name: str = "compute_signal") -> str:
    """DSL -> CUDA __device__ int strategy_check(...) 整函数 (gpu.py 通用模板注入)

    与 render_cuda_body 的区别: 外层包 __device__ 函数 ——
      - DSL 的 return X 直接成为函数返回, "提前返回跳过后续语句" 的语义
        与 Python/numba 端完全一致
      - 引用了 CUDA 签名之外的内核字段 (cur_mark/p8..p15) 时编译期即报错
      - params_spec 长度 > 8 时编译期即报错 (CUDA 通用模板只有 p0..p7 寄存器);
        CPU/numba 路径支持到 p15 (见 kernel_dsl.make_state_general 的 16 上限),
        想用更多参数必须拆分策略或走 CPU 路径。
    """
    body = _extract_dsl_body(strategy_class, method_name)
    wrapped = f"def _f(ctx):\n{textwrap.indent(body, '    ')}"
    tree = ast.parse(wrapped)
    _validate(tree)
    fn = tree.body[0]

    # 能力校验: 参数个数必须在 CUDA 通用模板上限 (p0..p7) 之内
    cls = strategy_class if isinstance(strategy_class, type) else type(strategy_class)
    n_params = len(getattr(cls, "params_spec", {}) or {})
    if n_params > 8:
        # 提早抛: 包含 p8 字样让既有用例 (字段引用校验) 的 pattern 仍命中
        raise CompileError(
            f"DSL 策略 {cls.__name__} 有 {n_params} 个参数, "
            f"超过 CUDA 通用内核上限 8 (p0..p7); p8..p15 在 CUDA 端不存在, "
            f"请用 CPU/numba 路径或拆分策略。"
        )

    # 引用校验: 映射后仍在 CUDA 签名之外的字段直接拒绝 (模板无此寄存器)
    used = {n.attr for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "ctx"}
    overflow = sorted(
        _CTX_TO_KERNEL.get(a, a) for a in used
        if a in _CTX_TO_KERNEL and _CTX_TO_KERNEL[a] not in _CUDA_SIG_FIELDS)
    if overflow:
        raise CompileError(
            f"CUDA 通用内核 ctx 契约不含 {overflow} (仅支持 "
            f"cur_ts/cur_open/cur_high/cur_low/cur_close/cur_volume/up/dw/"
            f"low_hit/high_hit/_low_acted/_high_acted/_bucket_ts/p0..p7)")

    return _CUDA_DEVICE_HEADER + "\n" + _render_cuda_state_body(tree) + "\n}"


# ====================================================================
# Python 端通用 DSL 上下文 (新策略复用; channel_deviation 自带 ctx 保持兼容)
# ====================================================================

class DSLCtx:
    """通用 DSL 上下文: 字段 = 内核状态契约 (与 _CTX_TO_KERNEL 一致)

    状态字段 (low_hit/_bucket_ts/...) 在 ctx 上跨桶持久, 由策略实例持有;
    DSL 里 ctx.low_dev 等临时字段会作为普通属性动态挂上 (无 __slots__ 限制)。
    """

    def __init__(self):
        self.cur_ts = 0
        self.cur_open = 0.0
        self.cur_high = 0.0
        self.cur_low = 0.0
        self.cur_close = 0.0
        self.cur_volume = 0.0
        self.up = 0.0
        self.dw = 0.0
        for i in range(16):
            setattr(self, f"p{i}", 0.0)
        self.low_hit = False
        self.high_hit = False
        self._bucket_ts = None
        self._low_acted = False
        self._high_acted = False


def dsl_check(strategy, ctx: DSLCtx, cur: dict, up, dw) -> int:
    """Python 端执行策略 DSL: 填 ctx -> 跑 compute_signal docstring -> 信号 int

    策略实例持有同一个 ctx (状态跨桶持久); DSL 首次调用时编译一次并缓存。
    """
    runner = getattr(strategy, "_dsl_runner", None)
    if runner is None:
        runner = make_python_runner(strategy, "compute_signal")
        strategy._dsl_runner = runner
    ctx.cur_ts = cur["ts"]
    ctx.cur_open = float(cur.get("open", 0.0))
    ctx.cur_high = float(cur["high"])
    ctx.cur_low = float(cur["low"])
    ctx.cur_close = float(cur["close"])
    ctx.cur_volume = float(cur.get("volume", 0.0))
    ctx.up = up
    ctx.dw = dw
    for i, k in enumerate(strategy.params_spec):
        setattr(ctx, f"p{i}", float(strategy.params[k]))
    return int(runner(ctx))
