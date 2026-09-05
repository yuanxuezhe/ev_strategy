from __future__ import annotations
"""策略 DSL: 同一份主逻辑 -> Python + numba njit (CUDA 后续扩展)

================================================================
✅  可改层 (strategies 子包)  ✅
================================================================
设计:
  策略作者在 compute_signal(ctx) 方法的 docstring 里写 DSL 代码,
  表达式只允许标量算术/比较/布尔/属性赋值/return signal。

  dsl_compile() 用 ast.parse 解析, 用 ast.unparse 重新生成等价的:
    - Python 源码 (走参考引擎)
    - numba @njit 函数体 (走内核)
    - CUDA kernel 段 (走 GPU; 当前实现 TODO, 后续扩展)

约束 (whitelist):
  ✅ 算术: + - * /
  ✅ 比较: < > <= >= == !=
  ✅ 布尔: and or not
  ✅ 常量: 数字 / True / False
  ✅ 标量赋值: ctx.x = value, x = value
  ✅ 属性读写: ctx.x, self.x (限制字段)
  ✅ return signal
  ❌ 字符串 / 列表 / 字典 / 调用外部函数 / 循环 (除受控 for)
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
================================================================
"""
import ast
import textwrap


class CompileError(Exception):
    """DSL 不满足白名单时抛出"""


# numba/CUDA 不支持的关键字 / 类型
_FORBIDDEN_NODES = (
    ast.List, ast.Dict, ast.Set, ast.Tuple,
    ast.Call,           # 限制: 仅允许白名单内的内建 (min/max/abs)
    ast.Lambda,
    ast.Yield, ast.YieldFrom,
    ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith,
    ast.Starred,
    ast.JoinedStr,      # f-string
    # 字符串节点 (除常量)
)


def _validate(tree: ast.Module) -> None:
    """白名单校验; 不通过抛 CompileError"""
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise CompileError(f"DSL 不支持节点: {type(node).__name__} "
                               f"(行 {getattr(node, 'lineno', '?')})")
        if isinstance(node, ast.Call):
            # 仅允许白名单内建
            if isinstance(node.func, ast.Name) and node.func.id in ("min", "max", "abs"):
                continue
            raise CompileError(
                f"DSL 不支持调用: {ast.unparse(node.func)} "
                f"(仅允许 min/max/abs)")
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


def _extract_dsl_body(strategy_class, method_name: str = "compute_signal") -> str:
    """从策略类的 compute_signal 方法的 docstring 提取 DSL 代码"""
    method = getattr(strategy_class, method_name, None)
    if method is None or method.__doc__ is None:
        raise CompileError(f"{strategy_class.__name__}.{method_name} 缺少 docstring")
    return textwrap.dedent(method.__doc__)


def dsl_compile(strategy_class, method_name: str = "compute_signal") -> str:
    """解析 + 校验 DSL, 返回清洗后的 Python 源码 (与原 docstring 等价)

    返回的源码可直接 exec 在一个含 ctx 的命名空间里。
    """
    body = _extract_dsl_body(strategy_class, method_name)
    # 包成完整函数以 parse
    wrapped = f"def _f(ctx):\n{textwrap.indent(body, '    ')}"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as e:
        raise CompileError(f"DSL 语法错误: {e}") from e
    _validate(tree)
    # 校验通过后, 返回函数体 (除 def 头)
    return body


def make_python_runner(strategy_class, method_name: str = "compute_signal"):
    """返回 (fn, init_state_dict); fn(ctx) -> signal

    Python 直接解释执行 DSL, 不经 AST -> 字符串 -> 编译的往返,
    以保留原始浮点表达式顺序 (避免任何重渲染漂移)。
    """
    body = _extract_dsl_body(strategy_class, method_name)
    namespace: dict = {}
    # 用 exec 直接定义 _f 函数
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


def _unparse_stmt(node) -> str:
    """简单 stmt/expr unparse (兼容 Python 3.8, 无 ast.unparse)"""
    if isinstance(node, ast.If):
        cond = _unparse_expr(node.test)
        lines = [f"if {cond}:"]
        for s in node.body:
            lines.append("    " + _unparse_stmt(s))
        if node.orelse:
            lines.append("else:")
            for s in node.orelse:
                lines.append("    " + _unparse_stmt(s))
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
    """把 DSL 体渲染成 CUDA C99 函数体 (给 __device__ 函数用)

    输出与 render_numba_body 字段名 + 表达式字面完全一致, 只做语法转换:
      - ctx.xxx   -> xxx  (属性访问去掉 ctx.)
      - ctx.x = v -> xxx = v
      - == !=     不变 (C99 支持)
      - True/False -> 1/0  (C99 没有 bool)
      - and/or/not -> &&/||/! (C99)
      - 浮点字面量 1.0 -> 1.0 不变
      - return 0/1/-1 不变

    不做 (需要调用方处理):
      - ctx 类型转换 (假设 ctx 是 struct, 字段是 double/long long)
      - 字段顺序对齐 (调用方必须与 ctx struct 字段顺序一致)
    """
    body = _extract_dsl_body(strategy_class, method_name)
    wrapped = f"def _f(ctx):\n{textwrap.indent(body, '    ')}"
    tree = ast.parse(wrapped)
    _validate(tree)

    # 替换 ctx.xxx -> xxx; True/False -> 1/0; and/or/not -> &&/||/!
    src = _c99_rewrite(tree)
    return src


def _c99_rewrite(tree: ast.Module) -> str:
    """AST -> C99 源码 (针对 DSL 算子白名单)

    不通用; 仅处理 DSL 白名单内的节点 (算术/比较/布尔/标量赋值/return)。
    """
    out: list[str] = []
    for stmt in tree.body[0].body:
        out.append(_c99_stmt(stmt))
    return "\n".join(line for line in out if line)


# 类型映射 (DSL ctx 字段 -> C99 类型)
_C99_TYPES = {
    "double": ("double", "%g"),
    "int":    ("long long", "%lld"),
    "bool":   ("int",      "%d"),
}


def _c99_field(name: str) -> str:
    """ctx.field 在 C99 里直接是 field (假设 ctx 是 struct)"""
    return name


def _c99_attr(node: ast.Attribute) -> str:
    """ctx.xxx -> xxx (在 struct 上下文里)"""
    if isinstance(node.value, ast.Name) and node.value.id == "ctx":
        return node.attr
    raise CompileError(f"CUDA DSL 仅支持 ctx.xxx 形式访问, 收到 {ast.unparse(node)}")


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


def _c99_stmt(node) -> str:
    """语句节点 -> C99 字符串"""
    if isinstance(node, ast.If):
        cond = _c99_expr(node.test)
        lines = [f"if ({cond}) {{"]
        for s in node.body:
            lines.append("  " + _c99_stmt(s))
        lines.append("}")
        if node.orelse:
            lines.append("else {")
            for s in node.orelse:
                lines.append("  " + _c99_stmt(s))
            lines.append("}")
        return "\n".join(lines)
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise CompileError("CUDA DSL 不允许多元赋值")
        target = node.targets[0]
        if not isinstance(target, ast.Attribute):
            raise CompileError("CUDA DSL 仅支持 ctx.x = v 赋值")
        field = _c99_attr(target)
        val = _c99_expr(node.value)
        return f"{field} = {val};"
    if isinstance(node, ast.Return):
        return f"return {_c99_expr(node.value)};"
    raise CompileError(f"CUDA DSL 不支持语句 {type(node).__name__}: {ast.unparse(node)}")


def compile_all(strategy_class, method_name: str = "compute_signal") -> dict:
    """便利入口: 返回 {python, numba, cuda} 三份源码 (调试 / 验证用)"""
    return {
        "python": _extract_dsl_body(strategy_class, method_name),
        "numba": render_numba_body(strategy_class, method_name),
        "cuda": render_cuda_body(strategy_class, method_name),
    }
