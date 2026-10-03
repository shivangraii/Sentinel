"""Safe arithmetic evaluator (AST based; no eval of arbitrary code)."""
from __future__ import annotations

import ast
import math
import operator as op

from sentinel.tools.base import EvidenceItem, Tool, ToolResult

_BIN = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs,
    "round": round, "min": min, "max": max, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "ceil": math.ceil, "floor": math.floor,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, r = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(r) > 1000:
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](left, r)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> ToolResult:
    expr = str(expression).strip().replace("^", "**")
    value = _eval(ast.parse(expr, mode="eval"))
    if isinstance(value, float):
        value = round(value, 10)
    text = f"{expression} = {value}"
    return ToolResult(text, [EvidenceItem(text=text, source="calculator", kind="calc", title="Calculation")])


CALCULATOR = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression (+ - * / ** %, sqrt, log, exp, min, max, round, pi, e).",
    args='expression: str',
    fn=calculate,
)
