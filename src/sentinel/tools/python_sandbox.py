"""Best-effort restricted Python execution for quick analysis.

Layers: AST allow-list (imports, no dunder access, no open/exec/eval), restricted builtins and
import hook inside a fresh `python -I` subprocess, CPU/memory rlimits, hard timeout, output cap.

NOTE: this is NOT a security boundary against a determined attacker. For untrusted multi-tenant use,
run the API container with no network/secrets (see docker-compose.yml) or swap in a real sandbox.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile

from sentinel.tools.base import EvidenceItem, Tool, ToolResult

ALLOWED_IMPORTS = {"math", "statistics", "json", "itertools", "collections", "random", "datetime", "re", "decimal", "fractions"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "memoryview", "help", "exit", "quit",
}
TIMEOUT_S = 5
MAX_OUT = 4000


class UnsafeCode(ValueError):
    pass


def validate_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise UnsafeCode(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise UnsafeCode(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCode(f"dunder attribute access not allowed: {node.attr}")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise UnsafeCode(f"name not allowed: {node.id}")


_RUNNER = r'''
import sys, builtins
ALLOWED = __ALLOWED__
_real_import = builtins.__import__
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED:
        raise ImportError("import not allowed: " + name)
    return _real_import(name, globals, locals, fromlist, level)
_names = ("abs all any bool dict divmod enumerate filter float format frozenset int isinstance len list "
          "map max min pow print range repr reversed round set slice sorted str sum tuple zip "
          "Exception ValueError TypeError ZeroDivisionError KeyError IndexError StopIteration").split()
SAFE = {n: getattr(builtins, n) for n in _names}
SAFE["__import__"] = _safe_import
code = sys.stdin.read()
exec(compile(code, "<sandbox>", "exec"), {"__builtins__": SAFE, "__name__": "__main__"})
'''.replace("__ALLOWED__", repr(sorted(ALLOWED_IMPORTS)))


def _limits() -> None:  # pragma: no cover (runs in child)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    except Exception:
        pass


def run_python(code: str) -> ToolResult:
    try:
        validate_code(code)
    except UnsafeCode as e:
        return ToolResult(f"Rejected unsafe code: {e}", error=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER],
                input=code, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=tmp,
                env={"PATH": os.environ.get("PATH", "")},
                preexec_fn=_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"Execution timed out after {TIMEOUT_S}s", error=True)
    out = (proc.stdout or "").strip()[:MAX_OUT]
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        return ToolResult(f"Python error: {err[0][:300]}", error=True)
    if not out:
        return ToolResult("Code ran but printed nothing. Use print() to output results.", error=True)
    ev = EvidenceItem(text=f"Code:\n{code}\nOutput:\n{out}", source="python", kind="python", title="Python result")
    return ToolResult(out, [ev])


PYTHON_TOOL = Tool(
    name="python",
    description=(
        "Run short Python for calculations/statistics. Allowed imports: "
        + ", ".join(sorted(ALLOWED_IMPORTS))
        + ". No file/network access. Must print() the result."
    ),
    args="code: str",
    fn=run_python,
)
