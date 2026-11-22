"""tool_forge.py — Sandboxed self-tool-making runtime for Vox.

Allows Vox to create, list, and execute lightweight Python mini-tools strictly inside
vox_v2/custom_tools/ with:
  - Strict realpath + identifier sandbox (cannot touch core vox_v2 or system files).
  - AST static security gate (blocks subprocess, os.system, shutil, eval/exec, etc.).
  - Isolated subprocess execution (`python3 -I`) with 10s hard timeout & 32KB output cap.
"""

from __future__ import annotations

import ast
import json
import os
import py_compile
import re
import subprocess
import sys
from typing import Any

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")

FORBIDDEN_IMPORTS = {
    "subprocess",
    "shutil",
    "ctypes",
    "multiprocessing",
    "pty",
    "signal",
    "socket",
    "pathlib",
    "importlib",
    "sys",
    "os",
}

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
}

TOOLFORGE_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_custom_tools",
            "description": "列出你自己在沙箱工具库里创建过的轻量 Python 小工具及说明。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_custom_tool",
            "description": "在专属沙箱里新建或更新一个轻量 Python 小工具（必须定义 def run(args: dict) -> dict 函数；允许使用 math/random/json/datetime/re/statistics 等标准计算库，禁止系统命令与文件读写）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "小写英文工具名，如 dinner_picker、loan_calc"},
                    "description": {"type": "string", "description": "一句话说明工具用途与参数"},
                    "code": {"type": "string", "description": "完整 Python 代码，必须包含 def run(args: dict) -> dict:"},
                },
                "required": ["name", "description", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_custom_tool",
            "description": "在独立隔离沙箱子进程中运行你创建的轻量小工具并返回 JSON 结果（10 秒超时）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "要运行的自定义工具名"},
                    "args": {"type": "object", "description": "传给 run(args) 的参数字典"},
                },
                "required": ["name"],
            },
        },
    },
]


def audit_python_ast(code: str, allow_net: bool = False) -> list[str]:
    """Static AST security check; returns list of violations (empty = safe)."""
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]

    has_run = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            has_run = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS or (not allow_net and root in ("urllib", "http", "requests")):
                    violations.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in FORBIDDEN_IMPORTS or (not allow_net and mod in ("urllib", "http", "requests")):
                violations.append(f"Forbidden from-import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                violations.append(f"Forbidden builtin call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
                violations.append(f"Forbidden dunder call: {node.func.attr}")

    if not has_run:
        violations.append("Missing required function: def run(args: dict) -> dict")
    return violations


class ToolForge:
    """Sandboxed custom tool manager for Vox."""

    def __init__(self, tools_dir: str, timeout_s: float = 10.0, allow_net: bool | None = None) -> None:
        self.tools_dir = os.path.realpath(tools_dir)
        os.makedirs(self.tools_dir, exist_ok=True)
        self.timeout_s = timeout_s
        self.allow_net = (os.environ.get("VOX_TOOLFORGE_ALLOW_NET", "0") == "1") if allow_net is None else allow_net
        self._meta_path = os.path.join(self.tools_dir, "_index.json")

    def _safe_tool_path(self, name: str) -> str | None:
        if not NAME_RE.match(name or ""):
            return None
        path = os.path.realpath(os.path.join(self.tools_dir, f"{name}.py"))
        if not path.startswith(self.tools_dir + os.sep):
            return None
        return path

    def _load_index(self) -> dict[str, str]:
        if not os.path.isfile(self._meta_path):
            return {}
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, idx: dict[str, str]) -> None:
        tmp = self._meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._meta_path)

    def list_tools(self) -> dict[str, Any]:
        idx = self._load_index()
        items = []
        for fn in sorted(os.listdir(self.tools_dir)):
            if fn.endswith(".py") and not fn.startswith("_"):
                nm = fn[:-3]
                items.append({"name": nm, "description": idx.get(nm, "")})
        return {"ok": True, "count": len(items), "tools": items}

    def save_tool(self, name: str, description: str, code: str) -> dict[str, Any]:
        path = self._safe_tool_path(name)
        if not path:
            return {"ok": False, "reason": "工具名不合法（需小写字母开头，仅含小写字母/数字/下划线）"}
        violations = audit_python_ast(code, allow_net=self.allow_net)
        if violations:
            return {"ok": False, "reason": "安全或结构检查未通过", "violations": violations}

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(code.rstrip() + "\n")
        try:
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as exc:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return {"ok": False, "reason": f"编译失败: {exc}"}

        os.replace(tmp, path)
        idx = self._load_index()
        idx[name] = (description or "").strip()[:120]
        self._save_index(idx)
        return {"ok": True, "name": name, "description": idx[name]}

    def run_tool(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._safe_tool_path(name)
        if not path or not os.path.isfile(path):
            return {"ok": False, "reason": f"未找到自定义工具: {name}"}

        # Re-audit before execution
        with open(path, "r", encoding="utf-8") as f:
            violations = audit_python_ast(f.read(), allow_net=self.allow_net)
        if violations:
            return {"ok": False, "reason": "安全校验失败", "violations": violations}

        runner = (
            "import importlib.util, json, sys\n"
            "spec = importlib.util.spec_from_file_location('vox_custom_tool', sys.argv[1])\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "args = json.loads(sys.argv[2])\n"
            "res = mod.run(args)\n"
            "print(json.dumps({'ok': True, 'result': res}, ensure_ascii=False))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner, path, json.dumps(args or {}, ensure_ascii=False)],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=self.tools_dir,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": f"工具运行超时（>{self.timeout_s}s）"}

        out = (proc.stdout or "").strip()[:32_768]
        err = (proc.stderr or "").strip()[:1024]
        if proc.returncode != 0:
            return {"ok": False, "reason": err or f"exit_code={proc.returncode}"}
        try:
            # Take last JSON line in case tool printed anything
            last_line = [ln for ln in out.splitlines() if ln.strip()][-1]
            return json.loads(last_line)
        except Exception:
            return {"ok": True, "raw_output": out}

    def exec_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "list_custom_tools":
            return self.list_tools()
        if name == "save_custom_tool":
            return self.save_tool(
                name=str(args.get("name", "")),
                description=str(args.get("description", "")),
                code=str(args.get("code", "")),
            )
        if name == "run_custom_tool":
            return self.run_tool(
                name=str(args.get("name", "")),
                args=args.get("args") if isinstance(args.get("args"), dict) else {},
            )
        return {"ok": False, "reason": f"unknown tool: {name}"}
