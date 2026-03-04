"""Ring 2 code validation and extraction utilities.

Extracted from evolver.py and prompts.py during the evolution→reflection
refactoring.  These functions are used by the reflection system and
sentinel to validate Ring 2 code mutations.
"""

from __future__ import annotations

import ast
import re


def validate_ring2_code(source: str) -> tuple[bool, str]:
    """Pre-deployment validation of Ring 2 code.

    Checks:
    1. Compiles without syntax errors
    2. Contains heartbeat mechanism (PROTEA_HEARTBEAT)
    3. Has a main() function
    """
    try:
        compile(source, "<ring2>", "exec")
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    if "PROTEA_HEARTBEAT" not in source:
        return False, "Missing PROTEA_HEARTBEAT reference"

    if "def main" not in source:
        return False, "Missing main() function"

    return True, "OK"


def compress_source(code: str) -> str:
    """Strip docstrings, comments, and excess blank lines to save tokens."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)):
                ds = node.body[0]
                for ln in range(ds.lineno, ds.end_lineno + 1):
                    docstring_lines.add(ln)

    lines = code.splitlines()
    result: list[str] = []
    prev_blank = False
    for i, line in enumerate(lines, 1):
        if i in docstring_lines:
            continue
        stripped = line.strip()
        if stripped.startswith('#') and not stripped.startswith('#!'):
            continue
        if not stripped:
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        result.append(line.rstrip())

    return '\n'.join(result)


def extract_python_code(response: str) -> str | None:
    """Extract Python code from an LLM response with 3-tier fallback.

    Tier 1: ```python / ```py fenced block (preferred).
    Tier 2: Bare ``` fenced block (no language tag) — validated with compile().
    Tier 3: Whole response as code — strip leading prose, validate with
            compile() + must contain "PROTEA_HEARTBEAT" and "def main".

    Returns None if no valid code is found.
    """
    pattern = r"```(?:python|py)\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
        if code:
            return code

    bare_pattern = r"```\s*\n(.*?)```"
    match = re.search(bare_pattern, response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            try:
                compile(code, "<extract>", "exec")
                return code
            except SyntaxError:
                pass

    lines = response.strip().splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "#!")):
            start = i
            break
    if start > 0 or lines:
        candidate = "\n".join(lines[start:]).strip()
        if candidate and "PROTEA_HEARTBEAT" in candidate and "def main" in candidate:
            try:
                compile(candidate, "<extract>", "exec")
                return candidate
            except SyntaxError:
                pass

    return None
