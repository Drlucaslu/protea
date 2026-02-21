"""Shell exec tool — run commands in a subprocess with safety guards.

Pure stdlib.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.shell")

_MAX_OUTPUT = 50_000  # truncate output to 50K chars

# Patterns that are too dangerous to execute.
_DENY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-[^\s]*r[^\s]*f\b.*\s+/\s*$"),  # rm -rf /
    re.compile(r"\brm\s+-[^\s]*f[^\s]*r\b.*\s+/\s*$"),  # rm -fr /
    re.compile(r"\bdd\s+"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\binit\s+[06]\b"),
    re.compile(r":\s*\(\s*\)\s*\{"), # fork bomb :(){ :|:& };:
    re.compile(r"\bcurl\b.*\|\s*(?:ba)?sh\b"),
    re.compile(r"\bwget\b.*\|\s*(?:ba)?sh\b"),
    re.compile(r"\bchmod\s+.*777\s+/"),
    re.compile(r"\bchown\s+.*\s+/\s*$"),
    re.compile(r">\s*/dev/[sh]da"),
    re.compile(r"\bsystemctl\s+(?:stop|disable|mask)\b"),
    # Process lifecycle — Sentinel/CommitWatcher manage these automatically.
    re.compile(r"\bgit\s+(?:pull|push|reset|rebase|merge)\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\bkillall\b"),
    re.compile(r"\bnohup\b"),
    re.compile(r"\bpython[23]?\s+.*\brun\.py\b"),
    re.compile(r"\bstop_run\b"),        # stop script
    re.compile(r"\bsignal\.SIGTERM\b"),
    re.compile(r"\bos\.kill\b"),
]


_KILL_RE = re.compile(r"\bkill\b")


def _is_descendant_of(pid: int, ancestor: int) -> bool:
    """Return True if *pid* is a descendant of *ancestor* in the process tree."""
    current = pid
    for _ in range(64):  # guard against loops
        if current == ancestor:
            return True
        if current <= 1:
            return False
        try:
            out = subprocess.check_output(
                ["ps", "-o", "ppid=", "-p", str(current)],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            current = int(out)
        except (subprocess.CalledProcessError, ValueError):
            return False
    return False


def _in_same_pgid(pid: int, reference: int) -> bool:
    """Return True if *pid* belongs to the same process group as *reference*."""
    try:
        return os.getpgid(pid) == os.getpgid(reference)
    except OSError:
        return False


def _is_safe_kill(command: str) -> str | None:
    """Allow ``kill`` only for processes within the Protea process tree.

    A target PID is allowed if it is either:
    * a descendant of the current process (normal child/grandchild), **or**
    * in the same process group (covers orphaned siblings from a previous
      Sentinel restart — PGID is inherited and survives reparenting to init).

    Returns a reason string if the command should be blocked, else ``None``.
    """
    # Reject complex forms we cannot safely parse.
    if re.search(r"[|`]|\$\(", command):
        return "Blocked: kill with pipes/subshells is not allowed"

    # Tokenise: split on && / ; / || to handle chained commands, then
    # inspect only the segments that contain a bare `kill`.
    segments = re.split(r"&&|;|\|\|", command)
    for segment in segments:
        segment = segment.strip()
        if not _KILL_RE.search(segment):
            continue
        tokens = segment.split()
        # Find the `kill` token and collect PID args after it.
        try:
            idx = next(i for i, t in enumerate(tokens) if t == "kill")
        except StopIteration:
            continue
        pids: list[int] = []
        for token in tokens[idx + 1 :]:
            if token.startswith("-"):
                continue  # skip signal flags like -9, -TERM
            if token.isdigit():
                pids.append(int(token))
            else:
                return f"Blocked: cannot verify kill target '{token}'"
        if not pids:
            return "Blocked: kill with no PID"
        my_pid = os.getpid()
        for pid in pids:
            if _in_same_pgid(pid, my_pid):
                continue
            if _is_descendant_of(pid, my_pid):
                continue
            return f"Blocked: PID {pid} is not in the Protea process tree"
    return None


def _is_denied(command: str) -> str | None:
    """Return a reason string if *command* matches a deny pattern, else None."""
    for pattern in _DENY_PATTERNS:
        if pattern.search(command):
            return f"Blocked: command matches deny pattern ({pattern.pattern})"
    # Special handling: allow `kill` only for descendant PIDs.
    if _KILL_RE.search(command):
        return _is_safe_kill(command)
    return None


def make_shell_tool(workspace_path: str, timeout: int = 30) -> Tool:
    """Create a Tool instance for shell command execution."""

    def _exec_shell(inp: dict) -> str:
        command = inp["command"]
        cmd_timeout = inp.get("timeout", timeout)

        # Safety check
        reason = _is_denied(command)
        if reason:
            log.warning("Shell deny: %s — %s", command, reason)
            return reason

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=cmd_timeout,
                cwd=workspace_path,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {cmd_timeout}s"
        except Exception as exc:
            return f"Error: {exc}"

        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            parts.append(f"[exit code: {result.returncode}]")

        output = "\n".join(parts) if parts else "(no output)"

        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + "\n... (truncated)"

        return output

    return Tool(
        name="exec",
        description=(
            "Execute a shell command and return its output. The command runs "
            "in the workspace directory. Generated output files should be saved "
            "to the output/ subdirectory. Dangerous commands (rm -rf /, dd, "
            "mkfs, shutdown, etc.) are blocked. Code updates are handled "
            "automatically by CommitWatcher — do NOT run git pull/push/reset "
            "or kill/restart processes. Use git log/status/diff for read-only "
            "git operations."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default {timeout}).",
                },
            },
            "required": ["command"],
        },
        execute=_exec_shell,
    )
