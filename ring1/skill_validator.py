"""Security validator for skills downloaded from the Hub.

Performs static analysis on skill source code to detect dangerous patterns
before allowing installation.  Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Dangerous pattern definitions
# ---------------------------------------------------------------------------

# Patterns that indicate potentially dangerous code.  Each entry is
# (compiled_regex, human-readable reason).
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Command execution
    (re.compile(r"\bos\.system\s*\("), "os.system() — arbitrary command execution"),
    (re.compile(r"\bos\.popen\s*\("), "os.popen() — arbitrary command execution"),
    (re.compile(r"\bos\.exec[lv]p?e?\s*\("), "os.exec*() — process replacement"),
    (re.compile(r"\bsubprocess\.\w+\s*\("), "subprocess — arbitrary command execution"),
    (re.compile(r"\bcommands\.\w+\s*\("), "commands module — command execution"),

    # Dynamic code execution
    (re.compile(r"\beval\s*\("), "eval() — arbitrary code execution"),
    (re.compile(r"\bexec\s*\("), "exec() — arbitrary code execution"),
    (re.compile(r"\bcompile\s*\(.+['\"]exec['\"]"), "compile() with exec mode"),
    (re.compile(r"\b__import__\s*\("), "__import__() — dynamic import"),

    # Filesystem destruction
    (re.compile(r"\bshutil\.rmtree\s*\("), "shutil.rmtree() — recursive deletion"),
    (re.compile(r"\bos\.remove\s*\("), "os.remove() — file deletion"),
    (re.compile(r"\bos\.unlink\s*\("), "os.unlink() — file deletion"),
    (re.compile(r"\bos\.rmdir\s*\("), "os.rmdir() — directory deletion"),
    (re.compile(r"\bos\.removedirs\s*\("), "os.removedirs() — recursive dir deletion"),

    # Sensitive file access
    (re.compile(r"""['"](/etc/passwd|/etc/shadow|~?/\.ssh|~?/\.env|~?/\.aws)"""),
     "access to sensitive system files"),
    (re.compile(r"""['"]\S*(credentials|secrets?|password|private.key)\S*['"]""", re.IGNORECASE),
     "access to credential/secret files"),

    # Network exfiltration (non-localhost)
    (re.compile(r"\bsocket\.socket\s*\("), "raw socket creation"),
    (re.compile(r"\bsmtplib\b"), "SMTP — email sending"),
    (re.compile(r"\bftplib\b"), "FTP — file transfer"),

    # Privilege escalation
    (re.compile(r"\bos\.setuid\s*\("), "os.setuid() — privilege escalation"),
    (re.compile(r"\bos\.setgid\s*\("), "os.setgid() — privilege escalation"),
    (re.compile(r"\bctypes\b"), "ctypes — low-level system access"),
]

# Patterns that are suspicious but not outright blocked — logged as warnings.
_WARNING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bopen\s*\([^)]*['\"]w"), "writing to files"),
    (re.compile(r"\burllib\.request\.urlopen\s*\("), "outbound HTTP request"),
    (re.compile(r"\bhttp\.client\b"), "outbound HTTP via http.client"),
    (re.compile(r"\bpickle\.loads?\s*\("), "pickle deserialization (potential RCE)"),
]


class ValidationResult:
    """Result of skill security validation."""

    __slots__ = ("safe", "errors", "warnings")

    def __init__(self) -> None:
        self.safe: bool = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def __repr__(self) -> str:
        status = "SAFE" if self.safe else "BLOCKED"
        parts = [f"ValidationResult({status}"]
        if self.errors:
            parts.append(f", errors={self.errors}")
        if self.warnings:
            parts.append(f", warnings={self.warnings}")
        parts.append(")")
        return "".join(parts)


def validate_skill(source_code: str) -> ValidationResult:
    """Validate skill source code for security risks.

    Returns a ValidationResult with:
    - safe=True: code passed static analysis, OK to install
    - safe=False: code contains dangerous patterns, should be rejected

    Warnings are informational and do not block installation.
    """
    result = ValidationResult()

    if not source_code or not source_code.strip():
        result.errors.append("empty source code")
        result.safe = False
        return result

    # Strip comments and strings to reduce false positives.
    # We check against the raw source to catch patterns in strings too
    # (a malicious skill might hide code in string-based eval).

    for pattern, reason in _DANGEROUS_PATTERNS:
        matches = pattern.findall(source_code)
        if matches:
            result.errors.append(reason)
            result.safe = False

    for pattern, reason in _WARNING_PATTERNS:
        matches = pattern.findall(source_code)
        if matches:
            result.warnings.append(reason)

    return result
