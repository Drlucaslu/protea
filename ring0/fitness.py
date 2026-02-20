"""Fitness tracker backed by SQLite.

Records and queries fitness scores for every generation in the
self-evolving lifecycle.  Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import re
import sqlite3

from ring0.sqlite_store import SQLiteStore

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS fitness_log (
    id          INTEGER PRIMARY KEY,
    generation  INTEGER  NOT NULL,
    commit_hash TEXT     NOT NULL,
    score       REAL     NOT NULL,
    runtime_sec REAL     NOT NULL,
    survived    BOOLEAN  NOT NULL,
    timestamp   TEXT     DEFAULT CURRENT_TIMESTAMP
)
"""


# Patterns that indicate actual program errors (tracebacks, Python exceptions).
# Deliberately excludes generic "error" substring to avoid false positives when
# the program is *reporting on* errors (e.g. log analyzers printing "15 errors").
_REAL_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r"^\s+File \".+\", line \d+"),          # traceback frame
    re.compile(r"\b[A-Z]\w*(?:Error|Exception)\s*:"),   # TypeError:, ValueError:, …
]


def _is_real_error_line(line: str) -> bool:
    """Return True only for lines that look like actual program errors."""
    return any(pat.search(line) for pat in _REAL_ERROR_PATTERNS)


_STRUCTURED_PATTERNS = [
    re.compile(r"^\s*[\[{]"),           # JSON array/object start
    re.compile(r"^\s*\|.*\|"),          # Markdown/ASCII table row
    re.compile(r"^[+-]{3,}"),           # table separator
    re.compile(r"^={3,}"),              # section separator
    re.compile(r"^\s*\w+\s*:\s+\S"),   # key: value pairs
]

# Patterns that indicate functional/interactive behaviour (beyond pure output).
_FUNCTIONAL_PATTERNS = [
    re.compile(r"(?:http|https)://", re.IGNORECASE),
    re.compile(r"(?:GET|POST|PUT|DELETE)\s+/", re.IGNORECASE),
    re.compile(r"listening on|server started|bound to port", re.IGNORECASE),
    re.compile(r"wrote .+ bytes|saved to|created file", re.IGNORECASE),
    re.compile(r"connected to|socket|tcp|udp", re.IGNORECASE),
    re.compile(r"api response|fetched|downloaded", re.IGNORECASE),
]


def _output_fingerprint(meaningful_lines: list[str]) -> set[str]:
    """Build a token-level fingerprint of output for novelty comparison.

    Extracts significant tokens (length >= 3, non-numeric) from output
    to enable Jaccard distance computation.
    """
    tokens: set[str] = set()
    for line in meaningful_lines:
        for word in line.split():
            # Keep alphanumeric tokens of length >= 3 that aren't pure numbers.
            cleaned = re.sub(r"[^a-zA-Z0-9_]", "", word).lower()
            if len(cleaned) >= 3 and not cleaned.isdigit():
                tokens.add(cleaned)
    return tokens


def compute_novelty(
    current_tokens: set[str],
    recent_token_sets: list[set[str]],
) -> float:
    """Compute novelty score (0.0–1.0) via average Jaccard distance.

    Returns 1.0 if there are no recent outputs to compare against.
    """
    if not recent_token_sets or not current_tokens:
        return 1.0

    distances = []
    for prev_tokens in recent_token_sets:
        if not prev_tokens:
            distances.append(1.0)
            continue
        intersection = len(current_tokens & prev_tokens)
        union = len(current_tokens | prev_tokens)
        jaccard_sim = intersection / union if union > 0 else 0.0
        distances.append(1.0 - jaccard_sim)

    return sum(distances) / len(distances)


def _count_persistent_errors(output_lines: list[str]) -> list[str]:
    """Extract unique error signatures from output for cross-generation tracking.

    Only counts lines that look like actual program errors (tracebacks,
    Python exceptions), not lines that merely mention the word "error".
    """
    errors: list[str] = []
    for ln in output_lines:
        if not _is_real_error_line(ln):
            continue
        low = ln.lower().strip()
        # Normalize: strip line numbers and memory addresses.
        sig = re.sub(r"0x[0-9a-f]+", "0xADDR", low)
        sig = re.sub(r"line \d+", "line N", sig)
        errors.append(sig)
    return list(dict.fromkeys(errors))[:10]  # unique, max 10


def evaluate_output(
    output_lines: list[str],
    survived: bool,
    elapsed: float,
    max_runtime: float,
    recent_fingerprints: list[set[str]] | None = None,
) -> tuple[float, dict]:
    """Score a Ring 2 run based on output quality.

    Returns (score, detail_dict) where score is 0.0–1.0 and detail_dict
    contains the scoring breakdown.

    Args:
        recent_fingerprints: Token sets from recent generations for novelty
            scoring.  Pass None or [] to disable novelty penalty.
    """
    if not survived:
        ratio = min(elapsed / max_runtime, 0.99) if max_runtime > 0 else 0.0
        score = ratio * 0.49
        return score, {"basis": "failure", "elapsed_ratio": round(ratio, 4)}

    # --- Survivor scoring (0.50 – 1.0) ---
    base = 0.50

    # Filter meaningful lines (non-empty, non-whitespace-only).
    meaningful = [ln for ln in output_lines if ln.strip()]
    total = len(meaningful)

    # Volume: up to 0.10 (reduced from 0.15 to make room for novelty).
    volume = min(total / 50, 1.0) * 0.10

    # Diversity: unique content ratio.  Up to 0.10 (reduced from 0.15).
    if total > 0:
        unique = len(set(meaningful))
        diversity = (unique / total) * 0.10
    else:
        diversity = 0.0

    # Novelty: how different is this output from recent generations.  Up to 0.05.
    current_fp = _output_fingerprint(meaningful)
    if recent_fingerprints:
        novelty_raw = compute_novelty(current_fp, recent_fingerprints)
    else:
        novelty_raw = 1.0  # no history — full novelty
    novelty = novelty_raw * 0.05

    # Structured output: proportion of lines matching structured patterns.
    structured_count = 0
    for ln in meaningful:
        if any(pat.match(ln) for pat in _STRUCTURED_PATTERNS):
            structured_count += 1
    structure = min(structured_count / max(total, 1) * 2, 1.0) * 0.10

    # Functional bonus: output suggests real I/O or API interaction.  Up to 0.05.
    functional_count = 0
    for ln in meaningful:
        if any(pat.search(ln) for pat in _FUNCTIONAL_PATTERNS):
            functional_count += 1
    functional = min(functional_count / max(total, 1) * 5, 1.0) * 0.05

    # Error penalty: only count actual program errors (tracebacks, exceptions).
    error_count = sum(1 for ln in output_lines if _is_real_error_line(ln))
    error_penalty = min(error_count / max(total, 1), 1.0) * 0.10

    # Extract error signatures for cross-generation tracking.
    error_signatures = _count_persistent_errors(output_lines)

    score = base + volume + diversity + novelty + structure + functional - error_penalty
    score = max(0.50, min(score, 1.0))

    detail = {
        "basis": "survived",
        "base": base,
        "volume": round(volume, 4),
        "diversity": round(diversity, 4),
        "novelty": round(novelty, 4),
        "structure": round(structure, 4),
        "functional": round(functional, 4),
        "error_penalty": round(error_penalty, 4),
        "meaningful_lines": total,
        "error_lines": error_count,
        "fingerprint": sorted(list(current_fp))[:50],  # compact: top 50 tokens
        "error_signatures": error_signatures,
    }
    return round(score, 4), detail


_CREATE_LLM_USAGE = """\
CREATE TABLE IF NOT EXISTS llm_usage (
    id            INTEGER PRIMARY KEY,
    generation    INTEGER,
    caller        TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    timestamp     TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


class FitnessTracker(SQLiteStore):
    """Evaluate and record fitness scores in a local SQLite database."""

    _TABLE_NAME = "fitness_log"
    _CREATE_TABLE = _CREATE_TABLE

    def _migrate(self, con: sqlite3.Connection) -> None:
        try:
            con.execute("ALTER TABLE fitness_log ADD COLUMN detail TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        con.execute(_CREATE_LLM_USAGE)

    def record(
        self,
        generation: int,
        commit_hash: str,
        score: float,
        runtime_sec: float,
        survived: bool,
        detail: dict | None = None,
    ) -> int:
        """Insert a fitness entry and return its *rowid*."""
        detail_json = json.dumps(detail) if detail else None
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO fitness_log "
                "(generation, commit_hash, score, runtime_sec, survived, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (generation, commit_hash, score, runtime_sec, survived, detail_json),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_best(self, n: int = 5) -> list[dict]:
        """Return the top *n* entries ordered by score descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM fitness_log ORDER BY score DESC LIMIT ?",
                (n,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_generation_stats(self, generation: int) -> dict | None:
        """Return aggregate stats for a single generation.

        Returns a dict with keys *avg_score*, *max_score*, *min_score*,
        and *count*, or ``None`` if the generation has no entries.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT AVG(score) AS avg_score, MAX(score) AS max_score, "
                "MIN(score) AS min_score, COUNT(*) AS count "
                "FROM fitness_log WHERE generation = ?",
                (generation,),
            ).fetchone()
            if row is None or row["count"] == 0:
                return None
            return self._row_to_dict(row)

    def get_max_generation(self) -> int:
        """Return the highest recorded generation number, or -1 if empty."""
        with self._connect() as con:
            row = con.execute(
                "SELECT MAX(generation) AS max_gen FROM fitness_log"
            ).fetchone()
            val = row["max_gen"] if row else None
            return val if val is not None else -1

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent entries ordered by *id* descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM fitness_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_recent_fingerprints(self, limit: int = 10) -> list[set[str]]:
        """Return token fingerprints from recent survived generations.

        Parses the 'fingerprint' field from the detail JSON of recent
        survived entries.  Returns a list of token sets for novelty scoring.
        """
        with self._connect() as con:
            rows = con.execute(
                "SELECT detail FROM fitness_log "
                "WHERE survived = 1 AND detail IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        fingerprints: list[set[str]] = []
        for row in rows:
            try:
                detail = json.loads(row["detail"])
                fp = detail.get("fingerprint", [])
                if fp:
                    fingerprints.append(set(fp))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return fingerprints

    def get_recent_error_signatures(self, limit: int = 5, max_gen_age: int = 5) -> list[str]:
        """Return error signatures that persisted across recent generations.

        Finds errors appearing in 2+ of the last *limit* survived entries
        **within the last *max_gen_age* generations**, indicating bugs that
        evolution has failed to fix.  Older entries are ignored so that
        already-resolved errors don't linger as false positives.
        """
        with self._connect() as con:
            # Find the latest generation number.
            latest = con.execute(
                "SELECT MAX(generation) AS g FROM fitness_log",
            ).fetchone()
            min_gen = (latest["g"] or 0) - max_gen_age

            rows = con.execute(
                "SELECT detail FROM fitness_log "
                "WHERE survived = 1 AND detail IS NOT NULL "
                "AND generation >= ? "
                "ORDER BY id DESC LIMIT ?",
                (min_gen, limit),
            ).fetchall()

        from collections import Counter
        error_counter: Counter[str] = Counter()
        for row in rows:
            try:
                detail = json.loads(row["detail"])
                for sig in detail.get("error_signatures", []):
                    error_counter[sig] += 1
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        # Return errors appearing in 2+ generations (persistent).
        return [sig for sig, count in error_counter.most_common(5) if count >= 2]

    # ------------------------------------------------------------------
    # LLM token usage tracking
    # ------------------------------------------------------------------

    def record_llm_usage(
        self, generation: int, caller: str, input_tokens: int, output_tokens: int,
    ) -> int:
        """Record an LLM usage entry and return its rowid."""
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO llm_usage (generation, caller, input_tokens, output_tokens) "
                "VALUES (?, ?, ?, ?)",
                (generation, caller, input_tokens, output_tokens),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_llm_usage(self, limit: int = 50) -> list[dict]:
        """Return recent LLM usage entries, most recent first."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM llm_usage ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_llm_usage_summary(self) -> dict:
        """Return aggregate LLM usage statistics.

        Returns a dict with keys:
        - total_input, total_output, total_calls
        - by_caller: {caller: {input_tokens, output_tokens, calls}}
        """
        with self._connect() as con:
            # Overall totals.
            row = con.execute(
                "SELECT COALESCE(SUM(input_tokens), 0) AS total_input, "
                "COALESCE(SUM(output_tokens), 0) AS total_output, "
                "COUNT(*) AS total_calls FROM llm_usage",
            ).fetchone()
            summary: dict = {
                "total_input": row["total_input"],
                "total_output": row["total_output"],
                "total_calls": row["total_calls"],
            }

            # Per-caller breakdown.
            rows = con.execute(
                "SELECT caller, "
                "SUM(input_tokens) AS input_tokens, "
                "SUM(output_tokens) AS output_tokens, "
                "COUNT(*) AS calls "
                "FROM llm_usage GROUP BY caller",
            ).fetchall()
            summary["by_caller"] = {
                r["caller"]: {
                    "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                    "calls": r["calls"],
                }
                for r in rows
            }
            return summary

    def score_task_alignment(
        self,
        output_lines: list[str],
        user_categories: dict[str, float],
    ) -> float:
        """Score how well Ring 2 output aligns with user interest categories.

        Checks if output tokens overlap with high-weight user interest keywords.
        Returns a bonus score (0.0–0.10) to be added to the base fitness.

        Args:
            output_lines: Lines of Ring 2 output.
            user_categories: Dict mapping category name → weight (0.0–1.0).
        """
        if not user_categories or not output_lines:
            return 0.0

        meaningful = [ln for ln in output_lines if ln.strip()]
        if not meaningful:
            return 0.0

        # Build token set from output.
        output_tokens: set[str] = set()
        for line in meaningful:
            for word in line.split():
                cleaned = re.sub(r"[^a-zA-Z0-9_]", "", word).lower()
                if len(cleaned) >= 3 and not cleaned.isdigit():
                    output_tokens.add(cleaned)

        if not output_tokens:
            return 0.0

        # Check overlap with each category's keywords, weighted by category weight.
        total_score = 0.0
        total_weight = 0.0
        for category, weight in user_categories.items():
            if weight < 0.1:
                continue
            cat_tokens = {
                re.sub(r"[^a-zA-Z0-9_]", "", t).lower()
                for t in category.split()
                if len(t) >= 2
            }
            if not cat_tokens:
                continue
            overlap = len(output_tokens & cat_tokens)
            cat_score = min(overlap / max(len(cat_tokens), 1), 1.0)
            total_score += cat_score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        alignment = total_score / total_weight
        return round(min(alignment * 0.15, 0.15), 4)

    def is_plateaued(self, window: int = 5, epsilon: float = 0.03) -> bool:
        """Check if recent survived scores are stagnant.

        Returns True if the last *window* survived entries have scores
        within *epsilon* range (max - min <= epsilon).
        """
        with self._connect() as con:
            rows = con.execute(
                "SELECT score FROM fitness_log "
                "WHERE survived = 1 "
                "ORDER BY id DESC LIMIT ?",
                (window,),
            ).fetchall()

        scores = [row["score"] for row in rows]
        if len(scores) < window:
            return False  # not enough data yet
        return (max(scores) - min(scores)) <= epsilon
