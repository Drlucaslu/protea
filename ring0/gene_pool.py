"""Gene pool — top-N gene storage for evolutionary inheritance.

Extracts compact code summaries from Ring 2 source and stores the best
ones in SQLite.  During evolution, the top genes are injected into the
prompt so the LLM can build upon proven patterns.
Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import pathlib
import re
import sqlite3
import subprocess

from ring0.sqlite_store import SQLiteStore

log = logging.getLogger("protea.gene_pool")

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS gene_pool (
    id          INTEGER PRIMARY KEY,
    generation  INTEGER NOT NULL,
    score       REAL    NOT NULL,
    source_hash TEXT    NOT NULL UNIQUE,
    gene_summary TEXT   NOT NULL,
    created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
)
"""

# Heartbeat boilerplate patterns to skip during summary extraction.
_SKIP_NAMES = frozenset({
    "heartbeat_loop", "write_heartbeat", "main", "_heartbeat_loop",
    "heartbeat_thread", "start_heartbeat", "_write_heartbeat",
})

_STOPWORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "it", "by", "as", "be", "do", "if", "no", "so", "up", "from",
    "with", "that", "this", "not", "but", "are", "was", "has", "its",
    "self", "def", "class", "return", "none", "true", "false", "import",
})


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class GenePool(SQLiteStore):
    """Top-N gene storage for evolutionary inheritance."""

    _TABLE_NAME = "gene_pool"
    _CREATE_TABLE = _CREATE_TABLE

    def __init__(self, db_path: pathlib.Path, max_size: int = 100, embedding_provider=None) -> None:
        self.max_size = max_size
        self.embedding_provider = embedding_provider
        super().__init__(db_path)
        self._backfill_tags()
        self._backfill_embeddings()

    def _migrate(self, con: sqlite3.Connection) -> None:
        migrations = [
            "ALTER TABLE gene_pool ADD COLUMN tags TEXT",
            "ALTER TABLE gene_pool ADD COLUMN hit_count INTEGER DEFAULT 0",
            "ALTER TABLE gene_pool ADD COLUMN last_hit_gen INTEGER DEFAULT 0",
            "ALTER TABLE gene_pool ADD COLUMN task_hit_count INTEGER DEFAULT 0",
            "ALTER TABLE gene_pool ADD COLUMN last_task_hit_gen INTEGER DEFAULT 0",
            "ALTER TABLE gene_pool ADD COLUMN total_task_hits INTEGER DEFAULT 0",
            "ALTER TABLE gene_pool ADD COLUMN embedding TEXT DEFAULT ''",
        ]
        for sql in migrations:
            try:
                con.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists
        con.execute(
            "CREATE TABLE IF NOT EXISTS gene_blacklist (source_hash TEXT PRIMARY KEY)"
        )

    def _backfill_tags(self) -> None:
        """Compute and store tags for existing genes that lack them."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, gene_summary FROM gene_pool "
                "WHERE tags IS NULL AND gene_summary != ''"
            ).fetchall()
            for row in rows:
                tags = self.extract_tags(row["gene_summary"])
                con.execute(
                    "UPDATE gene_pool SET tags = ? WHERE id = ?",
                    (" ".join(tags), row["id"]),
                )
            if rows:
                log.info("Gene pool: backfilled tags for %d genes", len(rows))

    def _backfill_embeddings(self) -> None:
        """Compute and store embeddings for existing genes that lack them."""
        if not self.embedding_provider:
            return
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, gene_summary FROM gene_pool "
                "WHERE (embedding IS NULL OR embedding = '') AND gene_summary != ''"
            ).fetchall()
            for row in rows:
                try:
                    vecs = self.embedding_provider.embed([row["gene_summary"]])
                    if vecs:
                        con.execute(
                            "UPDATE gene_pool SET embedding = ? WHERE id = ?",
                            (json.dumps(vecs[0]), row["id"]),
                        )
                except Exception:
                    pass
            if rows:
                log.info("Gene pool: backfilled embeddings for %d genes", len(rows))

    def add(self, generation: int, score: float, source_code: str, detail: str | None = None) -> bool:
        """Extract gene summary from source_code and store if score qualifies.

        Returns True if added (score high enough or pool not full).
        """
        source_hash = hashlib.sha256(source_code.encode()).hexdigest()
        gene_summary = self.extract_summary(source_code)
        if not gene_summary.strip():
            return False

        tags_str = " ".join(self.extract_tags(gene_summary))

        embedding_json = ""
        if self.embedding_provider:
            try:
                vecs = self.embedding_provider.embed([gene_summary])
                if vecs:
                    embedding_json = json.dumps(vecs[0])
            except Exception:
                pass

        with self._connect() as con:
            # Check blacklist (tombstoned genes can never be re-added).
            blacklisted = con.execute(
                "SELECT 1 FROM gene_blacklist WHERE source_hash = ?",
                (source_hash,),
            ).fetchone()
            if blacklisted:
                return False

            # Check for duplicate source.
            existing = con.execute(
                "SELECT id FROM gene_pool WHERE source_hash = ?",
                (source_hash,),
            ).fetchone()
            if existing:
                return False

            # Check for near-duplicate by tag Jaccard similarity.
            new_tags = set(tags_str.split()) if tags_str else set()
            if new_tags:
                rows = con.execute(
                    "SELECT id, score, tags, hit_count FROM gene_pool WHERE tags IS NOT NULL AND tags != ''"
                ).fetchall()
                for row in rows:
                    existing_tags = set(row["tags"].split())
                    intersection = len(new_tags & existing_tags)
                    union = len(new_tags | existing_tags)
                    if union > 0 and intersection / union > 0.75:
                        if score > row["score"]:
                            # Better variant — replace, preserve hit_count.
                            con.execute(
                                "UPDATE gene_pool SET generation = ?, score = ?, "
                                "source_hash = ?, gene_summary = ?, tags = ?, embedding = ? WHERE id = ?",
                                (generation, score, source_hash, gene_summary, tags_str, embedding_json, row["id"]),
                            )
                            log.info(
                                "Gene pool: replaced similar gene %d (%.2f→%.2f, jaccard=%.2f)",
                                row["id"], row["score"], score, intersection / union,
                            )
                            return True
                        else:
                            log.debug(
                                "Gene pool: skipped near-duplicate of gene %d (jaccard=%.2f)",
                                row["id"], intersection / union,
                            )
                            return False

            count = con.execute("SELECT COUNT(*) AS cnt FROM gene_pool").fetchone()["cnt"]

            if count < self.max_size:
                con.execute(
                    "INSERT INTO gene_pool (generation, score, source_hash, gene_summary, tags, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (generation, score, source_hash, gene_summary, tags_str, embedding_json),
                )
                log.info("Gene pool: added gen-%d (score=%.2f, pool=%d/%d)",
                         generation, score, count + 1, self.max_size)
                return True

            # Pool is full — evict lowest if new score is higher.
            min_row = con.execute(
                "SELECT id, score FROM gene_pool ORDER BY score ASC LIMIT 1"
            ).fetchone()
            if min_row and score > min_row["score"]:
                con.execute("DELETE FROM gene_pool WHERE id = ?", (min_row["id"],))
                con.execute(
                    "INSERT INTO gene_pool (generation, score, source_hash, gene_summary, tags, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (generation, score, source_hash, gene_summary, tags_str, embedding_json),
                )
                log.info("Gene pool: replaced lowest (%.2f) with gen-%d (score=%.2f)",
                         min_row["score"], generation, score)
                return True

            return False

    def get_id_by_hash(self, source_hash: str) -> int | None:
        """Look up gene ID by source hash. Returns None if not found."""
        with self._connect() as con:
            row = con.execute(
                "SELECT id FROM gene_pool WHERE source_hash = ?",
                (source_hash,),
            ).fetchone()
            return row["id"] if row else None

    def get_top(self, n: int = 3) -> list[dict]:
        """Return top N genes by score (then generation DESC as tiebreaker).

        Pass n=0 to return all genes.
        """
        with self._connect() as con:
            if n <= 0:
                rows = con.execute(
                    "SELECT id, generation, score, gene_summary, tags, hit_count, last_hit_gen, "
                    "task_hit_count, last_task_hit_gen, total_task_hits "
                    "FROM gene_pool ORDER BY score DESC, generation DESC",
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT id, generation, score, gene_summary, tags, hit_count, last_hit_gen, "
                    "task_hit_count, last_task_hit_gen, total_task_hits "
                    "FROM gene_pool ORDER BY score DESC, generation DESC LIMIT ?",
                    (n,),
                ).fetchall()
            return [dict(r) for r in rows]

    def delete_gene(self, gene_id: int) -> bool:
        """Permanently delete a gene and blacklist its source_hash.

        The blacklist prevents the gene from being re-added via
        backfill_from_git() or survival-path add().
        Returns True if the gene existed and was deleted.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT source_hash FROM gene_pool WHERE id = ?", (gene_id,)
            ).fetchone()
            if not row:
                return False
            con.execute(
                "INSERT OR IGNORE INTO gene_blacklist (source_hash) VALUES (?)",
                (row["source_hash"],),
            )
            con.execute("DELETE FROM gene_pool WHERE id = ?", (gene_id,))
            log.info("Gene %d deleted and blacklisted (hash=%s…)", gene_id, row["source_hash"][:12])
            return True

    def get_relevant(self, context: str, n: int = 3, query_embedding: list[float] | None = None) -> list[dict]:
        """Return top N genes by hybrid relevance score.

        Scores each gene by tag overlap, embedding cosine similarity,
        fitness score, and task hit count.  No fallback needed — the
        unified score always produces a meaningful ranking.
        """
        context_tags = set(self.extract_tags(context))

        with self._connect() as con:
            rows = con.execute(
                "SELECT id, generation, score, gene_summary, tags, hit_count, last_hit_gen, "
                "task_hit_count, last_task_hit_gen, total_task_hits, embedding "
                "FROM gene_pool"
            ).fetchall()

        if not rows:
            return []

        scored: list[tuple[float, dict]] = []
        for row in rows:
            gene = dict(row)
            gene_tags = set((gene.get("tags") or "").split())

            # Tag overlap score
            overlap = len(context_tags & gene_tags) if context_tags else 0

            # Embedding cosine similarity score
            cos_sim = 0.0
            if query_embedding and gene.get("embedding"):
                try:
                    gene_emb = json.loads(gene["embedding"])
                    cos_sim = _cosine_similarity(query_embedding, gene_emb)
                except (json.JSONDecodeError, TypeError):
                    pass

            task_hits = gene.get("total_task_hits") or 0
            relevance = overlap * 2 + cos_sim * 3 + gene["score"] + min(task_hits, 10) * 0.5
            scored.append((relevance, gene))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Strip embedding from returned dicts (large, not needed by callers)
        results = []
        for _, gene in scored[:n]:
            gene.pop("embedding", None)
            results.append(gene)
        return results

    def record_hits(self, gene_ids: list[int], generation: int) -> None:
        """Increment hit_count and update last_hit_gen for the given gene ids."""
        if not gene_ids:
            return
        with self._connect() as con:
            for gid in gene_ids:
                con.execute(
                    "UPDATE gene_pool SET hit_count = hit_count + 1, last_hit_gen = ? "
                    "WHERE id = ?",
                    (generation, gid),
                )

    def record_task_hits(self, gene_ids: list[int], generation: int) -> None:
        """Increment task_hit_count (consumable) and total_task_hits (cumulative)."""
        if not gene_ids:
            return
        with self._connect() as con:
            for gid in gene_ids:
                con.execute(
                    "UPDATE gene_pool SET task_hit_count = task_hit_count + 1, "
                    "total_task_hits = total_task_hits + 1, "
                    "last_task_hit_gen = ? WHERE id = ?",
                    (generation, gid),
                )

    def apply_task_boost(self) -> int:
        """Boost genes with task_hit_count > 0: +0.05 per hit (cap 1.0).

        Resets task_hit_count after boosting.  Returns number of genes boosted.
        """
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, score, task_hit_count FROM gene_pool WHERE task_hit_count > 0"
            ).fetchall()
            boosted = 0
            for row in rows:
                new_score = min(row["score"] + row["task_hit_count"] * 0.05, 1.0)
                con.execute(
                    "UPDATE gene_pool SET score = ?, task_hit_count = 0 WHERE id = ?",
                    (new_score, row["id"]),
                )
                boosted += 1
            return boosted

    def apply_boost(self) -> int:
        """Boost genes with hit_count >= 3: +0.01 per 3 hits (cap 1.0).

        Resets hit_count but preserves remainder.  Returns number of genes boosted.
        """
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, score, hit_count FROM gene_pool WHERE hit_count >= 3"
            ).fetchall()
            boosted = 0
            for row in rows:
                increments = row["hit_count"] // 3
                remainder = row["hit_count"] % 3
                new_score = min(row["score"] + increments * 0.01, 1.0)
                con.execute(
                    "UPDATE gene_pool SET score = ?, hit_count = ? WHERE id = ?",
                    (new_score, remainder, row["id"]),
                )
                boosted += 1
            return boosted

    def verify_adoption(
        self,
        new_source: str,
        gene_ids: list[int],
        generation: int,
    ) -> list[int]:
        """Verify which genes were actually adopted in the new Ring 2 code.

        Compares gene summary tags against tokens in the new source.
        Only genes with meaningful tag overlap are recorded as hits.

        Args:
            new_source: The new Ring 2 source code after evolution.
            gene_ids: Gene IDs that were injected into the prompt.
            generation: Current generation number.

        Returns:
            List of gene IDs that were verified as adopted.
        """
        if not gene_ids or not new_source:
            return []

        # Extract tokens from new source code.
        source_tokens: set[str] = set()
        for word in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', new_source):
            for part in word.split("_"):
                low = part.lower()
                if len(low) >= 3 and low not in _STOPWORDS:
                    source_tokens.add(low)

        adopted: list[int] = []
        with self._connect() as con:
            for gid in gene_ids:
                row = con.execute(
                    "SELECT tags FROM gene_pool WHERE id = ?", (gid,),
                ).fetchone()
                if not row or not row["tags"]:
                    continue
                gene_tags = set(row["tags"].split())
                if not gene_tags:
                    continue
                overlap = len(gene_tags & source_tokens)
                # Require at least 30% tag overlap to count as adopted.
                if overlap / len(gene_tags) >= 0.3:
                    adopted.append(gid)

        if adopted:
            self.record_hits(adopted, generation)
            log.info("Gene adoption verified: %d/%d genes adopted", len(adopted), len(gene_ids))

        return adopted

    def apply_decay(self, current_generation: int) -> int:
        """Decay genes not hit in >10 generations (floor 0.10).

        Both last_hit_gen and last_task_hit_gen must be stale.
        Genes with total_task_hits > 0 decay gently (-0.01) since they
        have proven user value.  Others decay faster (-0.03).

        Returns number of genes decayed.
        """
        threshold_gen = current_generation - 10
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, score, total_task_hits FROM gene_pool "
                "WHERE last_hit_gen < ? AND last_task_hit_gen < ? AND score > 0.10",
                (threshold_gen, threshold_gen),
            ).fetchall()
            decayed = 0
            for row in rows:
                rate = 0.01 if (row["total_task_hits"] or 0) > 0 else 0.03
                new_score = max(row["score"] - rate, 0.10)
                con.execute(
                    "UPDATE gene_pool SET score = ? WHERE id = ?",
                    (new_score, row["id"]),
                )
                decayed += 1
            return decayed

    def backfill(self, skill_store) -> int:
        """One-time backfill from existing crystallized skills.

        Reads source_code from the skill store and populates the gene
        pool.  Only runs if the pool is currently empty.
        Returns the number of genes added.
        """
        if self.count() > 0:
            return 0

        try:
            skills = skill_store.get_active(50)
        except Exception:
            return 0

        added = 0
        for skill in skills:
            source = skill.get("source_code", "")
            if not source or len(source) < 50:
                continue
            # Use usage_count as a proxy score (normalised to 0-1).
            usage = skill.get("usage_count", 0)
            score = min(0.50 + usage * 0.05, 0.95)
            gen = skill.get("id", 0)
            if self.add(gen, score, source):
                added += 1

        if added:
            log.info("Gene pool: backfilled %d genes from skill store", added)
        return added

    def backfill_from_git(self, ring2_path: pathlib.Path, fitness_tracker) -> int:
        """Backfill genes from git history using fitness_log records.

        Queries fitness_log for survived generations with high scores,
        recovers the Ring 2 source from git commits, and adds them to
        the pool.  Dedup via source_hash handles repeats.
        Returns the number of genes added.
        """
        slots = self.max_size - self.count()
        if slots <= 0:
            return 0

        git_dir = ring2_path / ".git"
        if not git_dir.is_dir():
            log.debug("backfill_from_git: no .git in %s", ring2_path)
            return 0

        # Query fitness_log for survived entries with commit hashes.
        min_score = 0.75
        with fitness_tracker._connect() as con:
            rows = con.execute(
                "SELECT generation, score, commit_hash FROM fitness_log "
                "WHERE survived = 1 AND commit_hash IS NOT NULL AND score > ? "
                "ORDER BY score DESC LIMIT ?",
                (min_score, slots),
            ).fetchall()

        if not rows:
            log.debug("backfill_from_git: no qualifying fitness_log entries")
            return 0

        added = 0
        for row in rows:
            generation = row["generation"]
            score = row["score"]
            commit_hash = row["commit_hash"]

            try:
                result = subprocess.run(
                    ["git", "show", f"{commit_hash}:main.py"],
                    cwd=str(ring2_path),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    continue
                source = result.stdout
            except (subprocess.TimeoutExpired, OSError):
                continue

            if not source or len(source) < 50:
                continue

            if self.add(generation, score, source):
                added += 1
                log.debug("backfill_from_git: added gen-%d (score=%.2f)", generation, score)

        if added:
            log.info("Gene pool: backfilled %d genes from git history", added)
        return added

    @staticmethod
    def extract_tags(text: str) -> list[str]:
        """Extract searchable tags from a gene summary or context string.

        - Splits PascalCase names: StreamAnalyzer → ["stream", "analyzer"]
        - Splits snake_case names: compute_fibonacci → ["compute", "fibonacci"]
        - Extracts words from docstrings/prose
        - Filters stopwords and short tokens (< 3 chars)
        - Returns deduplicated lowercase list
        """
        tokens: set[str] = set()

        # Split PascalCase identifiers.
        for match in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text):
            parts = re.findall(r'[A-Z][a-z]+', match.group(1))
            for p in parts:
                tokens.add(p.lower())

        # Split snake_case identifiers and extract all words.
        for word in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text):
            for part in word.split("_"):
                low = part.lower()
                if low:
                    tokens.add(low)

        # Extract hyphenated words (e.g. "real-time").
        for match in re.finditer(r'\b([a-z]+-[a-z]+)\b', text, re.IGNORECASE):
            for part in match.group(1).split("-"):
                low = part.lower()
                if low:
                    tokens.add(low)

        # Filter stopwords and short tokens.
        return sorted(t for t in tokens if len(t) >= 3 and t not in _STOPWORDS)

    @staticmethod
    def extract_summary(source_code: str) -> str:
        """Extract compact gene summary from Ring 2 source code.

        Uses AST parsing for precision (class methods, proper docstrings),
        with regex fallback for broken code that won't parse.
        - Skip heartbeat boilerplate (heartbeat_loop, write_heartbeat, main)
        - Include first-line docstrings
        - Cap at 500 chars
        """
        try:
            summary = _extract_summary_ast(source_code)
        except Exception:
            summary = _extract_summary_regex(source_code)

        if len(summary) > 500:
            summary = summary[:497] + "..."
        return summary


def _extract_summary_ast(source_code: str) -> str:
    """AST-based extraction — precise, handles methods inside classes."""
    tree = ast.parse(source_code)
    lines: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.lower() in _SKIP_NAMES:
                continue
            # Class signature.
            bases = [_ast_name(b) for b in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"class {node.name}{base_str}:")
            # Class docstring.
            doc = ast.get_docstring(node)
            if doc:
                first_line = doc.split("\n")[0].strip()
                if first_line:
                    lines.append(f'    """{first_line}"""')
            # Methods (skip boilerplate, skip __init__).
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    if child.name.lower() in _SKIP_NAMES or child.name == "__init__":
                        continue
                    args = _format_args(child.args)
                    lines.append(f"    def {child.name}({args}): ...")
                    mdoc = ast.get_docstring(child)
                    if mdoc:
                        mfirst = mdoc.split("\n")[0].strip()
                        if mfirst:
                            lines.append(f'        """{mfirst}"""')

        elif isinstance(node, ast.FunctionDef):
            if node.name.lower() in _SKIP_NAMES:
                continue
            args = _format_args(node.args)
            lines.append(f"def {node.name}({args}):")
            doc = ast.get_docstring(node)
            if doc:
                first_line = doc.split("\n")[0].strip()
                if first_line:
                    lines.append(f'    """{first_line}"""')

    return "\n".join(lines)


def _ast_name(node) -> str:
    """Get a readable name from an AST node (base class, etc.)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_ast_name(node.value)}.{node.attr}"
    return "?"


def _format_args(args: ast.arguments) -> str:
    """Format function arguments compactly: 'self, x, y=0'."""
    parts: list[str] = []
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        name = arg.arg
        default_idx = i - defaults_offset
        if default_idx >= 0:
            parts.append(f"{name}=...")
        else:
            parts.append(name)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _extract_summary_regex(source_code: str) -> str:
    """Regex fallback — for code that fails AST parsing."""
    lines: list[str] = []

    for match in re.finditer(
        r'^(class\s+(\w+)[^\n]*|def\s+(\w+)[^\n]*)', source_code, re.MULTILINE
    ):
        full_line = match.group(0)
        name = match.group(2) or match.group(3)

        if name and name.lower() in _SKIP_NAMES:
            continue

        lines.append(full_line)

        # Try to grab first-line docstring right after the def/class.
        end_pos = match.end()
        rest = source_code[end_pos:]
        doc_match = re.match(r'\s*\n\s*("""([^"]*?)"""|\'\'\'([^\']*?)\'\'\')', rest)
        if doc_match:
            docstring = (doc_match.group(2) or doc_match.group(3) or "").strip()
            first_line = docstring.split("\n")[0].strip()
            if first_line:
                lines.append(f'    """{first_line}"""')

    return "\n".join(lines)
