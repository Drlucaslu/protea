"""Tests for ring0.gene_pool."""

import pathlib
import subprocess

import pytest

from ring0.fitness import FitnessTracker
from ring0.gene_pool import GenePool


# --- Sample Ring 2 source code for testing ---

SAMPLE_SOURCE = '''\
import os, pathlib, time, threading, json

def write_heartbeat(path, pid):
    """Write heartbeat file."""
    path.write_text(f"{pid}\\n{time.time()}\\n")

def heartbeat_loop(hb_path, pid):
    while True:
        write_heartbeat(hb_path, pid)
        time.sleep(2)

class StreamAnalyzer:
    """Real-time anomaly detection in data streams."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.buffer = []

    def analyze(self, value):
        """Detect anomalies using z-score method."""
        self.buffer.append(value)

class PackageScanner:
    """PyPI dependency graph builder."""

    def scan(self, package_name):
        """Scan a package and its dependencies."""
        pass

def compute_fibonacci(n):
    """Calculate fibonacci sequence."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    t = threading.Thread(target=heartbeat_loop, args=(hb, pid), daemon=True)
    t.start()
    analyzer = StreamAnalyzer()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
'''

def _make_distinct_source(class_name: str, method_name: str, docstring: str) -> str:
    """Generate a distinct Ring 2 source with unique summary/tags."""
    return f'''\
import os, pathlib, time, threading

class {class_name}:
    """{docstring}"""

    def __init__(self):
        self.data = []

    def {method_name}(self, value):
        """Process the given value."""
        self.data.append(value)

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
'''

# Pre-built distinct sources for tests that need multiple different genes.
DISTINCT_SOURCES = [
    _make_distinct_source("WeatherForecast", "predict", "Atmospheric pressure prediction engine"),
    _make_distinct_source("MusicComposer", "compose", "Algorithmic melody generation system"),
    _make_distinct_source("ChessEngine", "evaluate", "Board position evaluation framework"),
    _make_distinct_source("GraphTraversal", "search", "Breadth-first graph exploration tool"),
]

TRIVIAL_SOURCE = '''\
import os, pathlib, time

def write_heartbeat(path, pid):
    path.write_text(f"{pid}\\n{time.time()}\\n")

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    while True:
        write_heartbeat(hb, pid)
        time.sleep(2)

if __name__ == "__main__":
    main()
'''


class TestExtractSummary:
    def test_extracts_class_signatures(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "class StreamAnalyzer" in summary
        assert "class PackageScanner" in summary

    def test_extracts_function_signatures(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "def compute_fibonacci" in summary

    def test_extracts_method_signatures(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        # AST can extract methods inside classes.
        assert "def analyze" in summary
        assert "def scan" in summary

    def test_extracts_docstrings(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "Real-time anomaly detection" in summary
        assert "PyPI dependency graph" in summary

    def test_skips_heartbeat_boilerplate(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "write_heartbeat" not in summary
        assert "heartbeat_loop" not in summary
        assert "def main" not in summary

    def test_caps_at_500_chars(self):
        # Create a source with many classes to exceed 500 chars.
        lines = []
        for i in range(50):
            lines.append(f"class Widget{i}:")
            lines.append(f'    """Widget number {i} with a description."""')
            lines.append(f"    pass")
        big_source = "\n".join(lines)
        summary = GenePool.extract_summary(big_source)
        assert len(summary) <= 500
        assert summary.endswith("...")

    def test_empty_source(self):
        summary = GenePool.extract_summary("")
        assert summary == ""

    def test_trivial_heartbeat_only(self):
        summary = GenePool.extract_summary(TRIVIAL_SOURCE)
        # Should be empty — only heartbeat boilerplate.
        assert summary.strip() == ""

    def test_regex_fallback_on_broken_code(self):
        """Code with syntax errors falls back to regex extraction."""
        broken = (
            "class GoodClass:\n"
            '    """A useful class."""\n'
            "    pass\n\n"
            "def broken_syntax(\n"  # unclosed paren — SyntaxError
        )
        summary = GenePool.extract_summary(broken)
        assert "GoodClass" in summary
        assert "A useful class" in summary

    def test_extracts_method_docstrings(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "Detect anomalies" in summary
        assert "Scan a package" in summary

    def test_skips_init_methods(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "__init__" not in summary


class TestGenePoolAdd:
    def test_add_basic(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=5)
        added = gp.add(1, 0.85, SAMPLE_SOURCE)
        assert added is True
        assert gp.count() == 1

    def test_add_dedup(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=5)
        gp.add(1, 0.85, SAMPLE_SOURCE)
        added = gp.add(2, 0.90, SAMPLE_SOURCE)  # same source
        assert added is False
        assert gp.count() == 1

    def test_add_trivial_rejected(self, tmp_path):
        """Trivial code with empty summary should not be added."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=5)
        added = gp.add(1, 0.85, TRIVIAL_SOURCE)
        assert added is False
        assert gp.count() == 0

    def test_evict_lowest_when_full(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=3)

        # Fill the pool with 3 genuinely different entries.
        for i in range(3):
            gp.add(i, 0.50 + i * 0.10, DISTINCT_SOURCES[i])
        assert gp.count() == 3

        # Add a higher score — should evict the lowest (0.50).
        added = gp.add(99, 0.95, SAMPLE_SOURCE)
        assert added is True
        assert gp.count() == 3

        # The lowest should now be 0.60 (not 0.50).
        top = gp.get_top(10)
        scores = [g["score"] for g in top]
        assert 0.50 not in scores
        assert 0.95 in scores

    def test_reject_when_full_and_low_score(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=2)

        gp.add(1, 0.80, DISTINCT_SOURCES[0])
        gp.add(2, 0.90, DISTINCT_SOURCES[1])
        assert gp.count() == 2

        # Score too low to enter.
        added = gp.add(3, 0.70, DISTINCT_SOURCES[2])
        assert added is False
        assert gp.count() == 2


class TestGenePoolGetTop:
    def test_get_top_sorted(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        gp.add(1, 0.60, DISTINCT_SOURCES[0])
        gp.add(2, 0.90, DISTINCT_SOURCES[1])
        gp.add(3, 0.75, DISTINCT_SOURCES[2])

        top = gp.get_top(2)
        assert len(top) == 2
        assert top[0]["score"] == 0.90
        assert top[1]["score"] == 0.75

    def test_get_top_with_fewer(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE + "\n# only\n")

        top = gp.get_top(5)
        assert len(top) == 1

    def test_get_top_empty(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        top = gp.get_top(3)
        assert top == []

    def test_get_top_contains_summary(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# x\n")

        top = gp.get_top(1)
        assert len(top) == 1
        assert "gene_summary" in top[0]
        assert "StreamAnalyzer" in top[0]["gene_summary"]


class TestGenePoolBackfill:
    def test_backfill_from_skills(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        # Mock skill store.
        class MockSkillStore:
            def get_active(self, limit):
                return [
                    {"id": 1, "source_code": DISTINCT_SOURCES[0], "usage_count": 5},
                    {"id": 2, "source_code": DISTINCT_SOURCES[1], "usage_count": 0},
                ]

        added = gp.backfill(MockSkillStore())
        assert added >= 1

    def test_backfill_skips_if_nonempty(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# pre\n")

        class MockSkillStore:
            def get_active(self, limit):
                return [{"id": 1, "source_code": SAMPLE_SOURCE + "\n# new\n", "usage_count": 5}]

        added = gp.backfill(MockSkillStore())
        assert added == 0

    def test_backfill_skips_short_source(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        class MockSkillStore:
            def get_active(self, limit):
                return [
                    {"id": 1, "source_code": "x=1", "usage_count": 5},
                    {"id": 2, "source_code": "", "usage_count": 0},
                ]

        added = gp.backfill(MockSkillStore())
        assert added == 0


def _init_git_repo(ring2_path: pathlib.Path, commits: list[tuple[str, str]]):
    """Create a git repo with commits. Each commit is (source_code, label).

    Returns list of actual commit hashes.
    """
    subprocess.run(["git", "init"], cwd=str(ring2_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(ring2_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(ring2_path), capture_output=True)

    hashes = []
    for source, _ in commits:
        (ring2_path / "main.py").write_text(source)
        subprocess.run(["git", "add", "main.py"], cwd=str(ring2_path), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "gen"], cwd=str(ring2_path), capture_output=True, check=True)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ring2_path), capture_output=True, text=True, check=True,
        )
        hashes.append(result.stdout.strip())
    return hashes


class TestBackfillFromGit:
    def test_backfills_from_fitness_log(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        hashes = _init_git_repo(ring2, [(DISTINCT_SOURCES[0], ""), (DISTINCT_SOURCES[1], "")])

        gp = GenePool(db, max_size=10)
        ft = FitnessTracker(db)
        ft.record(1, hashes[0], 0.85, 1.0, True)
        ft.record(2, hashes[1], 0.90, 1.0, True)

        added = gp.backfill_from_git(ring2, ft)
        assert added == 2
        assert gp.count() == 2

    def test_skips_when_pool_full(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        hashes = _init_git_repo(ring2, [(SAMPLE_SOURCE + "\n# x\n", "")])

        gp = GenePool(db, max_size=1)
        ft = FitnessTracker(db)
        ft.record(1, hashes[0], 0.90, 1.0, True)

        # Fill the pool first.
        gp.add(99, 0.95, SAMPLE_SOURCE + "\n# existing\n")
        assert gp.count() == 1

        added = gp.backfill_from_git(ring2, ft)
        assert added == 0

    def test_dedup_same_source(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        # One commit, but two fitness_log entries pointing to it.
        source = SAMPLE_SOURCE + "\n# same\n"
        hashes = _init_git_repo(ring2, [(source, "")])

        gp = GenePool(db, max_size=10)
        ft = FitnessTracker(db)
        ft.record(1, hashes[0], 0.85, 1.0, True)
        ft.record(2, hashes[0], 0.90, 1.0, True)

        added = gp.backfill_from_git(ring2, ft)
        assert added == 1  # dedup by source_hash
        assert gp.count() == 1

    def test_skips_low_score(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        hashes = _init_git_repo(ring2, [(SAMPLE_SOURCE + "\n# low\n", "")])

        gp = GenePool(db, max_size=10)
        ft = FitnessTracker(db)
        ft.record(1, hashes[0], 0.50, 1.0, True)  # below 0.75 threshold

        added = gp.backfill_from_git(ring2, ft)
        assert added == 0

    def test_skips_without_git_dir(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()  # No .git

        gp = GenePool(db, max_size=10)
        ft = FitnessTracker(db)

        added = gp.backfill_from_git(ring2, ft)
        assert added == 0


class TestGetTopWithId:
    def test_results_contain_id(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# id_test\n")
        top = gp.get_top(1)
        assert len(top) == 1
        assert "id" in top[0]
        assert isinstance(top[0]["id"], int)

    def test_results_contain_hit_fields(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# hit_fields\n")
        top = gp.get_top(1)
        assert top[0]["hit_count"] == 0
        assert top[0]["last_hit_gen"] == 0


class TestGetRelevantWithId:
    def test_results_contain_id(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# rel_id\n")
        relevant = gp.get_relevant("stream analyzer", 1)
        assert len(relevant) == 1
        assert "id" in relevant[0]
        assert isinstance(relevant[0]["id"], int)


class TestRecordHits:
    def test_increments_hit_count(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# hit1\n")
        gene_id = gp.get_top(1)[0]["id"]
        gp.record_hits([gene_id], generation=5)
        top = gp.get_top(1)
        assert top[0]["hit_count"] == 1
        assert top[0]["last_hit_gen"] == 5

    def test_multiple_genes(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, DISTINCT_SOURCES[0])
        gp.add(2, 0.90, DISTINCT_SOURCES[1])
        ids = [g["id"] for g in gp.get_top(0)]
        gp.record_hits(ids, generation=10)
        for g in gp.get_top(0):
            assert g["hit_count"] == 1
            assert g["last_hit_gen"] == 10

    def test_empty_list(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.record_hits([], generation=1)  # should not raise


class TestApplyBoost:
    def test_boost_at_threshold(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.70, SAMPLE_SOURCE + "\n# boost1\n")
        gene_id = gp.get_top(1)[0]["id"]
        # Simulate 3 hits.
        for gen in range(3):
            gp.record_hits([gene_id], generation=gen)
        boosted = gp.apply_boost()
        assert boosted == 1
        top = gp.get_top(1)
        assert abs(top[0]["score"] - 0.73) < 0.001
        assert top[0]["hit_count"] == 0  # remainder is 0

    def test_cap_at_1_0(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.99, SAMPLE_SOURCE + "\n# cap1\n")
        gene_id = gp.get_top(1)[0]["id"]
        for gen in range(6):
            gp.record_hits([gene_id], generation=gen)
        boosted = gp.apply_boost()
        assert boosted == 1
        top = gp.get_top(1)
        assert top[0]["score"] == 1.0

    def test_remainder_preserved(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.70, SAMPLE_SOURCE + "\n# rem1\n")
        gene_id = gp.get_top(1)[0]["id"]
        # 5 hits → 1 boost (3 used) + 2 remainder
        for gen in range(5):
            gp.record_hits([gene_id], generation=gen)
        gp.apply_boost()
        top = gp.get_top(1)
        assert top[0]["hit_count"] == 2

    def test_no_boost_below_threshold(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.70, SAMPLE_SOURCE + "\n# nobst\n")
        gene_id = gp.get_top(1)[0]["id"]
        gp.record_hits([gene_id], generation=1)
        gp.record_hits([gene_id], generation=2)
        boosted = gp.apply_boost()
        assert boosted == 0
        assert gp.get_top(1)[0]["hit_count"] == 2


class TestApplyDecay:
    def test_stale_decay(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE + "\n# decay1\n")
        # last_hit_gen=0, current_gen=20 → stale (0 < 20-10)
        decayed = gp.apply_decay(current_generation=20)
        assert decayed == 1
        top = gp.get_top(1)
        assert abs(top[0]["score"] - 0.78) < 0.001

    def test_floor_respect(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.11, SAMPLE_SOURCE + "\n# floor1\n")
        decayed = gp.apply_decay(current_generation=20)
        assert decayed == 1
        top = gp.get_top(1)
        assert abs(top[0]["score"] - 0.10) < 0.001

    def test_already_at_floor(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.10, SAMPLE_SOURCE + "\n# atfloor\n")
        decayed = gp.apply_decay(current_generation=20)
        assert decayed == 0  # score is already at floor

    def test_skip_recently_hit(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE + "\n# recent1\n")
        gene_id = gp.get_top(1)[0]["id"]
        gp.record_hits([gene_id], generation=15)
        # current_gen=20, last_hit_gen=15 → 15 is NOT < 20-10=10, so no decay
        decayed = gp.apply_decay(current_generation=20)
        assert decayed == 0
        assert gp.get_top(1)[0]["score"] == 0.80


class TestExtractTags:
    def test_splits_pascal_case(self):
        tags = GenePool.extract_tags("StreamAnalyzer")
        assert "stream" in tags
        assert "analyzer" in tags

    def test_splits_snake_case(self):
        tags = GenePool.extract_tags("compute_fibonacci")
        assert "compute" in tags
        assert "fibonacci" in tags

    def test_extracts_docstring_words(self):
        tags = GenePool.extract_tags('"""Real-time anomaly detection"""')
        assert "real" in tags
        assert "time" in tags
        assert "anomaly" in tags
        assert "detection" in tags

    def test_filters_stopwords(self):
        tags = GenePool.extract_tags("the and for with class def")
        assert "the" not in tags
        assert "and" not in tags
        assert "for" not in tags

    def test_filters_short_tokens(self):
        tags = GenePool.extract_tags("a in x do it")
        assert tags == []

    def test_deduplicates(self):
        tags = GenePool.extract_tags("stream Stream STREAM stream_analyzer StreamAnalyzer")
        assert tags.count("stream") == 1

    def test_empty_input(self):
        tags = GenePool.extract_tags("")
        assert tags == []

    def test_mixed_summary(self):
        summary = (
            'class StreamAnalyzer:\n'
            '    """Real-time anomaly detection in data streams."""\n'
            '    def analyze(self, value): ...\n'
            'def compute_fibonacci(n):\n'
            '    """Calculate fibonacci sequence."""'
        )
        tags = GenePool.extract_tags(summary)
        assert "stream" in tags
        assert "analyzer" in tags
        assert "anomaly" in tags
        assert "detection" in tags
        assert "fibonacci" in tags
        assert "compute" in tags

    def test_returns_sorted(self):
        tags = GenePool.extract_tags("Zebra Apple Mango")
        assert tags == sorted(tags)


class TestGetRelevant:
    def test_returns_matching_genes(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        # Add genes with different themes.
        source_stream = SAMPLE_SOURCE + "\n# stream_version\n"
        gp.add(1, 0.80, source_stream)  # has "stream", "analyzer", "anomaly" tags

        # Add a different gene (fibonacci-focused, no stream).
        fib_source = '''\
import os, pathlib, time, threading

def compute_prime_sieve(n):
    """Compute primes using sieve of Eratosthenes."""
    pass

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
'''
        gp.add(2, 0.90, fib_source)

        # Query with context matching "stream" and "anomaly".
        relevant = gp.get_relevant("stream anomaly detection monitoring", 1)
        assert len(relevant) == 1
        assert relevant[0]["generation"] == 1  # stream gene, not the higher-score prime gene

    def test_fallback_to_score_when_no_overlap(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        gp.add(1, 0.60, DISTINCT_SOURCES[0])
        gp.add(2, 0.90, DISTINCT_SOURCES[1])

        # Context with completely unrelated terms.
        relevant = gp.get_relevant("xyzzy quantum entanglement", 2)
        assert len(relevant) == 2
        # Should fall back to score ordering.
        assert relevant[0]["score"] == 0.90

    def test_fallback_when_empty_context(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# x\n")

        relevant = gp.get_relevant("", 3)
        assert len(relevant) == 1

    def test_empty_pool(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        relevant = gp.get_relevant("stream analyzer", 3)
        assert relevant == []

    def test_returns_tags_field(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# t\n")

        relevant = gp.get_relevant("stream analyzer", 1)
        assert len(relevant) == 1
        assert "tags" in relevant[0]
        assert "stream" in relevant[0]["tags"]


class TestTagBackfill:
    def test_backfills_tags_on_init(self, tmp_path):
        db = tmp_path / "test.db"

        # Create pool and insert a gene WITHOUT tags (simulate pre-migration).
        import sqlite3
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE IF NOT EXISTS gene_pool ("
            "    id INTEGER PRIMARY KEY,"
            "    generation INTEGER NOT NULL,"
            "    score REAL NOT NULL,"
            "    source_hash TEXT NOT NULL UNIQUE,"
            "    gene_summary TEXT NOT NULL,"
            "    created_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        con.execute(
            "INSERT INTO gene_pool (generation, score, source_hash, gene_summary) "
            "VALUES (?, ?, ?, ?)",
            (1, 0.85, "abc123", 'class StreamAnalyzer:\n    """Real-time anomaly detection"""'),
        )
        con.commit()
        con.close()

        # Re-open with GenePool — should migrate + backfill tags.
        gp = GenePool(db, max_size=10)
        top = gp.get_top(1)
        assert len(top) == 1
        assert top[0]["tags"] is not None
        assert "stream" in top[0]["tags"]
        assert "analyzer" in top[0]["tags"]

    def test_add_stores_tags(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# tag_test\n")

        top = gp.get_top(1)
        assert top[0]["tags"] is not None
        assert len(top[0]["tags"]) > 0
        # Should contain tags from the summary (StreamAnalyzer, etc.)
        assert "stream" in top[0]["tags"]


class TestDeleteGene:
    def test_delete_removes_gene(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# del1\n")
        gene_id = gp.get_top(1)[0]["id"]
        result = gp.delete_gene(gene_id)
        assert result is True
        assert gp.count() == 0

    def test_delete_nonexistent_returns_false(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        result = gp.delete_gene(9999)
        assert result is False


class TestBlacklist:
    def test_deleted_gene_cannot_be_readded(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        source = SAMPLE_SOURCE + "\n# bl1\n"
        gp.add(1, 0.85, source)
        gene_id = gp.get_top(1)[0]["id"]
        gp.delete_gene(gene_id)
        # Try to re-add the same source.
        added = gp.add(2, 0.95, source)
        assert added is False
        assert gp.count() == 0

    def test_different_gene_still_addable(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        source_a = SAMPLE_SOURCE + "\n# bl_a\n"
        source_b = SAMPLE_SOURCE + "\n# bl_b\n"
        gp.add(1, 0.85, source_a)
        gene_id = gp.get_top(1)[0]["id"]
        gp.delete_gene(gene_id)
        # Different source should still be addable.
        added = gp.add(2, 0.90, source_b)
        assert added is True
        assert gp.count() == 1

    def test_blacklist_survives_reopen(self, tmp_path):
        db = tmp_path / "test.db"
        source = SAMPLE_SOURCE + "\n# bl_persist\n"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, source)
        gene_id = gp.get_top(1)[0]["id"]
        gp.delete_gene(gene_id)
        # Re-open the database.
        gp2 = GenePool(db, max_size=10)
        added = gp2.add(2, 0.95, source)
        assert added is False


class TestSimilarityDedup:
    """Near-duplicate detection via tag Jaccard similarity."""

    # Variant with a tiny comment change — same summary/tags as SAMPLE_SOURCE.
    VARIANT_SOURCE = SAMPLE_SOURCE + "\n# minor_tweak_xyz\n"

    def test_similar_higher_score_replaces(self, tmp_path):
        """A higher-score near-duplicate replaces the existing gene."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.70, SAMPLE_SOURCE)
        assert gp.count() == 1
        old_id = gp.get_top(1)[0]["id"]

        # Same summary, different source_hash, higher score → replace.
        added = gp.add(2, 0.90, self.VARIANT_SOURCE)
        assert added is True
        assert gp.count() == 1  # replaced, not added
        top = gp.get_top(1)[0]
        assert top["id"] == old_id  # same id preserved
        assert top["score"] == 0.90
        assert top["generation"] == 2

    def test_similar_lower_score_skipped(self, tmp_path):
        """A lower-score near-duplicate is skipped."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.90, SAMPLE_SOURCE)

        added = gp.add(2, 0.70, self.VARIANT_SOURCE)
        assert added is False
        assert gp.count() == 1
        assert gp.get_top(1)[0]["score"] == 0.90  # original kept

    def test_similar_preserves_hit_count(self, tmp_path):
        """Replacing a similar gene preserves its hit_count."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.70, SAMPLE_SOURCE)
        gene_id = gp.get_top(1)[0]["id"]
        gp.record_hits([gene_id], generation=5)
        gp.record_hits([gene_id], generation=6)

        gp.add(2, 0.90, self.VARIANT_SOURCE)
        top = gp.get_top(1)[0]
        assert top["hit_count"] == 2  # preserved

    def test_different_genes_not_affected(self, tmp_path):
        """Genes with genuinely different tags are not considered similar."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE)

        # Completely different gene.
        different_source = '''\
import os, pathlib, time, threading

class WeatherForecast:
    """Predict weather patterns using atmospheric pressure data."""

    def predict(self, pressure_readings):
        """Generate forecast from pressure readings."""
        pass

    def calibrate(self, historical_data):
        """Calibrate model with historical weather data."""
        pass

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
'''
        added = gp.add(2, 0.75, different_source)
        assert added is True
        assert gp.count() == 2  # both kept
