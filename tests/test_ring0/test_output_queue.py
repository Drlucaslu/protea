"""Tests for ring0.output_queue."""

from __future__ import annotations

import time

import pytest

from ring0.output_queue import OutputQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def oq(tmp_path):
    return OutputQueue(tmp_path / "protea.db")


# ---------------------------------------------------------------------------
# TestBasic
# ---------------------------------------------------------------------------

class TestBasic:
    def test_creates_table(self, tmp_path):
        oq = OutputQueue(tmp_path / "protea.db")
        assert oq.count() == 0

    def test_idempotent_init(self, tmp_path):
        db = tmp_path / "protea.db"
        OutputQueue(db)
        OutputQueue(db)  # should not raise


# ---------------------------------------------------------------------------
# TestAddAndGetPending
# ---------------------------------------------------------------------------

class TestAddAndGetPending:
    def test_add_and_get_pending(self, oq):
        item_id = oq.add(gene_id=1, generation=5, capability="BtcMonitor", summary="Monitors BTC price")
        assert item_id is not None
        pending = oq.get_pending()
        assert len(pending) == 1
        assert pending[0]["capability"] == "BtcMonitor"
        assert pending[0]["status"] == "pending"

    def test_add_without_gene_id(self, oq):
        item_id = oq.add(gene_id=None, generation=3, capability="TestCap", summary="Test")
        assert item_id is not None
        item = oq.get_by_id(item_id)
        assert item["gene_id"] is None

    def test_get_pending_limit(self, oq):
        for i in range(5):
            oq.add(gene_id=i, generation=1, capability=f"Cap{i}", summary="s")
        assert len(oq.get_pending(limit=3)) == 3

    def test_get_pending_order(self, oq):
        oq.add(gene_id=1, generation=1, capability="First", summary="s")
        oq.add(gene_id=2, generation=1, capability="Second", summary="s")
        pending = oq.get_pending()
        assert pending[0]["capability"] == "First"


# ---------------------------------------------------------------------------
# TestMarkDelivered
# ---------------------------------------------------------------------------

class TestMarkDelivered:
    def test_mark_delivered(self, oq):
        item_id = oq.add(gene_id=1, generation=1, capability="X", summary="s")
        oq.mark_delivered(item_id, telegram_msg_id=42)
        item = oq.get_by_id(item_id)
        assert item["status"] == "delivered"
        assert item["telegram_msg_id"] == 42
        assert item["delivered_at"] is not None
        # No longer in pending
        assert len(oq.get_pending()) == 0


# ---------------------------------------------------------------------------
# TestMarkFeedback
# ---------------------------------------------------------------------------

class TestMarkFeedback:
    def test_mark_feedback_accepted(self, oq):
        item_id = oq.add(gene_id=1, generation=1, capability="X", summary="s")
        oq.mark_feedback(item_id, "accepted")
        item = oq.get_by_id(item_id)
        assert item["status"] == "accepted"
        assert item["feedback_at"] is not None

    def test_mark_feedback_rejected(self, oq):
        item_id = oq.add(gene_id=1, generation=1, capability="X", summary="s")
        oq.mark_feedback(item_id, "rejected", feedback_text="not useful")
        item = oq.get_by_id(item_id)
        assert item["status"] == "rejected"
        assert item["feedback_text"] == "not useful"

    def test_mark_feedback_scheduled(self, oq):
        item_id = oq.add(gene_id=1, generation=1, capability="X", summary="s")
        oq.mark_feedback(item_id, "scheduled")
        item = oq.get_by_id(item_id)
        assert item["status"] == "scheduled"


# ---------------------------------------------------------------------------
# TestExpireOld
# ---------------------------------------------------------------------------

class TestExpireOld:
    def test_expire_old(self, oq):
        item_id = oq.add(gene_id=1, generation=1, capability="X", summary="s")
        # Manually set created_at to 2 days ago
        with oq._connect() as con:
            con.execute(
                "UPDATE output_queue SET created_at = ? WHERE id = ?",
                (time.time() - 48 * 3600, item_id),
            )
        expired = oq.expire_old(max_age_hours=24)
        assert expired == 1
        item = oq.get_by_id(item_id)
        assert item["status"] == "expired"

    def test_expire_old_skips_recent(self, oq):
        oq.add(gene_id=1, generation=1, capability="X", summary="s")
        expired = oq.expire_old(max_age_hours=24)
        assert expired == 0

    def test_expire_old_skips_accepted(self, oq):
        item_id = oq.add(gene_id=1, generation=1, capability="X", summary="s")
        oq.mark_feedback(item_id, "accepted")
        with oq._connect() as con:
            con.execute(
                "UPDATE output_queue SET created_at = ? WHERE id = ?",
                (time.time() - 48 * 3600, item_id),
            )
        expired = oq.expire_old(max_age_hours=24)
        assert expired == 0  # accepted items should not expire


# ---------------------------------------------------------------------------
# TestGetAcceptedAndRejected
# ---------------------------------------------------------------------------

class TestGetAcceptedAndRejected:
    def test_get_accepted(self, oq):
        id1 = oq.add(gene_id=1, generation=1, capability="A", summary="s")
        id2 = oq.add(gene_id=2, generation=1, capability="B", summary="s")
        id3 = oq.add(gene_id=3, generation=1, capability="C", summary="s")
        oq.mark_feedback(id1, "accepted")
        oq.mark_feedback(id2, "scheduled")
        oq.mark_feedback(id3, "rejected")
        accepted = oq.get_accepted()
        assert len(accepted) == 2
        caps = {a["capability"] for a in accepted}
        assert caps == {"A", "B"}

    def test_get_rejected(self, oq):
        id1 = oq.add(gene_id=1, generation=1, capability="A", summary="s")
        id2 = oq.add(gene_id=2, generation=1, capability="B", summary="s")
        oq.mark_feedback(id1, "rejected")
        oq.mark_feedback(id2, "accepted")
        rejected = oq.get_rejected()
        assert len(rejected) == 1
        assert rejected[0]["capability"] == "A"


# ---------------------------------------------------------------------------
# TestDailyPushCount
# ---------------------------------------------------------------------------

class TestDailyPushCount:
    def test_daily_push_count(self, oq):
        assert oq.daily_push_count() == 0
        oq.add(gene_id=1, generation=1, capability="A", summary="s")
        oq.add(gene_id=2, generation=1, capability="B", summary="s")
        assert oq.daily_push_count() == 2

    def test_daily_push_count_excludes_old(self, oq):
        item_id = oq.add(gene_id=1, generation=1, capability="A", summary="s")
        with oq._connect() as con:
            con.execute(
                "UPDATE output_queue SET created_at = ? WHERE id = ?",
                (time.time() - 2 * 86400, item_id),
            )
        assert oq.daily_push_count() == 0


# ---------------------------------------------------------------------------
# TestGetByTelegramMsgId
# ---------------------------------------------------------------------------

class TestGetByTelegramMsgId:
    def test_get_by_telegram_msg_id(self, oq):
        item_id = oq.add(gene_id=1, generation=1, capability="X", summary="s")
        oq.mark_delivered(item_id, telegram_msg_id=999)
        found = oq.get_by_telegram_msg_id(999)
        assert found is not None
        assert found["id"] == item_id

    def test_get_by_telegram_msg_id_not_found(self, oq):
        assert oq.get_by_telegram_msg_id(12345) is None
