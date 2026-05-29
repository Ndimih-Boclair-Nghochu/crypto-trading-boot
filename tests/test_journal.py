from __future__ import annotations

import asyncio
import os

import pytest

from config import Settings
from db.connection import Database
from learning.journal import Journal, JournalEvent
from trading.strategy_engine import TradeSignal


class FakeDB:
    sessionmaker = None

    def __init__(self) -> None:
        self.executed = []

    async def execute(self, statement, params=None):
        self.executed.append((statement, params))


def run(coro):
    return asyncio.run(coro)


def test_journal_system_event_write_is_non_throwing() -> None:
    db = FakeDB()
    journal = Journal(db)  # type: ignore[arg-type]
    run(
        journal.write_event(
            JournalEvent(
                "system_event",
                {"event_type": "TEST", "severity": "INFO", "message": "hello", "context": {"ok": True}},
            )
        )
    )
    assert len(db.executed) == 1
    assert "system_events" in db.executed[0][0]


def test_journal_no_trade_serializes_indicator_state() -> None:
    db = FakeDB()
    journal = Journal(db)  # type: ignore[arg-type]
    signal = TradeSignal(
        symbol="BTCUSDT",
        direction="NO_TRADE",
        strategy_used="WAIT",
        regime_at_entry="MIXED",
        lstm_confidence=0.3,
        rl_confidence=1.0,
        confluence_score=20,
        indicator_agreement=1,
        indicator_state={"nested": {"value": 1}},
    )

    async def scenario():
        await journal.log_no_trade(signal, "gate failed")
        event = await journal.queue.get()
        await journal.write_event(event)

    run(scenario())
    assert len(db.executed) == 1
    assert "no_trade_log" in db.executed[0][0]
    assert "nested" in db.executed[0][1]["indicator_state"]


def test_postgres_roundtrip_when_test_database_url_is_set() -> None:
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL for PostgreSQL integration test")
    db = Database(Settings(database_url=url))

    async def scenario():
        await db.initialize()
        try:
            await db.run_migrations()
            await db.execute(
                """
                INSERT INTO system_events (event_type, severity, message, context)
                VALUES ('TEST', 'INFO', 'roundtrip', '{}'::jsonb)
                """
            )
            row = await db.fetch_one(
                "SELECT message FROM system_events WHERE event_type = 'TEST' ORDER BY occurred_at DESC LIMIT 1"
            )
            assert row and row["message"] == "roundtrip"
        finally:
            await db.close()

    run(scenario())
