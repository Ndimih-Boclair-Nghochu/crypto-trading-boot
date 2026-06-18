from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from analysis.narrative import build_no_trade_narrative
from db.connection import Database, database
from trading.risk_manager import TradePlan
from trading.strategy_engine import TradeSignal
from utils.logger import logger

try:
    from sqlalchemy import text
except Exception:  # pragma: no cover
    text = None


@dataclass(frozen=True)
class JournalEvent:
    kind: str
    payload: dict[str, Any]


class Journal:
    def __init__(self, db: Database = database, max_queue: int = 10_000) -> None:
        self.db = db
        self.queue: asyncio.Queue[JournalEvent] = asyncio.Queue(maxsize=max_queue)
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker and not self._worker.done():
            return
        self._worker = asyncio.create_task(self._run(), name="journal-writer")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    async def log_trade_open(self, plan: TradePlan, binance_order_id: str) -> None:
        await self._put(
            "trade_open",
            {
                "plan": plan,
                "binance_order_id": binance_order_id,
                "logged_at": datetime.now(UTC),
            },
        )

    async def log_trade_exit(self, symbol: str, exit_price: Decimal | float | str, exit_reason: str) -> None:
        await self._put(
            "trade_exit",
            {
                "symbol": symbol,
                "exit_price": str(exit_price),
                "exit_reason": exit_reason,
                "exit_time": datetime.now(UTC),
            },
        )

    async def log_partial_exit(
        self,
        symbol: str,
        exit_price: Decimal | float | str,
        quantity_closed: Decimal | float | str,
        pnl_usd: Decimal | float | str,
        r_multiple: Decimal | float | str,
        reason: str = "TP1",
    ) -> None:
        await self._put(
            "system_event",
            {
                "event_type": f"PARTIAL_EXIT_{reason}",
                "severity": "INFO",
                "message": f"{symbol} partial exit at {exit_price}",
                "context": {
                    "symbol": symbol,
                    "exit_price": str(exit_price),
                    "quantity_closed": str(quantity_closed),
                    "pnl_usd": str(pnl_usd),
                    "r_multiple": str(r_multiple),
                    "reason": reason,
                },
            },
        )

    async def log_no_trade(self, signal: TradeSignal, gate_failed: str) -> None:
        await self._put("no_trade", {"signal": signal, "gate_failed": gate_failed, "logged_at": datetime.now(UTC)})

    async def log_equity(
        self,
        balance_usdt: Decimal | float | str,
        open_pnl: Decimal | float | str,
        total_equity: Decimal | float | str,
        peak_equity: Decimal | float | str,
        drawdown_pct: Decimal | float | str,
    ) -> None:
        await self._put(
            "equity",
            {
                "balance_usdt": str(balance_usdt),
                "open_pnl": str(open_pnl),
                "total_equity": str(total_equity),
                "peak_equity": str(peak_equity),
                "drawdown_pct": str(drawdown_pct),
            },
        )

    async def log_event(self, event_type: str, severity: str, message: str, context: dict[str, Any] | None = None) -> None:
        await self._put(
            "system_event",
            {
                "event_type": event_type,
                "severity": severity,
                "message": message,
                "context": context or {},
            },
        )

    async def _put(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(JournalEvent(kind, payload))
        except asyncio.QueueFull:
            logger.critical(f"Journal queue full; dropping {kind} event")

    async def _run(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                await self.write_event(event)
            except Exception as exc:
                logger.exception(f"Journal write failed but trading loop continues: {exc}")
            finally:
                self.queue.task_done()

    async def write_event(self, event: JournalEvent) -> None:
        if event.kind == "trade_open":
            await self._write_trade_open(event.payload)
        elif event.kind == "trade_exit":
            await self._write_trade_exit(event.payload)
        elif event.kind == "no_trade":
            await self._write_no_trade(event.payload)
        elif event.kind == "equity":
            await self._write_equity(event.payload)
        elif event.kind == "system_event":
            await self._write_system_event(event.payload)

    async def _write_trade_open(self, payload: dict[str, Any]) -> None:
        if not self.db.sessionmaker or text is None:
            logger.info("Trade open journal skipped; database not initialized.")
            return
        plan: TradePlan = payload["plan"]
        async with self.db.sessionmaker() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        """
                        INSERT INTO trades (
                            symbol, direction, entry_price, sl_price, tp1_price, tp2_price, quantity,
                            entry_time, strategy_used, regime_at_entry, lstm_confidence, rl_confidence,
                            confluence_score, outcome, binance_order_id
                        ) VALUES (
                            :symbol, :direction, :entry_price, :sl_price, :tp1_price, :tp2_price, :quantity,
                            :entry_time, :strategy_used, :regime_at_entry, :lstm_confidence, :rl_confidence,
                            :confluence_score, 'OPEN', :binance_order_id
                        )
                        RETURNING trade_id
                        """
                    ),
                    {
                        "symbol": plan.symbol,
                        "direction": plan.direction,
                        "entry_price": str(plan.entry_price),
                        "sl_price": str(plan.sl_price),
                        "tp1_price": str(plan.tp1_price),
                        "tp2_price": str(plan.tp2_price) if plan.tp2_price else None,
                        "quantity": str(plan.quantity),
                        "entry_time": payload["logged_at"],
                        "strategy_used": plan.strategy_used,
                        "regime_at_entry": plan.regime_at_entry,
                        "lstm_confidence": plan.lstm_confidence,
                        "rl_confidence": plan.rl_confidence,
                        "confluence_score": plan.confluence_score,
                        "binance_order_id": payload["binance_order_id"],
                    },
                )
                trade_id = result.scalar_one()
                await session.execute(
                    text(
                        """
                        INSERT INTO market_snapshots (
                            trade_id, symbol, captured_at, timeframe, indicators, regime, raw_candles
                        ) VALUES (
                            :trade_id, :symbol, :captured_at, :timeframe,
                            CAST(:indicators AS JSONB), :regime, CAST(:raw_candles AS JSONB)
                        )
                        """
                    ),
                    {
                        "trade_id": trade_id,
                        "symbol": plan.symbol,
                        "captured_at": payload["logged_at"],
                        "timeframe": "1h",
                        "indicators": json.dumps(_jsonable(plan.indicator_state)),
                        "regime": plan.regime_at_entry,
                        "raw_candles": json.dumps(_jsonable(plan.indicator_state.get("raw_candles", []))),
                    },
                )

    async def _write_trade_exit(self, payload: dict[str, Any]) -> None:
        await self.db.execute(
            """
            UPDATE trades
            SET exit_price = CAST(:exit_price AS numeric),
                exit_time = :exit_time,
                exit_reason = :exit_reason,
                outcome = CASE
                    WHEN CAST(:exit_price AS numeric) > entry_price AND direction = 'LONG' THEN 'WIN'
                    WHEN CAST(:exit_price AS numeric) < entry_price AND direction = 'SHORT' THEN 'WIN'
                    WHEN CAST(:exit_price AS numeric) = entry_price THEN 'BREAKEVEN'
                    ELSE 'LOSS'
                END,
                pnl_usd = CASE
                    WHEN direction = 'LONG' THEN (CAST(:exit_price AS numeric) - entry_price) * quantity
                    ELSE (entry_price - CAST(:exit_price AS numeric)) * quantity
                END,
                pnl_pct = CASE
                    WHEN direction = 'LONG' THEN ((CAST(:exit_price AS numeric) - entry_price) / entry_price) * 100
                    ELSE ((entry_price - CAST(:exit_price AS numeric)) / entry_price) * 100
                END,
                r_multiple = CASE
                    WHEN abs(entry_price - sl_price) > 0 AND direction = 'LONG'
                        THEN (CAST(:exit_price AS numeric) - entry_price) / abs(entry_price - sl_price)
                    WHEN abs(entry_price - sl_price) > 0 AND direction = 'SHORT'
                        THEN (entry_price - CAST(:exit_price AS numeric)) / abs(entry_price - sl_price)
                    ELSE NULL
                END
            WHERE trade_id = (
                SELECT trade_id FROM trades
                WHERE symbol = :symbol AND outcome = 'OPEN'
                ORDER BY entry_time DESC
                LIMIT 1
            )
            """,
            payload,
        )

    async def _write_no_trade(self, payload: dict[str, Any]) -> None:
        signal: TradeSignal = payload["signal"]
        gate_reasons = list(getattr(signal, "reasons", []) or [payload["gate_failed"]])
        narrative = build_no_trade_narrative(
            symbol=signal.symbol,
            regime=signal.regime_at_entry,
            lstm_confidence=signal.lstm_confidence,
            confluence_score=signal.confluence_score,
            reasons=gate_reasons,
        )
        params = {
            "symbol": signal.symbol,
            "logged_at": payload["logged_at"],
            "regime": signal.regime_at_entry,
            "lstm_confidence": signal.lstm_confidence,
            "confluence_score": signal.confluence_score,
            "gate_failed": payload["gate_failed"][:100],
            "indicator_state": json.dumps(_jsonable(signal.indicator_state)),
            "gate_reasons": json.dumps(gate_reasons),
            "analysis_notes": narrative,
        }
        try:
            await self.db.execute(
                """
                INSERT INTO no_trade_log (
                    symbol, logged_at, regime, lstm_confidence, confluence_score, gate_failed,
                    indicator_state, gate_reasons, analysis_notes
                ) VALUES (
                    :symbol, :logged_at, :regime, :lstm_confidence, :confluence_score, :gate_failed,
                    CAST(:indicator_state AS JSONB), CAST(:gate_reasons AS JSONB), :analysis_notes
                )
                """,
                params,
            )
        except Exception as exc:
            # Falls back to the original columns only, in case this is
            # running against a database that hasn't been migrated to
            # 0002_no_trade_analysis_notes yet.
            logger.warning(f"no_trade_log insert with analysis_notes failed, retrying without it: {exc}")
            await self.db.execute(
                """
                INSERT INTO no_trade_log (
                    symbol, logged_at, regime, lstm_confidence, confluence_score, gate_failed, indicator_state
                ) VALUES (
                    :symbol, :logged_at, :regime, :lstm_confidence, :confluence_score, :gate_failed,
                    CAST(:indicator_state AS JSONB)
                )
                """,
                {k: v for k, v in params.items() if k in {
                    "symbol", "logged_at", "regime", "lstm_confidence", "confluence_score",
                    "gate_failed", "indicator_state",
                }},
            )

    async def _write_equity(self, payload: dict[str, Any]) -> None:
        await self.db.execute(
            """
            INSERT INTO equity_snapshots (balance_usdt, open_pnl, total_equity, peak_equity, drawdown_pct)
            VALUES (:balance_usdt, :open_pnl, :total_equity, :peak_equity, :drawdown_pct)
            """,
            payload,
        )

    async def _write_system_event(self, payload: dict[str, Any]) -> None:
        await self.db.execute(
            """
            INSERT INTO system_events (event_type, severity, message, context)
            VALUES (:event_type, :severity, :message, CAST(:context AS JSONB))
            """,
            {
                **payload,
                "context": json.dumps(_jsonable(payload.get("context", {}))),
            },
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
