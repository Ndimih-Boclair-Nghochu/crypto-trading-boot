from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from analysis.technical_analysis import enrich_indicators
from db.connection import Database, database
from models.lstm_model import LSTMModelService
from models.rl_agent import RLAgentService
from utils.alerts import AlertManager
from utils.logger import logger


class LearningEngine:
    def __init__(
        self,
        db: Database = database,
        lstm: LSTMModelService | None = None,
        rl: RLAgentService | None = None,
    ) -> None:
        self.db = db
        self.lstm = lstm or LSTMModelService()
        self.rl = rl or RLAgentService()
        self.alerter = AlertManager()
        self.trade_counter = 0
        self.last_weekly_run: datetime | None = None

    async def record_state(self) -> None:
        self.trade_counter += 1
        now = datetime.now(UTC)
        weekly_due = not self.last_weekly_run or now - self.last_weekly_run >= timedelta(days=7)
        if self.trade_counter >= 20 or weekly_due:
            await self.run()
            self.trade_counter = 0
            self.last_weekly_run = now

    async def run(self) -> None:
        try:
            await self.update_strategy_performance()
            await self.flag_and_retrain_weak_models()
            await self.run_rl_online_update()
            await self.detect_regime_shift()
        except Exception as exc:
            logger.exception(f"Learning engine failed without stopping trading: {exc}")

    async def update_strategy_performance(self) -> None:
        await self.db.execute(
            """
            INSERT INTO strategy_performance (
                strategy_name, regime, period_start, period_end, total_trades, wins, losses,
                win_rate, avg_r_multiple, profit_factor, max_drawdown
            )
            SELECT
                strategy_used,
                regime_at_entry,
                date_trunc('week', entry_time) AS period_start,
                date_trunc('week', entry_time) + interval '7 days' AS period_end,
                count(*) AS total_trades,
                count(*) FILTER (WHERE outcome = 'WIN') AS wins,
                count(*) FILTER (WHERE outcome = 'LOSS') AS losses,
                count(*) FILTER (WHERE outcome = 'WIN')::numeric / NULLIF(count(*), 0) AS win_rate,
                avg(r_multiple) AS avg_r_multiple,
                sum(greatest(pnl_usd, 0)) / NULLIF(abs(sum(least(pnl_usd, 0))), 0) AS profit_factor,
                COALESCE((SELECT max(drawdown_pct) FROM equity_snapshots), 0) AS max_drawdown
            FROM trades
            WHERE outcome IN ('WIN', 'LOSS', 'BREAKEVEN')
            GROUP BY strategy_used, regime_at_entry, date_trunc('week', entry_time)
            ON CONFLICT (strategy_name, regime, period_start)
            DO UPDATE SET
                period_end = EXCLUDED.period_end,
                total_trades = EXCLUDED.total_trades,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                win_rate = EXCLUDED.win_rate,
                avg_r_multiple = EXCLUDED.avg_r_multiple,
                profit_factor = EXCLUDED.profit_factor,
                max_drawdown = EXCLUDED.max_drawdown,
                updated_at = NOW()
            """
        )

    async def flag_and_retrain_weak_models(self) -> None:
        rows = await self.db.fetch_all(
            """
            SELECT symbol, strategy_used, avg(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS win_rate, count(*) AS n
            FROM (
                SELECT *, row_number() OVER (PARTITION BY strategy_used ORDER BY entry_time DESC) AS rn
                FROM trades
                WHERE outcome IN ('WIN', 'LOSS')
            ) ranked
            WHERE rn <= 50
            GROUP BY symbol, strategy_used
            HAVING count(*) >= 20 AND avg(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) < 0.45
            """
        )
        for row in rows:
            symbol = row["symbol"]
            snapshots = await self.db.fetch_all(
                """
                SELECT raw_candles
                FROM market_snapshots
                WHERE symbol = :symbol AND raw_candles IS NOT NULL
                ORDER BY captured_at DESC
                LIMIT 200
                """,
                {"symbol": symbol},
            )
            frame = self._snapshots_to_frame(snapshots)
            if frame is not None and len(frame) > 100:
                metrics = self.lstm.train(symbol, frame)
                logger.warning(f"Retrained LSTM for {symbol} after weak win rate: {metrics}")
                await self.alerter.send(
                    f"MODEL RETRAINED: {symbol} LSTM",
                    {"reason": "win_rate_below_45_percent", "metrics": metrics},
                )
            rl_snapshots = await self.db.fetch_all(
                """
                SELECT raw_candles
                FROM market_snapshots
                WHERE symbol = :symbol AND raw_candles IS NOT NULL
                ORDER BY captured_at DESC
                LIMIT 100
                """,
                {"symbol": symbol},
            )
            rl_frame = self._snapshots_to_frame(rl_snapshots)
            if rl_frame is not None and len(rl_frame) > 50:
                self.rl.online_update(rl_frame.to_dict(orient="records"), timesteps=2_000)
                logger.info(f"RL agent online update complete for {symbol}")
                await self.alerter.send(
                    f"RL ONLINE UPDATE COMPLETE: {symbol}",
                    {"samples": len(rl_frame), "timesteps": 2_000},
                )
            await self.db.execute(
                """
                INSERT INTO system_events (event_type, severity, message, context)
                VALUES ('MODEL_RETRAIN_TRIGGER', 'WARNING', :message, CAST(:context AS JSONB))
                """,
                {
                    "message": f"{symbol} {row['strategy_used']} win rate below 45%",
                    "context": json.dumps(
                        {"symbol": symbol, "strategy": row["strategy_used"], "win_rate": str(row["win_rate"])}
                    ),
                },
            )

    async def run_rl_online_update(self) -> None:
        """Run RL online update from the most recent market snapshots across all symbols."""
        try:
            rows = await self.db.fetch_all(
                """
                SELECT raw_candles
                FROM market_snapshots
                WHERE raw_candles IS NOT NULL
                ORDER BY captured_at DESC
                LIMIT 200
                """
            )
            frame = self._snapshots_to_frame(rows)
            if frame is not None and len(frame) > 50:
                self.rl.online_update(frame.to_dict(orient="records"), timesteps=2_000)
                await self.alerter.send(
                    "RL ONLINE UPDATE COMPLETE",
                    {"samples": len(frame), "timesteps": 2_000},
                )
        except Exception as exc:
            logger.warning(f"RL online update skipped: {exc}")

    async def detect_regime_shift(self) -> None:
        rows = await self.db.fetch_all(
            """
            SELECT regime_at_entry, count(*) AS n
            FROM trades
            WHERE entry_time >= NOW() - interval '7 days'
            GROUP BY regime_at_entry
            ORDER BY n DESC
            """
        )
        total = sum(int(row["n"]) for row in rows)
        if total < 10:
            return
        dominant = rows[0]
        if int(dominant["n"]) / total > 0.75:
            await self.db.execute(
                """
                INSERT INTO system_events (event_type, severity, message, context)
                VALUES ('REGIME_SHIFT', 'WARNING', :message, CAST(:context AS JSONB))
                """,
                {
                    "message": f"Dominant 7-day regime shifted to {dominant['regime_at_entry']}",
                    "context": json.dumps({"distribution": rows}, default=str),
                },
            )

    def _snapshots_to_frame(self, snapshots: list[dict[str, Any]]) -> pd.DataFrame | None:
        rows: list[dict[str, Any]] = []
        for snapshot in snapshots:
            raw = snapshot.get("raw_candles") or []
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = []
            if isinstance(raw, list):
                rows.extend(item for item in raw if isinstance(item, dict))
        if not rows:
            return None
        frame = pd.DataFrame(rows)
        if "open_time" in frame.columns:
            frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
        else:
            frame = frame.drop_duplicates().reset_index(drop=True)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.set_index("timestamp", drop=False)
        elif "open_time" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True, errors="coerce")
            frame = frame.set_index("timestamp", drop=False)
        if "atr_14" not in frame.columns and all(column in frame.columns for column in ("high", "low", "close")):
            try:
                frame = enrich_indicators(frame).replace([float("inf"), float("-inf")], pd.NA).ffill().bfill()
            except Exception as exc:
                logger.warning(f"Could not enrich snapshot frame: {exc}")
        return frame if len(frame) > 10 else None
