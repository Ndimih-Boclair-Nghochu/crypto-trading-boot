from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from analysis.features import MODEL_FEATURES, build_model_features
from analysis.technical_analysis import enrich_indicators
from config import Settings, settings
from db.connection import Database, database
from models.lstm_model import LSTMModelService
from models.rl_agent import SEGMENT_KEY, RLAgentService
from utils.alerts import AlertManager
from utils.logger import logger


class LearningEngine:
    """Post-trade adaptation, with brakes.

    The previous version ran every 60 seconds and would retrain a symbol's LSTM
    on roughly 90 rows recovered from recent snapshots -- overwriting weights
    fitted on tens of thousands of bars. It also pulled RL training rows across
    all symbols at once and then deduplicated them by `open_time`, so candles
    sharing an hourly timestamp collapsed to one arbitrary symbol and the rest
    were discarded. Both paths made the models worse the longer the system ran.

    Every retrain here is now (a) symbol-scoped, (b) refused below a minimum row
    count, and (c) reverted if the new weights score worse out of sample than
    the ones they replaced.
    """

    def __init__(
        self,
        db: Database = database,
        lstm: LSTMModelService | None = None,
        rl: RLAgentService | None = None,
        cfg: Settings = settings,
    ) -> None:
        self.db = db
        self.settings = cfg
        self.lstm = lstm or LSTMModelService()
        self.rl = rl or RLAgentService()
        self.alerter = AlertManager()
        self.trade_counter = 0
        self.last_run: datetime | None = None
        self._last_regime: str | None = None

    async def record_state(self) -> None:
        self.trade_counter += 1
        if self.trade_counter >= 20:
            await self.run()

    def due(self) -> bool:
        if self.last_run is None:
            return True
        return datetime.now(UTC) - self.last_run >= timedelta(minutes=self.settings.learning_interval_minutes)

    async def run(self) -> None:
        self.last_run = datetime.now(UTC)
        self.trade_counter = 0
        try:
            await self.update_strategy_performance()
            await self.flag_and_retrain_weak_models()
            await self.run_rl_online_update()
            await self.detect_regime_shift()
        except Exception as exc:
            logger.exception(f"Learning engine failed without stopping trading: {exc}")

    # ---------------------------------------------------------- performance

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

    # -------------------------------------------------------------- retrain

    async def flag_and_retrain_weak_models(self) -> None:
        rows = await self.db.fetch_all(
            """
            SELECT symbol, avg(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS win_rate, count(*) AS n
            FROM (
                SELECT *, row_number() OVER (PARTITION BY symbol ORDER BY entry_time DESC) AS rn
                FROM trades
                WHERE outcome IN ('WIN', 'LOSS')
            ) ranked
            WHERE rn <= 50
            GROUP BY symbol
            HAVING count(*) >= 25 AND avg(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) < 0.45
            """
        )
        for row in rows:
            symbol = row["symbol"]
            await self._log_event(
                "MODEL_RETRAIN_TRIGGER",
                "WARNING",
                f"{symbol} win rate below 45% over the last {int(row['n'])} trades",
                {"symbol": symbol, "win_rate": str(row["win_rate"])},
            )
            frame = await self._symbol_frame(symbol, limit=600)
            if frame is None or len(frame) < self.settings.min_retrain_rows:
                have = 0 if frame is None else len(frame)
                logger.info(
                    f"{symbol}: skipping retrain, only {have} rows available "
                    f"(floor is {self.settings.min_retrain_rows}). Run --mode=download_data "
                    f"--mode=train_models for a full refit instead."
                )
                continue
            await self._retrain_symbol(symbol, frame)

    async def _retrain_symbol(self, symbol: str, frame: pd.DataFrame) -> None:
        weights = self.lstm.weights_dir
        weight_path = weights / f"lstm_{symbol}.pt"
        meta_path = weights / f"lstm_{symbol}.json"
        backup_weight = weights / f"lstm_{symbol}.previous.pt"
        backup_meta = weights / f"lstm_{symbol}.previous.json"

        previous_quality = self.lstm.model_quality(symbol)
        try:
            if weight_path.exists():
                shutil.copyfile(weight_path, backup_weight)
            if meta_path.exists():
                shutil.copyfile(meta_path, backup_meta)
        except Exception as exc:
            logger.warning(f"{symbol}: could not back up weights before retrain: {exc}")

        try:
            metrics = self.lstm.train(symbol, frame)
        except Exception as exc:
            logger.warning(f"{symbol}: retrain failed, keeping existing weights: {exc}")
            return

        new_quality = float(metrics.get("directional_precision", 0.0))
        if previous_quality > 0 and new_quality < previous_quality:
            # A retrain that scores worse out of sample is a regression, not an
            # update. Roll it back rather than trading on it.
            try:
                if backup_weight.exists():
                    shutil.copyfile(backup_weight, weight_path)
                if backup_meta.exists():
                    shutil.copyfile(backup_meta, meta_path)
                self.lstm._models.pop(symbol, None)
                self.lstm._meta.pop(symbol, None)
            except Exception as exc:
                logger.error(f"{symbol}: rollback after a worse retrain failed: {exc}")
            await self._log_event(
                "MODEL_RETRAIN_REJECTED",
                "WARNING",
                f"{symbol} retrain scored {new_quality:.3f} against {previous_quality:.3f}; reverted",
                {"symbol": symbol, "new": new_quality, "previous": previous_quality},
            )
            return

        logger.warning(f"Retrained LSTM for {symbol}: {metrics}")
        await self.alerter.send(f"MODEL RETRAINED: {symbol}", {"metrics": metrics, "previous_quality": previous_quality})
        await self._log_event(
            "MODEL_RETRAINED",
            "INFO",
            f"{symbol} LSTM retrained ({previous_quality:.3f} -> {new_quality:.3f})",
            {"symbol": symbol, **{k: float(v) for k, v in metrics.items()}},
        )

    async def run_rl_online_update(self) -> None:
        """Refresh the execution policy on per-symbol, chronologically ordered rows."""
        segments: list[dict[str, Any]] = []
        for index, symbol in enumerate(self.settings.symbols):
            frame = await self._symbol_frame(symbol, limit=400)
            if frame is None or len(frame) < 200:
                continue
            try:
                features = build_model_features(frame).ffill().fillna(0.0)
            except Exception as exc:
                logger.warning(f"{symbol}: RL feature build failed: {exc}")
                continue
            features["close"] = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
            features[SEGMENT_KEY] = index
            segments.extend(features.dropna(subset=["close"]).to_dict(orient="records"))

        if len(segments) < self.settings.min_rl_update_rows:
            logger.info(
                f"RL online update skipped: {len(segments)} rows across all symbols is below the "
                f"{self.settings.min_rl_update_rows}-row floor"
            )
            return
        result = self.rl.online_update(segments, min_rows=self.settings.min_rl_update_rows)
        if result:
            await self.alerter.send("RL ONLINE UPDATE COMPLETE", result)

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
        if total < 10 or not rows:
            return
        dominant = rows[0]
        if int(dominant["n"]) / total <= 0.75:
            return
        regime = str(dominant["regime_at_entry"])
        # Only worth an event when it actually changes. Emitting this on every
        # pass produced 226 warnings in 48 hours describing the same regime.
        if regime == self._last_regime:
            return
        self._last_regime = regime
        await self._log_event(
            "REGIME_SHIFT",
            "WARNING",
            f"Dominant 7-day regime shifted to {regime}",
            {"distribution": [{"regime": str(r["regime_at_entry"]), "n": int(r["n"])} for r in rows]},
        )

    # ------------------------------------------------------------- plumbing

    async def _log_event(self, event_type: str, severity: str, message: str, context: dict[str, Any]) -> None:
        try:
            await self.db.execute(
                """
                INSERT INTO system_events (event_type, severity, message, context)
                VALUES (:event_type, :severity, :message, CAST(:context AS JSONB))
                """,
                {
                    "event_type": event_type,
                    "severity": severity,
                    "message": message,
                    "context": json.dumps(context, default=str),
                },
            )
        except Exception as exc:
            logger.warning(f"Could not record {event_type}: {exc}")

    async def _symbol_frame(self, symbol: str, limit: int) -> pd.DataFrame | None:
        snapshots = await self.db.fetch_all(
            """
            SELECT raw_candles
            FROM market_snapshots
            WHERE symbol = :symbol AND raw_candles IS NOT NULL
            ORDER BY captured_at DESC
            LIMIT :limit
            """,
            {"symbol": symbol, "limit": limit},
        )
        return self._snapshots_to_frame(snapshots)

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
        if "open_time" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True, errors="coerce")
            frame = frame.set_index("timestamp", drop=False)
        elif "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.set_index("timestamp", drop=False)
        if "atr_14" not in frame.columns and all(column in frame.columns for column in ("high", "low", "close")):
            try:
                frame = enrich_indicators(frame).replace([float("inf"), float("-inf")], pd.NA).ffill()
            except Exception as exc:
                logger.warning(f"Could not enrich snapshot frame: {exc}")
        return frame if len(frame) > 10 else None


__all__ = ["LearningEngine", "MODEL_FEATURES"]
