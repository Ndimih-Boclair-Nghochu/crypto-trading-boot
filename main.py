from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.technical_analysis import TechnicalAnalysisEngine, candles_to_frame, enrich_indicators
from config import settings
from data.data_pipeline import MarketDataPipeline
from db.connection import database
from learning.journal import Journal
from learning.learning_engine import LearningEngine
from models.confidence_gate import ConfidenceGate
from models.lstm_model import LSTMModelService
from models.rl_agent import RLAgentService
from trading.execution_engine import ExecutionEngine
from trading.risk_manager import RiskManager, TradeCandidate
from trading.strategy_engine import StrategyEngine
from utils.alerts import AlertManager
from utils.binance_client import ResilientBinanceClient
from utils.logger import logger


class TradingSystem:
    def __init__(self) -> None:
        self.client = ResilientBinanceClient()
        self.data = MarketDataPipeline(self.client)
        self.ta = TechnicalAnalysisEngine()
        self.lstm = LSTMModelService()
        self.rl = RLAgentService()
        self.gate = ConfidenceGate()
        self.strategy = StrategyEngine()
        self.risk = RiskManager()
        self.journal = Journal()
        self.execution = ExecutionEngine(self.client, self.risk, self.journal)
        self.learning = LearningEngine(database, self.lstm, self.rl)
        self.alerter = AlertManager()
        self._reconcile_at = datetime.now(UTC)

    async def start(self) -> None:
        if not settings.use_testnet:
            settings.assert_live_trading_allowed()
        await self._set_status("STARTING", reason="Connecting to database and Binance")
        await database.initialize()
        await database.run_migrations()
        await self.journal.start()
        await self.client.initialize()
        self.client.start_health_check()
        await self.journal.log_event("SYSTEM_START", "INFO", "Trading system started", {"testnet": settings.use_testnet})
        await self._ensure_models_trained()

    async def _ensure_models_trained(self) -> None:
        """Train the LSTM and RL models on first run if no weights exist.

        Without this, models/lstm_model.py and models/rl_agent.py silently
        fall back to confidence=0.0 / action=HOLD whenever weights are
        missing -- which the confidence gate then (correctly) refuses to
        trade on, forever, with no error or crash. The system looked
        "healthy" (connected, analyzing) while being structurally unable to
        ever take a trade, because training was a separate manual CLI step
        that was never run against this deployment.
        """
        weights_dir = settings.runtime_dir.parent / "models" / "weights"
        lstm_missing = [s for s in settings.symbols if not (weights_dir / f"lstm_{s}.pt").exists()]
        rl_missing = not (weights_dir / "ppo_trading_agent.zip").exists()
        if not lstm_missing and not rl_missing:
            logger.info("Model weights found for all symbols; skipping training bootstrap.")
            return

        logger.warning(
            f"No trained weights found (missing LSTM for {lstm_missing or 'none'}, "
            f"RL missing={rl_missing}). Bootstrapping training now -- this runs once and "
            f"may take several minutes; the bot will not trade until it completes."
        )

        async def _heartbeat() -> None:
            # Training can run far longer than the dashboard's 120s
            # staleness window. Without this, a legitimate multi-minute
            # training run gets misreported as UNRESPONSIVE even though
            # the process is healthy and working.
            while True:
                await asyncio.sleep(45)
                await self._set_status(
                    "TRAINING",
                    reason=(
                        "Training the LSTM/RL models for the first time (one-time setup). "
                        "This can take several minutes; the bot will start analyzing once it finishes."
                    ),
                )

        await self._set_status(
            "TRAINING",
            reason=(
                "No trained models found. Downloading historical data and training the LSTM/RL "
                "models for the first time. This is a one-time step and may take several minutes."
            ),
        )
        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            await download_data()
            await train_models()

            # train_models() intentionally isolates per-symbol failures (so
            # one bad symbol's data doesn't abort the whole run) and always
            # logs "completed" regardless. That previously meant an
            # environment-level failure -- e.g. PyTorch failing to import,
            # which affects every symbol identically -- silently produced
            # zero weight files while still being reported as a success on
            # every restart. Verify on disk what was actually produced.
            lstm_still_missing = [s for s in settings.symbols if not (weights_dir / f"lstm_{s}.pt").exists()]
            rl_still_missing = not (weights_dir / "ppo_trading_agent.zip").exists()
            if lstm_still_missing and len(lstm_still_missing) == len(settings.symbols):
                raise RuntimeError(
                    "Training ran but produced no LSTM weight files for any symbol. This almost "
                    "always means PyTorch failed to import in this container -- check the Render "
                    "build/runtime logs for a 'PyTorch failed to import' error near startup."
                )
            if lstm_still_missing:
                logger.warning(f"LSTM training did not produce weights for: {lstm_still_missing}")
            if rl_still_missing:
                logger.warning("RL training did not produce a weights file (ppo_trading_agent.zip missing).")

            await self.journal.log_event(
                "MODEL_TRAINING",
                "INFO",
                "Initial model training completed",
                {
                    "symbols": list(settings.symbols),
                    "lstm_missing_after_training": lstm_still_missing,
                    "rl_missing_after_training": rl_still_missing,
                },
            )
            logger.info("Initial model training completed.")
        except Exception as exc:
            logger.exception(f"Initial model training failed: {exc}")
            await self.journal.log_event("MODEL_TRAINING_FAILED", "CRITICAL", str(exc), {})
            await self._set_status(
                "ERROR",
                reason=(
                    f"Initial model training failed ({exc}). The bot cannot generate trade signals "
                    "without trained models. Check Render logs for details."
                ),
            )
            raise
        finally:
            heartbeat_task.cancel()

    async def stop(self) -> None:
        await self.journal.log_event("SYSTEM_STOP", "INFO", "Trading system stopped", {})
        await self.journal.stop()
        await self.client.close()
        await database.close()

    async def main_loop(self) -> None:
        try:
            await self.start()
        except Exception as exc:
            logger.error(f"Startup failed: {exc}")
            await self._set_status("ERROR", reason=f"Startup failed: {exc}")
            raise
        try:
            while True:
                try:
                    if not self.trading_enabled():
                        await self._set_status("PAUSED", reason="FORCE_TRADING_PAUSED is set")
                        await asyncio.sleep(5)
                        continue

                    await self._set_status("ANALYZING")
                    snapshot = await self.data.update_all()
                    meta = snapshot["meta"]
                    for symbol in settings.symbols:
                        candles_by_tf = snapshot["candles"].get(symbol, {})
                        analysis = self.ta.compute_all(candles_by_tf, meta=meta)
                        if not analysis.get("timeframes"):
                            continue

                        primary = self.strategy.primary_frame(analysis)
                        lstm_signal = self.lstm.predict(symbol, primary)
                        latest_for_state = self.strategy.primary_latest(analysis)
                        existing_position = self.risk.open_positions.get(symbol)
                        open_pnl = 0.0
                        position_flag = 0.0
                        entry_price = 0.0
                        if existing_position:
                            current_price = float(latest_for_state.get("close", 0) or 0)
                            if existing_position.direction == "LONG":
                                open_pnl = float(existing_position.quantity) * (
                                    current_price - float(existing_position.entry_price)
                                )
                                position_flag = 1.0
                            else:
                                open_pnl = float(existing_position.quantity) * (
                                    float(existing_position.entry_price) - current_price
                                )
                                position_flag = -1.0
                            entry_price = float(existing_position.entry_price)
                        peak = float(self.risk.peak_equity) if self.risk.peak_equity > 0 else 1.0
                        current_eq = float(self.risk.current_equity) if self.risk.current_equity > 0 else 1.0
                        drawdown = max(0.0, (peak - current_eq) / peak)
                        rl_state = self.strategy.build_rl_state(
                            analysis,
                            lstm_signal,
                            open_pnl=open_pnl,
                            drawdown=drawdown,
                            position=position_flag,
                            entry_price=entry_price,
                        )
                        rl_decision = self.rl.decide(rl_state)
                        analysis["regime"] = str(self.strategy.classify_regime(analysis, meta.fear_greed).value)
                        gate = await self.gate.passes(lstm_signal, rl_decision, analysis, symbol)
                        signal = self.strategy.build_trade_signal(symbol, analysis, lstm_signal, rl_decision, gate, meta.fear_greed)

                        if not gate.approved:
                            await self.journal.log_no_trade(signal, gate.failed_gate or "confidence gate failed")
                            continue

                        latest = signal.indicator_state.get("latest", {})
                        balance = await self.client.get_usdt_balance()
                        if balance <= 0:
                            await self.journal.log_event("BALANCE_CHECK", "WARNING", "USDT balance unavailable or zero", {"symbol": symbol})
                            continue
                        candidate = TradeCandidate(
                            signal=signal,
                            entry_price=Decimal(str(latest.get("close", "0") or "0")),
                            atr=Decimal(str(latest.get("atr_14", "0") or "0")),
                            account_balance=balance,
                            available_balance=balance,
                            next_support=_nearest_level(latest.get("patterns", {}).get("support_levels", []), below=True, price=latest.get("close")),
                            next_resistance=_nearest_level(
                                latest.get("patterns", {}).get("resistance_levels", []), below=False, price=latest.get("close")
                            ),
                        )
                        plan = await self.risk.calculate(candidate)
                        if not plan.approved:
                            blocked_signal = signal
                            await self.journal.log_no_trade(blocked_signal, plan.reason)
                            continue

                        result = await self.execution.place_trade(plan)
                        if result.accepted:
                            await self.learning_tick()
                        else:
                            await self.journal.log_event("ORDER_REJECTED", "WARNING", result.reason or "order rejected", {"symbol": symbol})

                    await self._write_equity_snapshot()
                    if self.risk.circuit_breaker_active and self.execution.open_trades:
                        logger.critical("CIRCUIT BREAKER ACTIVE - emergency closing all positions")
                        await self.journal.log_event(
                            "CIRCUIT_BREAKER_TRIGGERED",
                            "CRITICAL",
                            f"Drawdown exceeded {settings.drawdown_circuit_breaker_pct}% - closing all positions",
                            {"open_trades": list(self.execution.open_trades.keys())},
                        )
                        await self.execution.emergency_close_all()
                        await self.alerter.send(
                            "CIRCUIT BREAKER TRIGGERED",
                            {
                                "reason": f"Drawdown > {settings.drawdown_circuit_breaker_pct}%",
                                "all_positions_closed": True,
                            },
                        )

                    if datetime.now(UTC) >= self._reconcile_at:
                        await self.execution.reconcile_orders()
                        self._reconcile_at = datetime.now(UTC) + timedelta(minutes=5)
                    await asyncio.sleep(30)
                except Exception as exc:
                    logger.exception(f"Unexpected main loop error; sleeping then resuming: {exc}")
                    await self.journal.log_event("MAIN_LOOP_ERROR", "CRITICAL", str(exc), {})
                    await self._set_status("ERROR", reason=str(exc))
                    await asyncio.sleep(60)
        finally:
            await self.stop()

    async def learning_tick(self) -> None:
        await self.learning.record_state()

    async def _get_current_price(self, symbol: str) -> Decimal:
        candles = await self.client.get_ohlcv(symbol, "1m", 1)
        return candles[-1].close if candles else Decimal("0")

    async def _write_equity_snapshot(self) -> None:
        try:
            balance = await self.client.get_usdt_balance()
            open_pnl = Decimal("0")
            for position in self.risk.open_positions.values():
                current_price = await self._get_current_price(position.symbol)
                if current_price <= 0:
                    continue
                if position.direction == "LONG":
                    open_pnl += position.quantity * (current_price - position.entry_price)
                else:
                    open_pnl += position.quantity * (position.entry_price - current_price)
            total_equity = Decimal(str(balance)) + open_pnl
            self.risk.circuit_breaker_hit(total_equity)
            peak = self.risk.peak_equity if self.risk.peak_equity > 0 else total_equity
            drawdown_pct = ((peak - total_equity) / peak * Decimal("100")) if peak > 0 else Decimal("0")
            await self.journal.log_equity(
                balance_usdt=balance,
                open_pnl=open_pnl,
                total_equity=total_equity,
                peak_equity=peak,
                drawdown_pct=drawdown_pct,
            )
        except Exception as exc:
            logger.warning(f"Equity snapshot failed: {exc}")

    def trading_enabled(self) -> bool:
        # The system is always on. Kept as a method (rather than inlined
        # 'True') so any future kill-switch (e.g. FORCE_TRADING_PAUSED for
        # ops/incident response) has one obvious place to live.
        return os.getenv("FORCE_TRADING_PAUSED", "").strip().lower() not in {"1", "true", "yes"}

    async def _set_status(self, status: str, *, reason: str | None = None) -> None:
        path = settings.trading_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "trading_enabled": self.trading_enabled(),
            "status": status,
            "reason": reason,
            "testnet": settings.use_testnet,
            "binance_connected": bool(getattr(self.client, "connected", False)),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")


async def migrate() -> None:
    await database.initialize()
    await database.run_migrations()
    await database.close()


async def download_data() -> None:
    client = ResilientBinanceClient()
    await client.initialize()
    end = datetime.now(UTC)
    start = end - timedelta(days=365 * 2)
    for i, symbol in enumerate(settings.symbols, start=1):
        logger.info(f"[{i}/{len(settings.symbols)}] Downloading two years of 1h data for {symbol}")
        try:
            candles = await client.get_historical_ohlcv(symbol, "1h", int(start.timestamp() * 1000), int(end.timestamp() * 1000))
            frame = pd.DataFrame([c.to_dict() for c in candles])
            path = settings.runtime_dir / f"training_{symbol}_1h.csv"
            frame.to_csv(path, index=False)
            logger.info(f"Wrote {len(frame)} candles to {path}")
        except Exception as exc:
            logger.warning(f"Historical data download failed for {symbol}, skipping: {exc}")
    await client.close()


async def train_models() -> None:
    from models.lstm_model import torch as _torch  # local import: reflects current module state

    ta = TechnicalAnalysisEngine()
    lstm = LSTMModelService()
    rl_rows: list[dict[str, float]] = []
    if _torch is None:
        raise RuntimeError(
            "PyTorch is unavailable in this environment, so no LSTM model can be trained for any "
            "symbol. Check the build/runtime logs for a 'PyTorch failed to import' error."
        )
    for i, symbol in enumerate(settings.symbols, start=1):
        path = settings.runtime_dir / f"training_{symbol}_1h.csv"
        if not path.exists():
            logger.warning(f"Training file missing for {symbol}: run --mode=download_data first")
            continue
        try:
            logger.info(f"[{i}/{len(settings.symbols)}] Training LSTM for {symbol}")
            raw = pd.read_csv(path)
            candles = [
                _row_to_candle(symbol, "1h", row)
                for _, row in raw.iterrows()
                if float(row.get("volume", 0) or 0) > 0
            ]
            frame = enrich_indicators(candles_to_frame(candles)).replace([float("inf"), float("-inf")], pd.NA).ffill().bfill()
            metrics = lstm.train(symbol, frame)
            logger.info(f"LSTM metrics for {symbol}: {metrics}")
            rl_rows.extend(frame.tail(2_000).to_dict(orient="records"))
        except Exception as exc:
            logger.warning(f"LSTM training failed for {symbol}, skipping: {exc}")
    if rl_rows:
        logger.info(f"Training RL agent on {len(rl_rows)} rows")
        try:
            RLAgentService().train(rl_rows)
        except Exception as exc:
            logger.warning(f"RL agent training failed, continuing without it: {exc}")


def _row_to_candle(symbol: str, timeframe: str, row: pd.Series) -> Any:
    from data.market_data import Candle

    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open_time=int(row["open_time"]),
        open=Decimal(str(row["open"])),
        high=Decimal(str(row["high"])),
        low=Decimal(str(row["low"])),
        close=Decimal(str(row["close"])),
        volume=Decimal(str(row["volume"])),
        close_time=int(row["close_time"]),
    )


def _nearest_level(levels: list[Any], below: bool, price: Any) -> Decimal | None:
    try:
        price_d = Decimal(str(price))
        parsed = [Decimal(str(level)) for level in levels]
        candidates = [level for level in parsed if level < price_d] if below else [level for level in parsed if level > price_d]
        if not candidates:
            return None
        return max(candidates) if below else min(candidates)
    except Exception:
        return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "migrate", "download_data", "train_models"], default="run")
    args = parser.parse_args()
    if args.mode == "migrate":
        await migrate()
    elif args.mode == "download_data":
        await download_data()
    elif args.mode == "train_models":
        await train_models()
    else:
        await TradingSystem().main_loop()


if __name__ == "__main__":
    asyncio.run(main())
