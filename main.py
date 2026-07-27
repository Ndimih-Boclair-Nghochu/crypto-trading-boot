from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.features import build_model_features
from analysis.technical_analysis import TechnicalAnalysisEngine, candles_to_frame, enrich_indicators
from config import settings
from data.data_pipeline import MarketDataPipeline
from db.connection import database
from learning.journal import Journal
from learning.learning_engine import LearningEngine
from models.confidence_gate import ConfidenceGate
from models.lstm_model import MODEL_VERSION, LSTMModelService
from models.rl_agent import AGENT_VERSION, SEGMENT_KEY, RLAgentService
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
        # Bar id of the last candle each symbol was evaluated on. Entries are
        # taken once per new bar rather than on every 30-second tick: with a 1h
        # primary timeframe the old loop re-scored an identical, unchanged
        # feature vector 120 times per candle.
        self._last_bar: dict[str, int] = {}
        # Dedicated fast task so the Stop button acts within ~1s instead of
        # waiting for the 30s analysis cycle to come around.
        self._close_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not settings.use_testnet:
            settings.assert_live_trading_allowed()
        await self._set_status("STARTING", reason="Connecting to database and Binance")
        await database.initialize()
        await database.run_migrations()
        await self.journal.start()
        await self.client.initialize()
        self.client.start_health_check()
        # Watch the manual-stop request file on a tight interval, independent of
        # the main analysis loop, so pressing Stop closes the position promptly.
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close_request_loop(), name="close-watcher")
        await self.journal.log_event(
            "SYSTEM_START",
            "INFO",
            "Trading system started",
            {
                "testnet": settings.use_testnet,
                "market_type": settings.market_type,
                "shorting_available": settings.shorting_available,
                "primary_timeframe": settings.primary_timeframe,
                "training_data_source": settings.training_data_source,
            },
        )
        if not settings.shorting_available:
            logger.warning(
                "MARKET_TYPE=%s: SHORT setups will be rejected by the gate because a spot account "
                "has nothing to sell. Set MARKET_TYPE=futures to trade both directions.",
                settings.market_type,
            )
        await self._ensure_models_trained()

    async def _ensure_models_trained(self) -> None:
        """Train on first run, and whenever the saved models predate a schema change.

        Without this the model services fall back to confidence 0.0 / HOLD when
        weights are missing, which the gate then correctly refuses to trade on,
        forever, with no error. The version check matters just as much: weights
        written against an older feature layout would otherwise be loaded and
        read with features in the wrong slots.
        """
        weights_dir = self.lstm.weights_dir
        lstm_missing = [s for s in settings.symbols if not _lstm_current(weights_dir, s)]
        rl_missing = not _rl_current(weights_dir)
        if not lstm_missing and not rl_missing:
            logger.info("Current model weights found for all symbols; skipping training bootstrap.")
            return

        logger.warning(
            f"Training required (stale or missing LSTM for {lstm_missing or 'none'}, RL missing={rl_missing}). "
            f"This runs once and may take several minutes; the bot will not trade until it completes."
        )
        _log_memory("bootstrap start")

        async def _heartbeat() -> None:
            while True:
                await asyncio.sleep(45)
                await self._set_status(
                    "TRAINING",
                    reason=(
                        "Training the LSTM/RL models (one-time setup after a model change). "
                        "This can take several minutes; the bot will start analysing once it finishes."
                    ),
                )

        await self._set_status(
            "TRAINING",
            reason="Downloading historical data and training the price and execution models.",
        )
        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            await download_data()
            await train_models()

            still_missing = [s for s in settings.symbols if not _lstm_current(weights_dir, s)]
            if still_missing and len(still_missing) == len(settings.symbols):
                raise RuntimeError(
                    "Training ran but produced no usable LSTM weights for any symbol. Check the logs "
                    "just above for the per-symbol error, or a 'PyTorch failed to import' message near startup."
                )
            if still_missing:
                logger.warning(f"LSTM training did not produce weights for: {still_missing}")
            if not _rl_current(weights_dir):
                logger.warning("RL training did not produce a usable policy; the gate will treat it as neutral.")

            await self.journal.log_event(
                "MODEL_TRAINING",
                "INFO",
                "Model training completed",
                {"symbols": list(settings.symbols), "lstm_missing_after_training": still_missing},
            )
        except Exception as exc:
            logger.exception(f"Model training failed: {exc}")
            await self.journal.log_event("MODEL_TRAINING_FAILED", "CRITICAL", str(exc), {})
            await self._set_status("ERROR", reason=f"Model training failed ({exc}).")
            raise
        finally:
            heartbeat_task.cancel()

    async def stop(self) -> None:
        if self._close_task:
            self._close_task.cancel()
        await self.journal.log_event("SYSTEM_STOP", "INFO", "Trading system stopped", {})
        await self.journal.stop()
        await self.client.close()
        await database.close()

    async def _close_request_loop(self) -> None:
        """Poll the manual-stop file frequently so Stop acts near-instantly."""
        interval = max(0.25, float(settings.close_poll_seconds))
        while True:
            try:
                await self._process_close_requests()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Close-request watcher error: {exc}")
            await asyncio.sleep(interval)

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
                        await self._evaluate_symbol(symbol, snapshot, meta)

                    # Manual stops are handled by the dedicated fast watcher
                    # (_close_request_loop); no need to poll them here too.
                    await self._write_equity_snapshot()
                    await self._handle_circuit_breaker()

                    now = datetime.now(UTC)
                    if now >= self._reconcile_at:
                        await self.execution.reconcile_orders()
                        self._reconcile_at = now + timedelta(minutes=5)
                    if self.learning.due():
                        await self.learning.run()
                    await asyncio.sleep(30)
                except Exception as exc:
                    logger.exception(f"Unexpected main loop error; sleeping then resuming: {exc}")
                    await self.journal.log_event("MAIN_LOOP_ERROR", "CRITICAL", str(exc), {})
                    await self._set_status("ERROR", reason=str(exc))
                    await asyncio.sleep(60)
        finally:
            await self.stop()

    async def _evaluate_symbol(self, symbol: str, snapshot: dict[str, Any], meta: Any) -> None:
        candles_by_tf = snapshot["candles"].get(symbol, {})
        analysis = self.ta.compute_all(candles_by_tf, meta=meta)
        if not analysis.get("timeframes"):
            return

        primary = self.strategy.primary_frame(analysis)
        latest_row = self.strategy.primary_latest(analysis)
        # Keep the monitor working from fresh volatility and momentum rather
        # than the snapshot frozen at entry time.
        self.execution.update_market_context(symbol, latest_row)

        if symbol in self.execution.open_trades:
            return

        bar_id = _bar_id(primary)
        if settings.entry_on_closed_candle and bar_id is not None:
            if self._last_bar.get(symbol) == bar_id:
                return
            self._last_bar[symbol] = bar_id

        lstm_signal = self.lstm.predict(symbol, primary)
        rl_features = self.strategy.build_rl_features(analysis)
        rl_decision = self.rl.decide(rl_features)

        analysis["regime"] = str(self.strategy.classify_regime(analysis, meta.fear_greed).value)
        gate = await self.gate.passes(
            lstm_signal,
            rl_decision,
            analysis,
            symbol,
            cooldown_reason=self.risk.cooldown_reason(symbol),
        )
        signal = self.strategy.build_trade_signal(symbol, analysis, lstm_signal, rl_decision, gate, meta.fear_greed)

        if not gate.approved:
            await self.journal.log_no_trade(signal, gate.failed_gate or "confidence gate failed")
            return

        balance = await self.client.get_usdt_balance()
        if balance <= 0:
            await self.journal.log_event("BALANCE_CHECK", "WARNING", "USDT balance unavailable or zero", {"symbol": symbol})
            return

        latest = signal.indicator_state.get("latest", {})
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
            conviction=gate.conviction,
        )
        plan = await self.risk.calculate(candidate)
        if not plan.approved:
            await self.journal.log_no_trade(signal, plan.reason)
            return

        result = await self.execution.place_trade(plan)
        if result.accepted:
            await self.learning.record_state()

    async def _handle_circuit_breaker(self) -> None:
        if not (self.risk.circuit_breaker_active and self.execution.open_trades):
            return
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
                "clears_at": f"drawdown back under {settings.circuit_breaker_reset_pct}%",
            },
        )

    async def _process_close_requests(self) -> None:
        path = settings.runtime_dir / "close_requests.json"
        if not path.exists():
            return
        try:
            symbols = json.loads(path.read_text(encoding="utf-8")).get("symbols", [])
        except Exception:
            return
        if not symbols:
            return
        for symbol in symbols:
            try:
                result = await self.execution.emergency_close_symbol(symbol)
                await self.journal.log_event(
                    "MANUAL_CLOSE",
                    "INFO",
                    f"Manual stop requested for {symbol}",
                    {"accepted": result.accepted, "reason": result.reason},
                )
            except Exception as exc:
                logger.warning(f"Manual close failed for {symbol}: {exc}")
        try:
            path.write_text(json.dumps({"symbols": []}), encoding="utf-8")
        except Exception:
            pass

    async def _get_current_price(self, symbol: str) -> Decimal:
        candles = await self.client.get_ohlcv(symbol, "1m", 1)
        return candles[-1].close if candles else Decimal("0")

    async def _write_equity_snapshot(self) -> None:
        try:
            balance = await self.client.get_usdt_balance()
            # Equity is free USDT plus the market value of what the bot's own
            # positions hold. Adding only unrealised PnL omitted the cost basis,
            # so opening a position dropped reported equity by its full notional
            # and manufactured a drawdown that latched the circuit breaker.
            holdings_value = Decimal("0")
            open_pnl = Decimal("0")
            for position in self.risk.open_positions.values():
                current_price = await self._get_current_price(position.symbol)
                if current_price <= 0:
                    continue
                if position.direction == "LONG":
                    holdings_value += position.quantity * current_price
                    open_pnl += position.quantity * (current_price - position.entry_price)
                else:
                    open_pnl += position.quantity * (position.entry_price - current_price)
            total_equity = Decimal(str(balance)) + holdings_value
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
        return os.getenv("FORCE_TRADING_PAUSED", "").strip().lower() not in {"1", "true", "yes"}

    async def _set_status(self, status: str, *, reason: str | None = None) -> None:
        path = settings.trading_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "trading_enabled": self.trading_enabled(),
            "status": status,
            "reason": reason,
            "testnet": settings.use_testnet,
            "market_type": settings.market_type,
            "binance_connected": bool(getattr(self.client, "connected", False)),
            "circuit_breaker_active": self.risk.circuit_breaker_active,
            "open_positions": len(self.risk.open_positions),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _lstm_current(weights_dir: Path, symbol: str) -> bool:
    weight = weights_dir / f"lstm_{symbol}.pt"
    meta = weights_dir / f"lstm_{symbol}.json"
    if not weight.exists() or not meta.exists():
        return False
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("model_version", 0)) == MODEL_VERSION
    except Exception:
        return False


def _rl_current(weights_dir: Path) -> bool:
    weight = weights_dir / "ppo_trading_agent.zip"
    meta = weights_dir / "ppo_trading_agent.json"
    if not weight.exists() or not meta.exists():
        return False
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("agent_version", 0)) == AGENT_VERSION
    except Exception:
        return False


def _bar_id(primary_frame: dict[str, Any]) -> int | None:
    rows = primary_frame.get("series_tail", [])
    if not rows:
        return None
    try:
        return int(rows[-1].get("open_time"))
    except (TypeError, ValueError):
        return None


async def migrate() -> None:
    await database.initialize()
    await database.run_migrations()
    await database.close()


async def download_data() -> None:
    client = ResilientBinanceClient()
    await client.initialize()
    end = datetime.now(UTC)
    start = end - timedelta(days=settings.training_days)
    interval = settings.training_interval
    source = "mainnet public klines" if settings.training_data_source != "venue" else "the trading venue"
    for i, symbol in enumerate(settings.symbols, start=1):
        logger.info(
            f"[{i}/{len(settings.symbols)}] Downloading {settings.training_days} days of {interval} "
            f"data for {symbol} from {source}"
        )
        try:
            candles = await client.get_historical_ohlcv(
                symbol, interval, int(start.timestamp() * 1000), int(end.timestamp() * 1000)
            )
            frame = pd.DataFrame([c.to_dict() for c in candles])
            path = settings.runtime_dir / f"training_{symbol}_{interval}.csv"
            frame.to_csv(path, index=False)
            span = ""
            if len(frame) and "open_time" in frame.columns:
                first = datetime.fromtimestamp(frame["open_time"].iloc[0] / 1000, tz=UTC)
                last = datetime.fromtimestamp(frame["open_time"].iloc[-1] / 1000, tz=UTC)
                span = f" ({first.date()} to {last.date()})"
            logger.info(f"Wrote {len(frame)} candles to {path}{span}")
            if len(frame) < 5_000:
                logger.warning(
                    f"{symbol}: only {len(frame)} candles retrieved. If TRAINING_DATA_SOURCE=venue and "
                    f"USE_TESTNET=true this is expected -- testnet retains very little history. Switch to "
                    f"TRAINING_DATA_SOURCE=mainnet for a real training set."
                )
        except Exception as exc:
            logger.warning(f"Historical data download failed for {symbol}, skipping: {exc}")
    await client.close()


async def train_models() -> None:
    from models.lstm_model import torch as _torch  # local import: reflects current module state

    if _torch is None:
        raise RuntimeError(
            "PyTorch is unavailable in this environment, so no LSTM model can be trained for any "
            "symbol. Check the build/runtime logs for a 'PyTorch failed to import' error."
        )

    lstm = LSTMModelService()
    interval = settings.training_interval
    rl_rows: list[dict[str, Any]] = []

    for i, symbol in enumerate(settings.symbols, start=1):
        path = settings.runtime_dir / f"training_{symbol}_{interval}.csv"
        if not path.exists():
            logger.warning(f"Training file missing for {symbol}: run --mode=download_data first")
            continue
        frame = None
        try:
            logger.info(f"[{i}/{len(settings.symbols)}] Training LSTM for {symbol}")
            _log_memory(f"before training {symbol}")
            raw = pd.read_csv(path)
            candles = [
                _row_to_candle(symbol, interval, row)
                for _, row in raw.iterrows()
                if float(row.get("volume", 0) or 0) > 0
            ]
            frame = enrich_indicators(candles_to_frame(candles)).replace([float("inf"), float("-inf")], pd.NA).ffill()
            metrics = lstm.train(symbol, frame)
            logger.info(f"LSTM metrics for {symbol}: {metrics}")
            _log_memory(f"after training {symbol}")
        except Exception as exc:
            logger.warning(f"LSTM training failed for {symbol}, skipping: {exc}")

        # Each symbol becomes its own episode segment. Concatenating them into
        # one tape (as before) meant the step from the last BTC bar to the first
        # ETH bar was scored as a price move of several thousand percent.
        if frame is not None:
            try:
                features = build_model_features(frame).ffill()
                features["close"] = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
                features[SEGMENT_KEY] = i
                rl_rows.extend(features.dropna().to_dict(orient="records"))
            except Exception as exc:
                logger.warning(f"Could not build RL rows for {symbol}: {exc}")

        raw = candles = frame = None
        gc.collect()

    if rl_rows:
        logger.info(f"Training RL agent on {len(rl_rows)} rows across {len(settings.symbols)} segments")
        try:
            RLAgentService().train(rl_rows)
        except Exception as exc:
            logger.warning(f"RL agent training failed, continuing without it: {exc}")
    else:
        logger.warning("No RL training rows were produced; the execution model will stay neutral.")


def _log_memory(label: str) -> None:
    try:
        import resource

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info(f"[memory] {label}: {rss_mb:.0f} MB RSS (peak)")
    except Exception:
        pass


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
