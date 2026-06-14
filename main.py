from __future__ import annotations

import argparse
import asyncio
import json
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
        await database.initialize()
        await database.run_migrations()
        await self.journal.start()
        await self.client.initialize()
        self.client.start_health_check()
        await self.journal.log_event("SYSTEM_START", "INFO", "Trading system started", {"testnet": settings.use_testnet})

    async def stop(self) -> None:
        await self.journal.log_event("SYSTEM_STOP", "INFO", "Trading system stopped", {})
        await self.journal.stop()
        await self.client.close()
        await database.close()

    async def main_loop(self) -> None:
        await self.start()
        try:
            while True:
                try:
                    await self._process_close_requests()
                    if not self.trading_enabled():
                        await self._set_status("PAUSED")
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

    async def _process_close_requests(self) -> None:
        state_path = settings.trading_state_path
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            close_requests = [symbol for symbol in state.get("close_requests", []) if symbol]
            for symbol in close_requests:
                logger.info(f"Manual close request received for {symbol}")
                await self.execution.emergency_close_symbol(symbol)
                await self.journal.log_event("MANUAL_CLOSE", "INFO", f"Manual close triggered for {symbol}", {})
            if close_requests:
                state["close_requests"] = []
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Close request processing failed: {exc}")

    def trading_enabled(self) -> bool:
        path = settings.trading_state_path
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            return bool(state.get("trading_enabled", False))
        except Exception:
            return False

    async def _set_status(self, status: str) -> None:
        path = settings.trading_state_path
        path.parent.mkdir(exist_ok=True)
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            state = {}
        state.update({"trading_enabled": bool(state.get("trading_enabled", False)), "status": status, "updated_at": datetime.now(UTC).isoformat()})
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
    for symbol in settings.symbols:
        logger.info(f"Downloading two years of 1h data for {symbol}")
        candles = await client.get_historical_ohlcv(symbol, "1h", int(start.timestamp() * 1000), int(end.timestamp() * 1000))
        frame = pd.DataFrame([c.to_dict() for c in candles])
        path = settings.runtime_dir / f"training_{symbol}_1h.csv"
        frame.to_csv(path, index=False)
        logger.info(f"Wrote {len(frame)} candles to {path}")
    await client.close()


async def train_models() -> None:
    ta = TechnicalAnalysisEngine()
    lstm = LSTMModelService()
    rl_rows: list[dict[str, float]] = []
    for symbol in settings.symbols:
        path = settings.runtime_dir / f"training_{symbol}_1h.csv"
        if not path.exists():
            logger.warning(f"Training file missing for {symbol}: run --mode=download_data first")
            continue
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
    if rl_rows:
        RLAgentService().train(rl_rows)


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
