from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from config import Settings, settings
from models.lstm_model import LSTMSignal
from models.rl_agent import RLDecision
from utils.logger import logger

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None


@dataclass(frozen=True)
class GateResult:
    approved: bool
    direction: str = "NO_TRADE"
    failed_gate: str | None = None
    reasons: list[str] = field(default_factory=list)
    indicator_agreement: int = 0


class ConfidenceGate:
    def __init__(self, cfg: Settings = settings) -> None:
        self.settings = cfg
        self._events_cache: list[dict[str, Any]] = []
        self._events_loaded_at: datetime | None = None

    async def passes(
        self,
        lstm_signal: LSTMSignal,
        rl_decision: RLDecision,
        indicators: dict[str, Any],
        symbol: str,
    ) -> GateResult:
        reasons: list[str] = []
        direction = lstm_signal.direction
        confluence = indicators.get("confluence", {})
        primary = self._primary_latest(indicators)
        extreme_regime = "EXTREME" in str(indicators.get("regime", ""))
        base_threshold = self._confidence_threshold()
        threshold = base_threshold * 0.85 if extreme_regime else base_threshold

        if direction not in {"LONG", "SHORT"}:
            if lstm_signal.reason:
                reasons.append(f"LSTM did not produce a tradeable signal ({lstm_signal.reason})")
            else:
                reasons.append("LSTM output is NO_TRADE")
        if lstm_signal.confidence < threshold:
            suffix = f" ({lstm_signal.reason})" if lstm_signal.reason and lstm_signal.confidence == 0.0 else ""
            reasons.append(f"LSTM confidence {lstm_signal.confidence:.2f} below {threshold:.2f}{suffix}")

        expected_action = "BUY" if direction == "LONG" else "SELL"
        if rl_decision.action != expected_action:
            reasons.append(f"RL action {rl_decision.action} does not match {expected_action}")

        if float(confluence.get("score", 0) or 0) < 65:
            reasons.append("multi-timeframe confluence below 65")
        if confluence.get("direction") not in {direction, None}:
            reasons.append("multi-timeframe direction contradicts LSTM")

        agreement = self._indicator_agreement(primary, direction)
        if agreement < 4:
            reasons.append(f"only {agreement} independent indicators agree")

        exceptional_confidence = (
            lstm_signal.confidence >= 0.90
            and rl_decision.confidence >= 0.90
            and float(confluence.get("score", 0) or 0) >= 90
            and rl_decision.action == expected_action
        )
        if await self._major_news_window(symbol, allow_degraded=exceptional_confidence):
            reasons.append("major news window within 30 minutes")

        if bool(primary.get("atr_spike")):
            reasons.append("ATR is above 3x its 20-period average")

        if reasons:
            return GateResult(False, "NO_TRADE", reasons[0], reasons, agreement)
        return GateResult(True, direction, None, [], agreement)

    def _primary_latest(self, indicators: dict[str, Any]) -> dict[str, Any]:
        frames = indicators.get("timeframes", {})
        for preferred in ("1h", "15m", "4h", "5m", "1m", "1d"):
            if preferred in frames:
                return frames[preferred].get("latest", {})
        if frames:
            return next(iter(frames.values())).get("latest", {})
        return indicators.get("latest", {})

    def _confidence_threshold(self) -> float:
        override_path = self.settings.runtime_dir / "risk_overrides.json"
        if not override_path.exists():
            return float(self.settings.confidence_threshold)
        try:
            overrides = json.loads(override_path.read_text(encoding="utf-8"))
            return float(overrides.get("confidence_threshold", self.settings.confidence_threshold))
        except Exception as exc:
            logger.warning(f"Confidence threshold override skipped: {exc}")
            return float(self.settings.confidence_threshold)

    def _indicator_agreement(self, latest: dict[str, Any], direction: str) -> int:
        if direction not in {"LONG", "SHORT"}:
            return 0
        bullish = direction == "LONG"
        checks = [
            _gt(latest.get("ema_21"), latest.get("ema_50")) if bullish else _lt(latest.get("ema_21"), latest.get("ema_50")),
            _gt(latest.get("macd_hist"), 0) if bullish else _lt(latest.get("macd_hist"), 0),
            _gt(latest.get("rsi_14"), 50) if bullish else _lt(latest.get("rsi_14"), 50),
            _gt(latest.get("di_plus"), latest.get("di_minus")) if bullish else _lt(latest.get("di_plus"), latest.get("di_minus")),
            _gt(latest.get("close"), latest.get("vwap")) if bullish else _lt(latest.get("close"), latest.get("vwap")),
            _gt(latest.get("cmf_20"), 0) if bullish else _lt(latest.get("cmf_20"), 0),
            latest.get("patterns", {}).get("bias", {}).get("direction") == direction,
        ]
        return sum(bool(item) for item in checks)

    async def _major_news_window(self, symbol: str, allow_degraded: bool = False) -> bool:
        now = datetime.now(UTC)
        if not self.settings.economic_calendar_api_url:
            if self.settings.require_economic_calendar:
                logger.warning(
                    f"Economic calendar URL missing for {symbol} and REQUIRE_ECONOMIC_CALENDAR is "
                    f"set; {'allowing degraded high-confidence trade' if allow_degraded else 'blocking setup'}"
                )
                return not allow_degraded
            # No calendar feed configured and it isn't required: don't gate
            # on news at all rather than silently vetoing every trade.
            return False
        if self._events_loaded_at and now - self._events_loaded_at < timedelta(hours=1):
            events = self._events_cache
        else:
            events = await self._fetch_events()
            if events is None:
                if self.settings.require_economic_calendar:
                    logger.warning(
                        f"Economic calendar unavailable for {symbol} and REQUIRE_ECONOMIC_CALENDAR is "
                        f"set; {'allowing degraded high-confidence trade' if allow_degraded else 'blocking setup'}"
                    )
                    return not allow_degraded
                logger.warning(f"Economic calendar unavailable for {symbol}; proceeding without news gating")
                return False
            self._events_cache = events
            self._events_loaded_at = now
        for event in events:
            try:
                event_time = datetime.fromisoformat(str(event["time"]).replace("Z", "+00:00"))
                impact = str(event.get("impact", "")).upper()
                if impact in {"HIGH", "MAJOR"} and abs(event_time - now) <= timedelta(minutes=30):
                    return True
            except Exception:
                continue
        return False

    async def _fetch_events(self) -> list[dict[str, Any]] | None:
        try:
            if aiohttp:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(self.settings.economic_calendar_api_url) as response:
                        response.raise_for_status()
                        payload = await response.json()
            else:
                payload = await asyncio.to_thread(lambda: requests.get(self.settings.economic_calendar_api_url, timeout=10).json())
            if isinstance(payload, dict):
                return list(payload.get("events", []))
            if isinstance(payload, list):
                return payload
        except Exception as exc:
            logger.warning(f"Economic calendar fetch failed: {exc}")
        return None


def _gt(left: Any, right: Any) -> bool:
    return _num(left) > _num(right)


def _lt(left: Any, right: Any) -> bool:
    return _num(left) < _num(right)


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
