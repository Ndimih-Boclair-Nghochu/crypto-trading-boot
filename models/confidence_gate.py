from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from config import Settings, settings
from models.conviction import ConvictionScore, score_conviction
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
    conviction: float = 0.0
    conviction_detail: dict[str, Any] = field(default_factory=dict)


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
        cooldown_reason: str | None = None,
    ) -> GateResult:
        reasons: list[str] = []
        direction = lstm_signal.direction
        confluence = indicators.get("confluence", {})
        primary = self._primary_latest(indicators)
        regime = str(indicators.get("regime", "MIXED"))
        extreme_regime = "EXTREME" in regime

        overrides = self._overrides()
        base_threshold = float(overrides.get("confidence_threshold", self.settings.confidence_threshold))
        threshold = base_threshold * 0.85 if extreme_regime else base_threshold
        min_margin = float(self.settings.lstm_min_margin)
        min_confluence = float(self.settings.min_confluence_score)
        min_agreement = int(self.settings.min_indicator_agreement)

        confluence_value = float(confluence.get("score", 0) or 0)
        agreement = self._indicator_agreement(primary, direction)

        conviction = score_conviction(
            lstm_signal,
            rl_decision,
            direction=direction if direction in {"LONG", "SHORT"} else "LONG",
            regime=regime,
            confluence_score=confluence_value,
            indicator_agreement=agreement,
            confidence_threshold=threshold,
            min_margin=min_margin,
            min_confluence=min_confluence,
            min_agreement=min_agreement,
        )

        if direction not in {"LONG", "SHORT"}:
            reasons.append(
                f"LSTM did not produce a tradeable signal ({lstm_signal.reason})"
                if lstm_signal.reason
                else "LSTM output is NO_TRADE"
            )
            return self._blocked(reasons, agreement, conviction)

        # Spot accounts cannot sell an asset they do not hold. Letting a SHORT
        # through here produced an entry order that Binance rejected with -2010
        # on every cycle -- 428 rejections in 48 hours, and four of five symbols
        # permanently stuck in that loop because their signal was almost always
        # SHORT.
        if direction == "SHORT" and not self.settings.shorting_available:
            reasons.append(
                "SHORT signals cannot be executed on a spot account "
                "(set MARKET_TYPE=futures to enable, or the symbol stays long-only)"
            )

        if cooldown_reason:
            reasons.append(cooldown_reason)

        if lstm_signal.confidence < threshold:
            suffix = f" ({lstm_signal.reason})" if lstm_signal.reason and lstm_signal.confidence == 0.0 else ""
            reasons.append(f"LSTM confidence {lstm_signal.confidence:.2f} below {threshold:.2f}{suffix}")

        # A model can be 40% LONG / 38% SHORT and still clear a raw probability
        # bar while holding no directional opinion at all. Margin is what
        # separates an edge from a coin flip.
        if lstm_signal.margin < min_margin:
            reasons.append(
                f"directional margin {lstm_signal.margin:.2f} below {min_margin:.2f} "
                f"(LONG {lstm_signal.probabilities.get('LONG', 0):.2f} vs "
                f"SHORT {lstm_signal.probabilities.get('SHORT', 0):.2f})"
            )

        min_quality = float(self.settings.min_model_quality)
        if lstm_signal.model_quality < min_quality:
            reasons.append(
                f"model for {symbol} has out-of-sample directional precision "
                f"{lstm_signal.model_quality:.2f}, below the {min_quality:.2f} minimum"
            )

        # RL vetoes only an outright contradiction. Requiring exact equality
        # meant a HOLD -- the most common output of any well-trained agent --
        # blocked every setup, which is how the system spent most of its life
        # refusing to trade.
        expected_action = "BUY" if direction == "LONG" else "SELL"
        opposite_action = "SELL" if direction == "LONG" else "BUY"
        if rl_decision.action == opposite_action and rl_decision.confidence >= float(self.settings.rl_veto_confidence):
            reasons.append(
                f"RL execution model calls {rl_decision.action} against a {direction} setup "
                f"with {rl_decision.confidence:.2f} confidence"
            )

        if confluence_value < min_confluence:
            reasons.append(f"multi-timeframe confluence {confluence_value:.0f} below {min_confluence:.0f}")
        if confluence.get("direction") not in {direction, None, "NEUTRAL"}:
            reasons.append("multi-timeframe direction contradicts LSTM")

        if agreement < min_agreement:
            reasons.append(f"only {agreement} independent indicators agree (need {min_agreement})")

        exceptional_confidence = (
            lstm_signal.confidence >= 0.80
            and rl_decision.action == expected_action
            and rl_decision.confidence >= 0.70
            and confluence_value >= 90
        )
        if await self._major_news_window(symbol, allow_degraded=exceptional_confidence):
            reasons.append("major news window within 30 minutes")

        if bool(primary.get("atr_spike")):
            reasons.append("ATR is above 3x its 20-period average")

        if float(self.settings.min_conviction) > 0 and conviction.value < float(self.settings.min_conviction):
            reasons.append(f"conviction {conviction.value:.2f} below {float(self.settings.min_conviction):.2f}")

        if reasons:
            return self._blocked(reasons, agreement, conviction)
        return GateResult(True, direction, None, [], agreement, conviction.value, conviction.as_dict())

    def _blocked(self, reasons: list[str], agreement: int, conviction: ConvictionScore) -> GateResult:
        return GateResult(False, "NO_TRADE", reasons[0], reasons, agreement, conviction.value, conviction.as_dict())

    def _primary_latest(self, indicators: dict[str, Any]) -> dict[str, Any]:
        frames = indicators.get("timeframes", {})
        for preferred in self.settings.timeframe_preference:
            if preferred in frames:
                return frames[preferred].get("latest", {})
        if frames:
            return next(iter(frames.values())).get("latest", {})
        return indicators.get("latest", {})

    def _overrides(self) -> dict[str, Any]:
        override_path = self.settings.runtime_dir / "risk_overrides.json"
        if not override_path.exists():
            return {}
        try:
            return json.loads(override_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Risk override read skipped: {exc}")
            return {}

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
