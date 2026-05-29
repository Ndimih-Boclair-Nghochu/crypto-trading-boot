from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analysis.regime_classifier import MarketRegime, classify_regime
from models.confidence_gate import GateResult
from models.lstm_model import LSTMSignal
from models.rl_agent import RLDecision


REGIME_STRATEGY_MAP = {
    MarketRegime.TRENDING_UP: ("EMA_TREND_PULLBACK_LONG", "ADX>30 and RSI 40-60 pullback zone"),
    MarketRegime.TRENDING_DOWN: ("SHORT_MOMENTUM_TREND", "ADX>30 with downside momentum confirmation"),
    MarketRegime.RANGING_TIGHT: ("BB_RSI_MEAN_REVERSION", "support/resistance hold"),
    MarketRegime.HIGH_VOLATILITY: ("VOLUME_CONFIRMED_BREAKOUT", "position size reduced by risk engine"),
    MarketRegime.EXTREME_FEAR: ("DCA_ACCUMULATION_SIGNAL", "BTC dominance rising"),
    MarketRegime.EXTREME_GREED: ("EXPOSURE_REDUCTION_DIVERGENCE_WATCH", "tightened stops"),
    MarketRegime.MIXED: ("WAIT_FOR_CONFLUENCE", "no primary edge"),
}


@dataclass(frozen=True)
class TradeSignal:
    symbol: str
    direction: str
    strategy_used: str
    regime_at_entry: str
    lstm_confidence: float
    rl_confidence: float
    confluence_score: float
    indicator_agreement: int
    indicator_state: dict[str, Any]
    reasons: list[str] = field(default_factory=list)


class StrategyEngine:
    def classify_regime(self, analysis_payload: dict[str, Any], fear_greed: float | None = None) -> MarketRegime:
        latest = self.primary_latest(analysis_payload)
        return classify_regime(latest, fear_greed=fear_greed)

    def select_strategy(self, regime: MarketRegime) -> tuple[str, str]:
        return REGIME_STRATEGY_MAP.get(regime, REGIME_STRATEGY_MAP[MarketRegime.MIXED])

    def build_trade_signal(
        self,
        symbol: str,
        analysis_payload: dict[str, Any],
        lstm_signal: LSTMSignal,
        rl_decision: RLDecision,
        gate: GateResult,
        fear_greed: float | None = None,
    ) -> TradeSignal:
        latest = self.primary_latest(analysis_payload)
        primary_frame = self.primary_frame(analysis_payload)
        regime = self.classify_regime(analysis_payload, fear_greed=fear_greed)
        strategy, secondary_check = self.select_strategy(regime)
        confluence_score = float(analysis_payload.get("confluence", {}).get("score", 0) or 0)
        return TradeSignal(
            symbol=symbol,
            direction=gate.direction if gate.approved else "NO_TRADE",
            strategy_used=strategy,
            regime_at_entry=str(regime.value),
            lstm_confidence=lstm_signal.confidence,
            rl_confidence=rl_decision.confidence,
            confluence_score=confluence_score,
            indicator_agreement=gate.indicator_agreement,
            indicator_state={
                "latest": latest,
                "confluence": analysis_payload.get("confluence", {}),
                "lstm_probabilities": lstm_signal.probabilities,
                "rl_action": rl_decision.action,
                "secondary_check": secondary_check,
                "raw_candles": primary_frame.get("series_tail", [])[-60:],
            },
            reasons=gate.reasons,
        )

    def build_rl_state(
        self,
        analysis_payload: dict[str, Any],
        lstm_signal: LSTMSignal,
        open_pnl: float = 0.0,
        drawdown: float = 0.0,
        position: float = 0.0,
        entry_price: float = 0.0,
    ) -> dict[str, Any]:
        latest = self.primary_latest(analysis_payload)
        state = dict(latest)
        state["lstm_confidence"] = lstm_signal.confidence
        state["confluence_score"] = float(analysis_payload.get("confluence", {}).get("score", 0) or 0)
        state["open_pnl"] = open_pnl
        state["drawdown"] = drawdown
        state["position"] = position
        state["entry_price"] = entry_price
        return state

    def primary_latest(self, analysis_payload: dict[str, Any]) -> dict[str, Any]:
        frames = analysis_payload.get("timeframes", {})
        for preferred in ("1h", "15m", "4h", "5m", "1m", "1d"):
            if preferred in frames:
                return frames[preferred].get("latest", {})
        if frames:
            return next(iter(frames.values())).get("latest", {})
        return analysis_payload.get("latest", {})

    def primary_frame(self, analysis_payload: dict[str, Any]) -> dict[str, Any]:
        frames = analysis_payload.get("timeframes", {})
        for preferred in ("1h", "15m", "4h", "5m", "1m", "1d"):
            if preferred in frames:
                return frames[preferred]
        if frames:
            return next(iter(frames.values()))
        return {}
