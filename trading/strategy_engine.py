from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analysis.features import MODEL_FEATURES, build_model_features
from analysis.regime_classifier import MarketRegime, classify_regime
from config import Settings, settings
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
    conviction: float = 0.0


class StrategyEngine:
    def __init__(self, cfg: Settings = settings) -> None:
        self.settings = cfg

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
        # Only take proven-profitable setups: if this regime's strategy has been
        # disabled (negative historical expectancy), force a NO_TRADE regardless
        # of the gate, so the bot concentrates on its winners.
        disabled = strategy.upper() in self.settings.disabled_strategies
        approved = gate.approved and not disabled
        reasons = list(gate.reasons)
        if disabled:
            reasons.append(f"Strategy {strategy} disabled (negative historical expectancy)")
        return TradeSignal(
            symbol=symbol,
            direction=gate.direction if approved else "NO_TRADE",
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
                "lstm_margin": round(lstm_signal.margin, 4),
                "model_quality": round(lstm_signal.model_quality, 4),
                "rl_action": rl_decision.action,
                "rl_probabilities": rl_decision.probabilities or {},
                "conviction": gate.conviction_detail,
                "secondary_check": secondary_check,
                "raw_candles": primary_frame.get("series_tail", [])[-120:],
            },
            reasons=reasons,
            conviction=gate.conviction,
        )

    def build_rl_features(self, analysis_payload: dict[str, Any]) -> dict[str, float]:
        """Latest stationary feature row, keyed exactly as the RL agent expects."""
        rows = self.primary_frame(analysis_payload).get("series_tail", [])
        if not rows:
            return {name: 0.0 for name in MODEL_FEATURES}
        frame = pd.DataFrame(rows)
        try:
            features = build_model_features(frame).ffill().fillna(0.0)
        except Exception:
            return {name: 0.0 for name in MODEL_FEATURES}
        if features.empty:
            return {name: 0.0 for name in MODEL_FEATURES}
        row = features.iloc[-1]
        return {name: float(row.get(name, 0.0) or 0.0) for name in MODEL_FEATURES}

    def primary_latest(self, analysis_payload: dict[str, Any]) -> dict[str, Any]:
        frames = analysis_payload.get("timeframes", {})
        for preferred in self.settings.timeframe_preference:
            if preferred in frames:
                return frames[preferred].get("latest", {})
        if frames:
            return next(iter(frames.values())).get("latest", {})
        return analysis_payload.get("latest", {})

    def primary_frame(self, analysis_payload: dict[str, Any]) -> dict[str, Any]:
        frames = analysis_payload.get("timeframes", {})
        for preferred in self.settings.timeframe_preference:
            if preferred in frames:
                return frames[preferred]
        if frames:
            return next(iter(frames.values()))
        return {}
