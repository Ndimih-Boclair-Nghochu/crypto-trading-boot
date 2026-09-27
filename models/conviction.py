from __future__ import annotations

"""How sure the system is about a setup, on a 0-1 scale.

The gate answers a yes/no question: may this trade happen at all. Conviction
answers a different one: given that it may, how much of the account should back
it. Every trade used to be sized identically regardless of whether the evidence
was marginal or overwhelming, which is the same as having no opinion.

The components are kept separate and returned alongside the score so the
dashboard can show *why* a position was sized the way it was, rather than
presenting a single opaque number.
"""

from dataclasses import dataclass, field
from typing import Any

from models.lstm_model import LSTMSignal
from models.rl_agent import RLDecision


# Sum to 1.0. Directional evidence from the price model carries the most weight;
# the RL agent is an execution-timing opinion and is deliberately minor.
WEIGHTS: dict[str, float] = {
    "probability": 0.24,
    "margin": 0.20,
    "confluence": 0.20,
    "agreement": 0.14,
    "model_quality": 0.12,
    "rl": 0.10,
}

REGIME_MULTIPLIER: dict[tuple[str, str], float] = {
    ("TRENDING_UP", "LONG"): 1.00,
    ("TRENDING_UP", "SHORT"): 0.75,
    ("TRENDING_DOWN", "SHORT"): 1.00,
    ("TRENDING_DOWN", "LONG"): 0.75,
    ("RANGING_TIGHT", "LONG"): 0.85,
    ("RANGING_TIGHT", "SHORT"): 0.85,
    ("HIGH_VOLATILITY", "LONG"): 0.80,
    ("HIGH_VOLATILITY", "SHORT"): 0.80,
    ("EXTREME_FEAR", "LONG"): 0.90,
    ("EXTREME_FEAR", "SHORT"): 0.70,
    ("EXTREME_GREED", "SHORT"): 0.90,
    ("EXTREME_GREED", "LONG"): 0.70,
    ("MIXED", "LONG"): 0.80,
    ("MIXED", "SHORT"): 0.80,
}


@dataclass(frozen=True)
class ConvictionScore:
    value: float
    components: dict[str, float] = field(default_factory=dict)
    regime_multiplier: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 4),
            "regime_multiplier": round(self.regime_multiplier, 3),
            "components": {key: round(item, 4) for key, item in self.components.items()},
        }


def _norm(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def score_conviction(
    lstm_signal: LSTMSignal,
    rl_decision: RLDecision,
    *,
    direction: str,
    regime: str,
    confluence_score: float,
    indicator_agreement: int,
    confidence_threshold: float,
    min_margin: float,
    min_confluence: float,
    min_agreement: int,
) -> ConvictionScore:
    expected_action = "BUY" if direction == "LONG" else "SELL"

    if rl_decision.action == expected_action:
        rl_component = _norm(rl_decision.confidence, 0.35, 0.90)
    elif rl_decision.action in {"HOLD", "CLOSE_LONG", "CLOSE_SHORT"}:
        # Neutral, not hostile. An undertrained or cautious agent should damp
        # size, not veto a setup the price model and the tape both agree on.
        rl_component = 0.30
    else:
        rl_component = 0.0

    components = {
        # Headroom above the pass mark, not the raw value: a signal that only
        # just cleared the threshold should not size like one that sailed past.
        "probability": _norm(lstm_signal.confidence, confidence_threshold, 0.85),
        "margin": _norm(lstm_signal.margin, min_margin, 0.60),
        "confluence": _norm(confluence_score, min_confluence, 100.0),
        "agreement": _norm(indicator_agreement, min_agreement, 7),
        "model_quality": _norm(lstm_signal.model_quality, 0.40, 0.70),
        "rl": rl_component,
    }

    base = sum(WEIGHTS[key] * value for key, value in components.items())
    multiplier = REGIME_MULTIPLIER.get((str(regime), direction), 0.80)
    return ConvictionScore(max(0.0, min(1.0, base * multiplier)), components, multiplier)
