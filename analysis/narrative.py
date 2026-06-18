"""Generates a plain-English explanation of why the bot did or didn't trade.

This exists purely for transparency on the dashboard: it turns the gate's
machine-oriented reason codes (e.g. "LSTM confidence 0.31 below 0.70") into a
short narrative a non-technical reader can follow, including what specific
risk the bot is avoiding by not trading right now.
"""

from __future__ import annotations

from analysis.regime_classifier import MarketRegime

REGIME_RISK_NOTES: dict[str, str] = {
    MarketRegime.EXTREME_FEAR.value: (
        "Market sentiment is in extreme fear. Prices in this regime can keep falling "
        "sharply on low conviction, and bounces are often sold into quickly -- entries "
        "here have a higher chance of being caught in a continued drop or a fakeout "
        "rally that reverses."
    ),
    MarketRegime.EXTREME_GREED.value: (
        "Market sentiment is in extreme greed. Moves in this regime are prone to sharp, "
        "fast reversals once momentum buyers are exhausted -- chasing a long here risks "
        "buying right before a pullback."
    ),
    MarketRegime.HIGH_VOLATILITY.value: (
        "Volatility is unusually high relative to its recent range. Stop losses are more "
        "likely to be hit by noise rather than a genuine trend change, and position sizing "
        "math becomes less reliable."
    ),
    MarketRegime.TRENDING_UP.value: (
        "The market is in an established uptrend. The main risk here is entering on a "
        "pullback that turns into a deeper reversal rather than a continuation."
    ),
    MarketRegime.TRENDING_DOWN.value: (
        "The market is in an established downtrend. The main risk here is a short squeeze "
        "or relief rally against the trend."
    ),
    MarketRegime.RANGING_TIGHT.value: (
        "The market is range-bound with low volatility. Breakout attempts from tight ranges "
        "frequently fail and snap back, so entries need strong confirmation."
    ),
    MarketRegime.MIXED.value: (
        "Signals across timeframes are mixed with no clear dominant regime, which makes "
        "the odds of any single direction working out less favorable."
    ),
}

# Maps substrings of ConfidenceGate's reason strings to a plain-English
# explanation of what that specific check protects against.
_REASON_EXPLANATIONS: list[tuple[str, str]] = [
    ("LSTM output is NO_TRADE", "the price-direction model itself isn't predicting a clear long or short move"),
    ("LSTM confidence", "the price-direction model's confidence in any direction is too low to trust"),
    ("RL action", "the reinforcement-learning execution model disagrees with the direction the price model suggests"),
    ("multi-timeframe confluence below", "not enough agreement across timeframes (1m-1d) to support a trade"),
    ("multi-timeframe direction contradicts", "shorter and longer timeframes are pointing in different directions"),
    ("only", "too few independent technical indicators (EMA, MACD, RSI, ADX, VWAP, volume flow, candle patterns) agree on direction"),
    ("major news window", "a high-impact economic news release is due within 30 minutes, which can cause unpredictable price spikes"),
    ("ATR is above", "current price swings (ATR) are abnormally large versus their recent average, making stop placement unreliable"),
]


def explain_reason(reason: str) -> str:
    for needle, explanation in _REASON_EXPLANATIONS:
        if needle in reason:
            return explanation
    return reason


def build_no_trade_narrative(
    symbol: str,
    regime: str,
    lstm_confidence: float,
    confluence_score: float,
    reasons: list[str],
) -> str:
    """Builds a short, readable paragraph explaining a NO_TRADE decision.

    Combines the regime's general risk profile with the specific gate
    checks that failed, so the dashboard can show *why* the bot is sitting
    out rather than just *that* it is.
    """
    if not reasons:
        return f"{symbol}: no trade taken, but no specific blocking reason was recorded."

    regime_note = REGIME_RISK_NOTES.get(regime, "")
    explained = [explain_reason(r) for r in reasons]
    # De-duplicate while preserving order, since several raw reasons can map
    # to the same plain-English explanation.
    seen: set[str] = set()
    unique_explained = []
    for item in explained:
        if item not in seen:
            seen.add(item)
            unique_explained.append(item)

    if len(unique_explained) == 1:
        reasons_sentence = f"The system held back because {unique_explained[0]}."
    else:
        *head, tail = unique_explained
        reasons_sentence = "The system held back because " + "; ".join(head) + f"; and {tail}."

    parts = [
        f"{symbol} ({regime.replace('_', ' ').title()}, LSTM confidence {lstm_confidence:.0%}, "
        f"confluence {confluence_score:.0f}/100): {reasons_sentence}"
    ]
    if regime_note:
        parts.append(regime_note)
    parts.append(
        "No position was opened or risked. The bot will re-evaluate on the next cycle and "
        "trade automatically once the confidence, confluence, and risk checks line up."
    )
    return " ".join(parts)
