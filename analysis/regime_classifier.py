from __future__ import annotations

from enum import StrEnum
from typing import Any


class MarketRegime(StrEnum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING_TIGHT = "RANGING_TIGHT"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    EXTREME_FEAR = "EXTREME_FEAR"
    EXTREME_GREED = "EXTREME_GREED"
    MIXED = "MIXED"


def classify_regime(indicators: dict[str, Any], fear_greed: float | None = None) -> MarketRegime:
    latest = indicators.get("latest", indicators)
    adx = float(latest.get("adx", 0) or 0)
    ema21 = float(latest.get("ema_21", 0) or 0)
    ema50 = float(latest.get("ema_50", 0) or 0)
    bb_width = float(latest.get("bb_width", 0) or 0)
    bb_width_percentile = float(latest.get("bb_width_percentile", 50) or 50)
    volatility_percentile = float(latest.get("volatility_percentile", 50) or 50)
    fg = fear_greed if fear_greed is not None else latest.get("fear_greed")
    fg_value = float(fg) if fg is not None else None

    if fg_value is not None and fg_value < 20:
        return MarketRegime.EXTREME_FEAR
    if fg_value is not None and fg_value > 80:
        return MarketRegime.EXTREME_GREED
    if volatility_percentile > 80:
        return MarketRegime.HIGH_VOLATILITY
    if adx > 25 and ema21 > ema50:
        return MarketRegime.TRENDING_UP
    if adx > 25 and ema21 < ema50:
        return MarketRegime.TRENDING_DOWN
    if adx < 20 and bb_width_percentile < 30 and bb_width > 0:
        return MarketRegime.RANGING_TIGHT
    return MarketRegime.MIXED
