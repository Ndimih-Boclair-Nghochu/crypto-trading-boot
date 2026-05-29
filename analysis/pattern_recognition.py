from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternResult:
    candlestick: dict[str, bool]
    chart: dict[str, bool]
    support_levels: list[float]
    resistance_levels: list[float]


def identify_candlestick_patterns(df: pd.DataFrame) -> dict[str, bool]:
    if len(df) < 3:
        return {
            "doji": False,
            "hammer": False,
            "shooting_star": False,
            "bullish_engulfing": False,
            "bearish_engulfing": False,
            "morning_star": False,
            "evening_star": False,
            "bullish_harami": False,
            "bearish_harami": False,
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]
    first = df.iloc[-3]

    body = abs(last.close - last.open)
    candle_range = max(last.high - last.low, 1e-12)
    upper_shadow = last.high - max(last.open, last.close)
    lower_shadow = min(last.open, last.close) - last.low

    doji = body <= candle_range * 0.1
    hammer = lower_shadow >= body * 2 and upper_shadow <= max(body, candle_range * 0.15) and last.close > last.open
    shooting_star = upper_shadow >= body * 2 and lower_shadow <= max(body, candle_range * 0.15) and last.close < last.open

    prev_bear = prev.close < prev.open
    prev_bull = prev.close > prev.open
    last_bull = last.close > last.open
    last_bear = last.close < last.open

    bullish_engulfing = prev_bear and last_bull and last.open <= prev.close and last.close >= prev.open
    bearish_engulfing = prev_bull and last_bear and last.open >= prev.close and last.close <= prev.open

    morning_star = (
        first.close < first.open
        and abs(prev.close - prev.open) < abs(first.close - first.open) * 0.5
        and last.close > last.open
        and last.close > (first.open + first.close) / 2
    )
    evening_star = (
        first.close > first.open
        and abs(prev.close - prev.open) < abs(first.close - first.open) * 0.5
        and last.close < last.open
        and last.close < (first.open + first.close) / 2
    )

    bullish_harami = prev_bear and last_bull and last.open > prev.close and last.close < prev.open
    bearish_harami = prev_bull and last_bear and last.open < prev.close and last.close > prev.open

    return {
        "doji": bool(doji),
        "hammer": bool(hammer),
        "shooting_star": bool(shooting_star),
        "bullish_engulfing": bool(bullish_engulfing),
        "bearish_engulfing": bool(bearish_engulfing),
        "morning_star": bool(morning_star),
        "evening_star": bool(evening_star),
        "bullish_harami": bool(bullish_harami),
        "bearish_harami": bool(bearish_harami),
    }


def swing_levels(df: pd.DataFrame, lookback: int = 50, window: int = 2) -> tuple[list[float], list[float]]:
    recent = df.tail(lookback)
    supports: list[float] = []
    resistances: list[float] = []
    highs = recent["high"].to_numpy()
    lows = recent["low"].to_numpy()
    for i in range(window, len(recent) - window):
        if lows[i] == min(lows[i - window : i + window + 1]):
            supports.append(float(lows[i]))
        if highs[i] == max(highs[i - window : i + window + 1]):
            resistances.append(float(highs[i]))
    return _dedupe_levels(supports), _dedupe_levels(resistances)


def _dedupe_levels(levels: list[float], tolerance_pct: float = 0.004) -> list[float]:
    out: list[float] = []
    for level in sorted(levels):
        if not out or abs(level - out[-1]) / max(out[-1], 1e-12) > tolerance_pct:
            out.append(level)
        else:
            out[-1] = (out[-1] + level) / 2
    return out[-8:]


def identify_chart_patterns(df: pd.DataFrame) -> dict[str, bool]:
    if len(df) < 30:
        return {
            "double_top": False,
            "double_bottom": False,
            "head_shoulders": False,
            "inverse_head_shoulders": False,
            "triangle": False,
            "bull_flag": False,
            "bear_flag": False,
        }

    supports, resistances = swing_levels(df, lookback=min(80, len(df)))
    close = df["close"]
    recent = df.tail(20)
    price = float(close.iloc[-1])

    double_top = len(resistances) >= 2 and abs(resistances[-1] - resistances[-2]) / price < 0.015 and price < resistances[-1]
    double_bottom = len(supports) >= 2 and abs(supports[-1] - supports[-2]) / price < 0.015 and price > supports[-1]

    swing_highs = _local_extrema(df["high"].to_numpy(), mode="high")[-5:]
    swing_lows = _local_extrema(df["low"].to_numpy(), mode="low")[-5:]
    head_shoulders = False
    inverse_head_shoulders = False
    if len(swing_highs) >= 3:
        a, b, c = [df["high"].iloc[i] for i in swing_highs[-3:]]
        head_shoulders = b > a and b > c and abs(a - c) / max(b, 1e-12) < 0.04
    if len(swing_lows) >= 3:
        a, b, c = [df["low"].iloc[i] for i in swing_lows[-3:]]
        inverse_head_shoulders = b < a and b < c and abs(a - c) / max(abs(b), 1e-12) < 0.04

    high_slope = np.polyfit(range(len(recent)), recent["high"], 1)[0]
    low_slope = np.polyfit(range(len(recent)), recent["low"], 1)[0]
    triangle = high_slope < 0 and low_slope > 0

    impulse = close.pct_change(10).iloc[-11]
    consolidation_width = (recent["high"].max() - recent["low"].min()) / max(price, 1e-12)
    bull_flag = impulse > 0.04 and consolidation_width < 0.035 and close.iloc[-1] > close.iloc[-5]
    bear_flag = impulse < -0.04 and consolidation_width < 0.035 and close.iloc[-1] < close.iloc[-5]

    return {
        "double_top": bool(double_top),
        "double_bottom": bool(double_bottom),
        "head_shoulders": bool(head_shoulders),
        "inverse_head_shoulders": bool(inverse_head_shoulders),
        "triangle": bool(triangle),
        "bull_flag": bool(bull_flag),
        "bear_flag": bool(bear_flag),
    }


def _local_extrema(values: np.ndarray, mode: str, window: int = 2) -> list[int]:
    indexes: list[int] = []
    for i in range(window, len(values) - window):
        chunk = values[i - window : i + window + 1]
        if mode == "high" and values[i] == np.max(chunk):
            indexes.append(i)
        if mode == "low" and values[i] == np.min(chunk):
            indexes.append(i)
    return indexes


def analyze_patterns(df: pd.DataFrame) -> PatternResult:
    supports, resistances = swing_levels(df)
    return PatternResult(
        candlestick=identify_candlestick_patterns(df),
        chart=identify_chart_patterns(df),
        support_levels=supports,
        resistance_levels=resistances,
    )


def pattern_bias(patterns: PatternResult) -> dict[str, Any]:
    bullish = sum(
        1
        for key in ("hammer", "bullish_engulfing", "morning_star", "bullish_harami")
        if patterns.candlestick.get(key)
    ) + sum(1 for key in ("double_bottom", "inverse_head_shoulders", "bull_flag") if patterns.chart.get(key))
    bearish = sum(
        1
        for key in ("shooting_star", "bearish_engulfing", "evening_star", "bearish_harami")
        if patterns.candlestick.get(key)
    ) + sum(1 for key in ("double_top", "head_shoulders", "bear_flag") if patterns.chart.get(key))
    if bullish > bearish:
        direction = "LONG"
    elif bearish > bullish:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    return {"direction": direction, "bullish_count": bullish, "bearish_count": bearish}
