from __future__ import annotations

from decimal import Decimal

import pytest

from analysis.technical_analysis import TechnicalAnalysisEngine, candles_to_frame, enrich_indicators
from data.market_data import Candle


def make_candles(n: int = 240, slope: float = 0.4) -> list[Candle]:
    candles: list[Candle] = []
    for i in range(n):
        price = 100 + i * slope
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1h",
                open_time=i * 3_600_000,
                open=Decimal(str(price)),
                high=Decimal(str(price + 2)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price + 1)),
                volume=Decimal("100"),
                close_time=i * 3_600_000 + 3_599_999,
            )
        )
    return candles


def test_indicator_engine_computes_required_fields() -> None:
    engine = TechnicalAnalysisEngine()
    result = engine.compute_timeframe(make_candles(), fear_greed=55)
    latest = result["latest"]
    for key in ("ema_21", "ema_50", "macd_hist", "adx", "rsi_14", "atr_14", "bb_percent_b", "obv", "vwap", "cmf_20"):
        assert key in latest
    assert latest["fear_greed"] == 55
    assert latest["patterns"]["support_levels"] is not None


def test_multi_timeframe_confluence_requires_three_timeframes() -> None:
    engine = TechnicalAnalysisEngine()
    payload = engine.compute_all({"1h": make_candles(), "4h": make_candles(), "15m": make_candles()})
    assert payload["confluence"]["direction"] in {"LONG", "SHORT", "NEUTRAL"}
    assert 0 <= payload["confluence"]["score"] <= 100


def test_invalid_candles_are_rejected() -> None:
    bad = make_candles(1)[0]
    invalid = Candle(
        symbol=bad.symbol,
        timeframe=bad.timeframe,
        open_time=bad.open_time,
        open=Decimal("0"),
        high=bad.high,
        low=bad.low,
        close=bad.close,
        volume=bad.volume,
        close_time=bad.close_time,
    )
    with pytest.raises(ValueError):
        candles_to_frame([invalid])


def test_enrich_indicators_marks_atr_spike_column() -> None:
    frame = enrich_indicators(candles_to_frame(make_candles()))
    assert "atr_spike" in frame.columns
