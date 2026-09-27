from __future__ import annotations

import asyncio

from config import Settings
from models.confidence_gate import ConfidenceGate
from models.lstm_model import LSTMSignal
from models.rl_agent import RLDecision


def run(coro):
    return asyncio.run(coro)


def strong_indicators(score: int = 80) -> dict:
    latest = {
        "ema_21": 110,
        "ema_50": 100,
        "macd_hist": 2,
        "rsi_14": 58,
        "di_plus": 30,
        "di_minus": 10,
        "close": 115,
        "vwap": 108,
        "cmf_20": 0.2,
        "atr_spike": False,
        "patterns": {"bias": {"direction": "LONG"}},
    }
    return {
        "timeframes": {"1h": {"latest": latest}},
        "confluence": {"score": score, "direction": "LONG"},
        "regime": "TRENDING_UP",
    }


def long_signal(confidence: float = 0.82, margin: float = 0.6, quality: float = 0.55) -> LSTMSignal:
    short_p = max(0.0, confidence - margin)
    return LSTMSignal(
        "LONG",
        confidence,
        {"LONG": confidence, "SHORT": short_p, "NO_TRADE": max(0.0, 1 - confidence - short_p)},
        margin=margin,
        model_quality=quality,
    )


def futures_settings(**kwargs) -> Settings:
    return Settings(use_testnet=True, market_type="futures", **kwargs)


def test_gate_allows_degraded_news_mode_for_exceptional_setup() -> None:
    gate = ConfidenceGate(futures_settings(confidence_threshold=0.70))
    result = run(gate.passes(long_signal(0.95, 0.92), RLDecision("BUY", 0.95), strong_indicators(95), "BTCUSDT"))
    assert result.approved
    assert result.indicator_agreement >= 4
    assert result.conviction > 0


def test_gate_blocks_missing_calendar_without_exceptional_confidence() -> None:
    gate = ConfidenceGate(futures_settings(confidence_threshold=0.70, require_economic_calendar=True))
    result = run(gate.passes(long_signal(0.82, 0.6), RLDecision("BUY", 0.75), strong_indicators(), "BTCUSDT"))
    assert not result.approved
    assert any("major news window" in reason for reason in result.reasons)


def test_gate_rejects_short_on_spot_account() -> None:
    gate = ConfidenceGate(Settings(use_testnet=True, market_type="spot", confidence_threshold=0.50))
    signal = LSTMSignal(
        "SHORT",
        0.80,
        {"LONG": 0.05, "SHORT": 0.80, "NO_TRADE": 0.15},
        margin=0.75,
        model_quality=0.60,
    )
    result = run(gate.passes(signal, RLDecision("SELL", 0.80), strong_indicators(), "BNBUSDT"))
    assert not result.approved
    assert any("spot account" in reason for reason in result.reasons)


def test_gate_blocks_low_directional_margin() -> None:
    """A near-tie between LONG and SHORT is not an edge, however high the argmax."""
    gate = ConfidenceGate(futures_settings(confidence_threshold=0.35))
    signal = LSTMSignal(
        "LONG",
        0.40,
        {"LONG": 0.40, "SHORT": 0.38, "NO_TRADE": 0.22},
        margin=0.02,
        model_quality=0.60,
    )
    result = run(gate.passes(signal, RLDecision("BUY", 0.70), strong_indicators(95), "BTCUSDT"))
    assert not result.approved
    assert any("directional margin" in reason for reason in result.reasons)


def test_gate_blocks_untrained_model() -> None:
    gate = ConfidenceGate(futures_settings(confidence_threshold=0.50))
    result = run(gate.passes(long_signal(quality=0.10), RLDecision("BUY", 0.80), strong_indicators(95), "BTCUSDT"))
    assert not result.approved
    assert any("directional precision" in reason for reason in result.reasons)


def test_rl_hold_does_not_veto() -> None:
    """HOLD is the most common output of a trained agent; it must not block."""
    gate = ConfidenceGate(futures_settings(confidence_threshold=0.50))
    result = run(gate.passes(long_signal(0.95, 0.92), RLDecision("HOLD", 0.90), strong_indicators(95), "BTCUSDT"))
    assert result.approved


def test_rl_opposite_call_vetoes() -> None:
    gate = ConfidenceGate(futures_settings(confidence_threshold=0.50))
    result = run(gate.passes(long_signal(0.95, 0.92), RLDecision("SELL", 0.90), strong_indicators(95), "BTCUSDT"))
    assert not result.approved
    assert any("against a LONG setup" in reason for reason in result.reasons)


def test_cooldown_reason_blocks_reentry() -> None:
    gate = ConfidenceGate(futures_settings(confidence_threshold=0.50))
    result = run(
        gate.passes(
            long_signal(0.95, 0.92),
            RLDecision("BUY", 0.90),
            strong_indicators(95),
            "BTCUSDT",
            cooldown_reason="cooling off after a loss on BTCUSDT for another 45 minute(s)",
        )
    )
    assert not result.approved
    assert any("cooling off" in reason for reason in result.reasons)


def test_conviction_rises_with_evidence() -> None:
    gate = ConfidenceGate(futures_settings(confidence_threshold=0.50))
    weak = run(gate.passes(long_signal(0.55, 0.20, 0.45), RLDecision("HOLD", 0.5), strong_indicators(65), "BTCUSDT"))
    strong = run(gate.passes(long_signal(0.90, 0.85, 0.68), RLDecision("BUY", 0.88), strong_indicators(98), "BTCUSDT"))
    assert strong.conviction > weak.conviction
