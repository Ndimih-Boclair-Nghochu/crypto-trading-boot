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
    }


def test_confidence_gate_allows_degraded_news_mode_for_exceptional_setup() -> None:
    gate = ConfidenceGate(Settings(use_testnet=True, confidence_threshold=0.70))
    result = run(
        gate.passes(
            LSTMSignal("LONG", 0.95, {"LONG": 0.95, "SHORT": 0.03, "NO_TRADE": 0.02}),
            RLDecision("BUY", 0.95),
            strong_indicators(95),
            "BTCUSDT",
        )
    )
    assert result.approved
    assert result.indicator_agreement >= 4


def test_confidence_gate_blocks_missing_calendar_without_exceptional_confidence() -> None:
    gate = ConfidenceGate(Settings(use_testnet=True, confidence_threshold=0.70))
    result = run(
        gate.passes(
            LSTMSignal("LONG", 0.82, {"LONG": 0.82, "SHORT": 0.1, "NO_TRADE": 0.08}),
            RLDecision("BUY", 0.75),
            strong_indicators(),
            "BTCUSDT",
        )
    )
    assert not result.approved
    assert any("major news window" in reason for reason in result.reasons)


def test_confidence_gate_blocks_live_without_calendar_api() -> None:
    gate = ConfidenceGate(Settings(use_testnet=False, live_trading_reviewed=True, testnet_trade_count=100))
    result = run(
        gate.passes(
            LSTMSignal("LONG", 0.82, {"LONG": 0.82, "SHORT": 0.1, "NO_TRADE": 0.08}),
            RLDecision("BUY", 0.75),
            strong_indicators(),
            "BTCUSDT",
        )
    )
    assert not result.approved
    assert any("major news window" in reason for reason in result.reasons)
