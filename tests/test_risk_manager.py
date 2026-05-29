from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from config import Settings
from trading.risk_manager import RiskManager, TradeCandidate
from trading.strategy_engine import TradeSignal


def signal(direction: str = "LONG") -> TradeSignal:
    return TradeSignal(
        symbol="BTCUSDT",
        direction=direction,
        strategy_used="EMA_TREND_PULLBACK_LONG",
        regime_at_entry="TRENDING_UP",
        lstm_confidence=0.8,
        rl_confidence=0.75,
        confluence_score=80,
        indicator_agreement=5,
        indicator_state={"latest": {"close": 100, "atr_14": 2}},
    )


def run(coro):
    return asyncio.run(coro)


def test_risk_manager_approves_conservative_position() -> None:
    manager = RiskManager(Settings())
    plan = run(
        manager.calculate(
            TradeCandidate(
                signal=signal(),
                entry_price=100,
                atr=2,
                account_balance=10_000,
                available_balance=10_000,
            )
        )
    )
    assert plan.approved
    assert plan.quantity == Decimal("25.00000000")
    assert plan.sl_price == Decimal("97.0")
    assert plan.tp1_price == Decimal("106.0")
    assert plan.reward_risk == Decimal("2")


def test_daily_loss_limit_blocks_trade() -> None:
    manager = RiskManager(Settings(max_daily_loss_pct=4))
    manager.daily_realized[datetime.now(UTC).date()] = Decimal("-500")
    plan = run(
        manager.calculate(
            TradeCandidate(
                signal=signal(),
                entry_price=100,
                atr=2,
                account_balance=10_000,
                available_balance=10_000,
            )
        )
    )
    assert not plan.approved
    assert plan.reason == "daily loss limit hit"


def test_drawdown_circuit_breaker_blocks_new_entries() -> None:
    manager = RiskManager(Settings(drawdown_circuit_breaker_pct=10))
    manager.peak_equity = Decimal("10000")
    plan = run(
        manager.calculate(
            TradeCandidate(
                signal=signal(),
                entry_price=100,
                atr=2,
                account_balance=8_900,
                available_balance=8_900,
            )
        )
    )
    assert not plan.approved
    assert plan.reason == "drawdown circuit breaker active"


def test_max_concurrent_trades_blocks_trade() -> None:
    manager = RiskManager(Settings(max_concurrent_trades=1))
    first = run(
        manager.calculate(
            TradeCandidate(signal=signal(), entry_price=100, atr=2, account_balance=10_000, available_balance=10_000)
        )
    )
    manager.register_open_position(first)
    second = run(
        manager.calculate(
            TradeCandidate(signal=signal("LONG"), entry_price=100, atr=2, account_balance=10_000, available_balance=10_000)
        )
    )
    assert not second.approved
    assert second.reason == "max concurrent open trades exceeded"
