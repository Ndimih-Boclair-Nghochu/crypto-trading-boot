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


def candidate(conviction: float = 0.0, **kwargs) -> TradeCandidate:
    defaults = {
        "signal": signal(),
        "entry_price": 100,
        "atr": 2,
        "account_balance": 10_000,
        "available_balance": 10_000,
        "conviction": conviction,
    }
    defaults.update(kwargs)
    return TradeCandidate(**defaults)


def sizing_settings(**kwargs) -> Settings:
    # max_position_pct is lifted so the risk-based leg of the sizing calculation
    # binds rather than the notional cap; the cap is exercised separately.
    base = {"max_position_pct": 100.0, "max_risk_per_trade_pct": 1.5, "risk_min_pct": 0.25, "risk_max_pct": 1.25,
            "final_target_r_multiple": 4.0, "pure_tp_sl": False}
    base.update(kwargs)
    return Settings(**base)


def test_risk_manager_builds_r_based_levels() -> None:
    plan = run(RiskManager(sizing_settings()).calculate(candidate()))
    assert plan.approved
    # 1.5 x ATR(2) = 3 of risk per unit
    assert plan.initial_risk_per_unit == Decimal("3.0")
    assert plan.sl_price == Decimal("97.0")
    assert plan.tp1_price == Decimal("103.0")
    assert plan.tp2_price == Decimal("106.0")
    assert plan.final_target_price == Decimal("112.0")
    assert plan.reward_risk == Decimal("4")


def test_position_size_scales_with_conviction() -> None:
    manager = sizing_settings()
    weak = run(RiskManager(manager).calculate(candidate(conviction=0.0)))
    middling = run(RiskManager(manager).calculate(candidate(conviction=0.5)))
    strong = run(RiskManager(manager).calculate(candidate(conviction=1.0)))
    assert weak.quantity < middling.quantity < strong.quantity
    # Floor and ceiling come from RISK_MIN_PCT / RISK_MAX_PCT. risk_pct is
    # recomputed from the lot-size-quantized quantity, so compare to a tolerance
    # rather than exactly.
    assert abs(weak.risk_pct - Decimal("0.25")) < Decimal("0.001")
    assert abs(strong.risk_pct - Decimal("1.25")) < Decimal("0.001")


def test_notional_cap_limits_size() -> None:
    plan = run(RiskManager(Settings(max_position_pct=2.0)).calculate(candidate(conviction=1.0)))
    assert plan.approved
    # 2% of 10,000 at a price of 100 is 2 units, well below the risk-based size.
    assert plan.quantity == Decimal("2.00000000")


def test_daily_loss_limit_blocks_trade() -> None:
    manager = RiskManager(sizing_settings(max_daily_loss_pct=4))
    manager.daily_realized[datetime.now(UTC).date()] = Decimal("-500")
    plan = run(manager.calculate(candidate()))
    assert not plan.approved
    assert plan.reason == "daily loss limit hit"


def test_drawdown_circuit_breaker_blocks_then_clears() -> None:
    manager = RiskManager(sizing_settings(drawdown_circuit_breaker_pct=10, circuit_breaker_reset_pct=5))
    manager.peak_equity = Decimal("10000")
    blocked = run(manager.calculate(candidate(account_balance=8_900, available_balance=8_900)))
    assert not blocked.approved
    assert blocked.reason == "drawdown circuit breaker active"

    # Recovering above the reset threshold re-arms the system. Without this the
    # flag latched forever and only a process restart resumed trading.
    manager.circuit_breaker_hit(Decimal("9_700"))
    assert not manager.circuit_breaker_active
    assert run(manager.calculate(candidate(account_balance=9_700, available_balance=9_700))).approved


def test_max_concurrent_trades_blocks_trade() -> None:
    manager = RiskManager(sizing_settings(max_concurrent_trades=1))
    manager.register_open_position(run(manager.calculate(candidate())))
    second = run(manager.calculate(candidate()))
    assert not second.approved
    assert second.reason == "max concurrent open trades exceeded"


def test_half_kelly_caps_size_after_a_weak_run() -> None:
    no_history = run(RiskManager(sizing_settings()).calculate(candidate(conviction=1.0)))

    manager = RiskManager(sizing_settings())
    for idx in range(11):
        manager.register_closed_trade(f"WIN{idx}", Decimal("100"), Decimal("0.5"))
    for idx in range(11):
        manager.register_closed_trade(f"LOSS{idx}", Decimal("-100"), Decimal("-1"))

    kelly = manager._half_kelly_risk_fraction()
    assert kelly is not None
    adjusted = run(manager.calculate(candidate(conviction=1.0)))
    assert adjusted.quantity < no_history.quantity


def test_profit_lock_moves_stop_above_entry_and_never_back() -> None:
    manager = RiskManager(Settings(profit_lock_arm_r=0.25, trail_arm_r=1.0, profit_lock_give_back=0.5))
    plan = run(RiskManager(sizing_settings()).calculate(candidate(conviction=0.5)))
    manager.register_open_position(plan)

    # Not yet armed: the stop is still the original one.
    assert manager.update_protective_stop("BTCUSDT", Decimal("100.5"), Decimal("2")) == Decimal("97.0")

    # 0.3R in front: the stop clears entry plus round-trip costs, so this trade
    # can no longer be closed at a loss.
    armed = manager.update_protective_stop("BTCUSDT", Decimal("101"), Decimal("2"))
    assert armed > plan.entry_price

    # 2R in front: keep all but the configured give-back.
    ratcheted = manager.update_protective_stop("BTCUSDT", Decimal("106"), Decimal("2"))
    assert ratcheted >= plan.entry_price + Decimal("1.5") * plan.initial_risk_per_unit

    # Price falling back must not loosen the stop.
    assert manager.update_protective_stop("BTCUSDT", Decimal("101"), Decimal("2")) == ratcheted


def test_loss_starts_cooldown_and_win_does_not() -> None:
    manager = RiskManager(Settings(loss_cooldown_bars=2))
    manager.register_closed_trade("BTCUSDT", Decimal("-50"), Decimal("-1"), bar_seconds=3600)
    assert "cooling off" in (manager.cooldown_reason("BTCUSDT") or "")

    manager.register_closed_trade("ETHUSDT", Decimal("50"), Decimal("1"), bar_seconds=3600)
    assert manager.cooldown_reason("ETHUSDT") is None
