from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from config import settings as _base_settings
from trading.execution_engine import ExecutionEngine, ManagedTrade
from trading.risk_manager import RiskManager, TradePlan
from utils.binance_client import OrderResult, SymbolFilters


@pytest.fixture(autouse=True)
def _full_exit_stack(monkeypatch):
    # These tests exercise the full event-based exit stack (peak-guard, scale-outs,
    # breakeven, profit-floor). Production may run with PURE_TP_SL=true, which
    # disables all of them; force it off here so the non-pure paths are tested.
    # The dedicated pure-mode test flips it back on for itself.
    monkeypatch.setattr("trading.execution_engine.settings", replace(_base_settings, pure_tp_sl=False))


class FakeClient:
    is_futures = False

    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self.current_price = Decimal("100")
        self.free_base = Decimal("1000")
        self.open_orders: list[dict] = []
        self.filters = SymbolFilters(
            symbol="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.0001"),
            min_notional=Decimal("10"),
            min_qty=Decimal("0.0001"),
        )

    async def get_usdt_balance(self) -> Decimal:
        return Decimal("10000")

    async def get_asset_free(self, asset: str) -> Decimal:
        return self.free_base

    async def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return OrderResult(True, order_id=str(len(self.orders)), status="FILLED", raw={"status": "FILLED"})

    async def place_oco_order(self, **kwargs):
        self.orders.append({"oco": kwargs})
        return OrderResult(True, order_id="oco-1", status="EXECUTING", raw={})

    async def get_order(self, symbol: str, order_id: str):
        return {"status": "FILLED"}

    async def cancel_order(self, symbol: str, order_id: str):
        self.cancelled.append(order_id)
        return True

    async def cancel_all_orders(self, symbol: str):
        for order in [o for o in self.open_orders if o.get("symbol") == symbol]:
            oid = order.get("orderId")
            if oid is not None:
                self.cancelled.append(str(oid))
        return True

    async def get_open_orders(self, symbol=None):
        if symbol:
            return [order for order in self.open_orders if order.get("symbol") == symbol]
        return self.open_orders

    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 1):
        return [SimpleNamespace(close=self.current_price)]

    async def get_symbol_filters(self, symbol: str):
        return self.filters if symbol == "BTCUSDT" else None


class FakeJournal:
    def __init__(self) -> None:
        self.exits = []
        self.events = []
        self.partials = []

    async def log_trade_exit(self, symbol, exit_price, exit_reason):
        self.exits.append({"symbol": symbol, "exit_price": exit_price, "exit_reason": exit_reason})

    async def log_event(self, event_type, severity, message, context=None):
        self.events.append({"event_type": event_type, "severity": severity, "message": message, "context": context or {}})

    async def log_partial_exit(self, symbol, exit_price, quantity_closed, pnl_usd, r_multiple, reason="TP1"):
        self.partials.append(
            {
                "symbol": symbol,
                "exit_price": exit_price,
                "quantity_closed": quantity_closed,
                "pnl_usd": pnl_usd,
                "r_multiple": r_multiple,
                "reason": reason,
            }
        )


def plan() -> TradePlan:
    return TradePlan(
        approved=True,
        reason="approved",
        checklist=[],
        symbol="BTCUSDT",
        direction="LONG",
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        sl_price=Decimal("97"),
        tp1_price=Decimal("103"),
        tp2_price=Decimal("106"),
        final_target_price=Decimal("112"),
        initial_risk_per_unit=Decimal("3"),
        strategy_used="EMA",
        regime_at_entry="TRENDING_UP",
        indicator_state={"latest": {"atr_14": 2}},
    )


def run(coro):
    return asyncio.run(coro)


def test_execution_places_limit_then_protection() -> None:
    fake = FakeClient()
    engine = ExecutionEngine(fake, RiskManager())
    result = run(engine.place_trade(plan()))
    assert result.accepted
    assert fake.orders[0]["type"] == "LIMIT"
    assert "oco" in fake.orders[1]
    if engine._monitor_task:
        engine._monitor_task.cancel()


def test_execution_falls_back_to_market_when_limit_unfilled() -> None:
    fake = FakeClient()
    engine = ExecutionEngine(fake, RiskManager())

    async def unfilled(*args, **kwargs):
        return False

    engine._wait_for_fill = unfilled  # type: ignore[method-assign]
    result = run(engine.place_trade(plan()))
    assert result.accepted
    assert fake.cancelled == ["1"]
    assert fake.orders[1]["type"] == "MARKET"
    if engine._monitor_task:
        engine._monitor_task.cancel()


def test_emergency_close_closes_managed_trade() -> None:
    fake = FakeClient()
    engine = ExecutionEngine(fake, RiskManager())
    async def scenario():
        await engine.place_trade(plan())
        return await engine.emergency_close_all()

    results = run(scenario())
    assert results[0].accepted
    assert "BTCUSDT" not in engine.open_trades
    if engine._monitor_task:
        engine._monitor_task.cancel()


def test_sl_hit_cleans_up_local_state() -> None:
    fake = FakeClient()
    fake.current_price = Decimal("96")
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    journal = FakeJournal()
    engine = ExecutionEngine(fake, risk, journal)  # type: ignore[arg-type]
    engine.open_trades["BTCUSDT"] = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)

    run(engine._monitor_once())

    assert "BTCUSDT" not in engine.open_trades
    assert "BTCUSDT" not in risk.open_positions
    assert risk.closed_trades[-1].pnl_usd < 0
    assert risk.closed_trades[-1].r_multiple < 0
    assert journal.exits[-1]["exit_reason"] == "SL"


def test_reconcile_cleans_stale_ghost_position() -> None:
    fake = FakeClient()
    fake.current_price = Decimal("96")
    fake.open_orders = []
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    journal = FakeJournal()
    engine = ExecutionEngine(fake, risk, journal)  # type: ignore[arg-type]
    engine.open_trades["BTCUSDT"] = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)

    run(engine.reconcile_orders())

    assert "BTCUSDT" not in engine.open_trades
    assert "BTCUSDT" not in risk.open_positions
    assert journal.exits[-1]["exit_reason"] == "SL"
    assert journal.events[-1]["event_type"] == "RECONCILE_CLOSE"


def test_peak_guard_banks_a_winner_that_rolls_over() -> None:
    """A trade that peaked at +1R and slipped back exits in profit, instantly."""
    fake = FakeClient()
    fake.current_price = Decimal("102")  # +0.67R now (entry 100, 1R = 3)
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    journal = FakeJournal()
    engine = ExecutionEngine(fake, risk, journal)  # type: ignore[arg-type]
    managed = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)
    managed.peak_r = Decimal("1.0")  # it had been +1R in front
    engine.open_trades["BTCUSDT"] = managed
    engine.update_market_context("BTCUSDT", {"atr_14": 2})

    run(engine._monitor_once())

    assert "BTCUSDT" not in engine.open_trades
    assert journal.exits[-1]["exit_reason"] == "PEAK_GUARD"
    assert risk.closed_trades[-1].pnl_usd > 0  # banked while still green


def test_peak_guard_does_not_fire_before_arming() -> None:
    """A tiny wiggle near breakeven must not trip the guard."""
    fake = FakeClient()
    fake.current_price = Decimal("100.3")
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    engine = ExecutionEngine(fake, risk, FakeJournal())  # type: ignore[arg-type]
    managed = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)
    managed.peak_r = Decimal("0.2")  # never reached the 0.5R arm level
    engine.open_trades["BTCUSDT"] = managed
    engine.update_market_context("BTCUSDT", {"atr_14": 2})

    run(engine._monitor_once())

    assert "BTCUSDT" in engine.open_trades  # still open


def test_profit_floor_banks_a_proven_trade_that_reverses(monkeypatch) -> None:
    """A trade that reached +0.3R then fell back to the +0.05R floor exits as a
    small win -- the loss-side fix for the inverted payoff. Peak guard (arm 0.5R)
    cannot fire here because the trade only peaked at 0.4R, so the floor owns it."""
    from config import settings as _settings

    cfg = replace(_settings, profit_floor_enabled=True, profit_floor_arm_r=0.3, profit_floor_r=0.05, pure_tp_sl=False)
    monkeypatch.setattr("trading.execution_engine.settings", cfg)
    fake = FakeClient()
    fake.current_price = Decimal("100.15")  # +0.05R (entry 100, 1R = 3)
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    journal = FakeJournal()
    engine = ExecutionEngine(fake, risk, journal)  # type: ignore[arg-type]
    managed = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)
    managed.peak_r = Decimal("0.4")  # had proven itself past the 0.3R arm
    engine.open_trades["BTCUSDT"] = managed
    engine.update_market_context("BTCUSDT", {"atr_14": 2})

    run(engine._monitor_once())

    assert "BTCUSDT" not in engine.open_trades
    assert journal.exits[-1]["exit_reason"] == "PROFIT_FLOOR"
    assert risk.closed_trades[-1].pnl_usd > 0  # banked a small gain, not a -1R loss


def test_profit_floor_does_not_fire_before_arming(monkeypatch) -> None:
    """A trade that never reached the 0.3R arm keeps its full stop -- the floor
    must not clamp a trade that has not yet proved itself."""
    from config import settings as _settings

    cfg = replace(_settings, profit_floor_enabled=True, profit_floor_arm_r=0.3, profit_floor_r=0.05, pure_tp_sl=False)
    monkeypatch.setattr("trading.execution_engine.settings", cfg)
    fake = FakeClient()
    fake.current_price = Decimal("100.15")  # +0.05R, but never armed
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    engine = ExecutionEngine(fake, risk, FakeJournal())  # type: ignore[arg-type]
    managed = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)
    managed.peak_r = Decimal("0.2")  # below the 0.3R arm
    engine.open_trades["BTCUSDT"] = managed
    engine.update_market_context("BTCUSDT", {"atr_14": 2})

    run(engine._monitor_once())

    assert "BTCUSDT" in engine.open_trades  # still open


def test_manual_stop_flattens_untracked_position() -> None:
    """The Stop button must work even for a position opened before a restart."""
    fake = FakeClient()
    fake.current_price = Decimal("105")
    fake.free_base = Decimal("0.5")  # 0.5 BTC held on the exchange, not tracked
    engine = ExecutionEngine(fake, RiskManager())  # empty open_trades

    result = run(engine.emergency_close_symbol("BTCUSDT"))

    assert result.accepted
    assert any(o.get("side") == "SELL" and o.get("type") == "MARKET" for o in fake.orders)


def test_manual_stop_untracked_with_nothing_held_is_safe() -> None:
    fake = FakeClient()
    fake.free_base = Decimal("0")
    engine = ExecutionEngine(fake, RiskManager())

    result = run(engine.emergency_close_symbol("BTCUSDT"))

    assert not result.accepted
    assert fake.orders == []


def test_execution_rejects_min_notional_filter_violation() -> None:
    fake = FakeClient()
    fake.filters = SymbolFilters(
        symbol="BTCUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.0001"),
        min_notional=Decimal("250"),
        min_qty=Decimal("0.0001"),
    )
    engine = ExecutionEngine(fake, RiskManager())

    result = run(engine.place_trade(plan()))

    assert not result.accepted
    assert result.reason and "Filter violation" in result.reason
    assert fake.orders == []


def test_sell_entry_rejected_when_base_asset_is_not_held() -> None:
    """The -2010 loop: a SELL spends the base asset, not USDT."""
    fake = FakeClient()
    fake.free_base = Decimal("0")
    engine = ExecutionEngine(fake, RiskManager())
    short_plan = replace(plan(), direction="SHORT", sl_price=Decimal("103"), tp1_price=Decimal("97"), final_target_price=Decimal("88"))

    result = run(engine.place_trade(short_plan))

    assert not result.accepted
    assert result.reason and "insufficient BTC to sell" in result.reason
    assert fake.orders == []


def test_winner_that_rolls_over_exits_in_profit() -> None:
    """A trade in front that loses momentum is banked, not handed back."""
    fake = FakeClient()
    fake.current_price = Decimal("101")
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    journal = FakeJournal()
    engine = ExecutionEngine(fake, risk, journal)  # type: ignore[arg-type]
    engine.open_trades["BTCUSDT"] = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)
    engine.update_market_context("BTCUSDT", {"atr_14": 2, "close": 101, "ema_9": 102, "macd_hist": -1})

    run(engine._monitor_once())

    assert "BTCUSDT" not in engine.open_trades
    assert journal.exits[-1]["exit_reason"] == "REVERSAL"
    assert risk.closed_trades[-1].pnl_usd > 0


def test_profit_lock_arms_so_a_pullback_cannot_close_red() -> None:
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    engine = ExecutionEngine(FakeClient(), risk)

    # Runs the stop forward at +1R, then checks it survives a retrace.
    risk.update_protective_stop("BTCUSDT", Decimal("103"), Decimal("2"))
    stop = risk.open_positions["BTCUSDT"].sl_price
    assert stop > trade_plan.entry_price
    assert engine._r_multiple(trade_plan, stop) > 0


def test_tp1_scales_out_and_leaves_a_runner() -> None:
    fake = FakeClient()
    fake.current_price = Decimal("103")
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    journal = FakeJournal()
    engine = ExecutionEngine(fake, risk, journal)  # type: ignore[arg-type]
    engine.open_trades["BTCUSDT"] = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)
    engine.update_market_context("BTCUSDT", {"atr_14": 2})

    run(engine._monitor_once())

    managed = engine.open_trades["BTCUSDT"]
    assert "TP1" in managed.scaled_out
    assert managed.remaining_quantity < trade_plan.quantity
    assert managed.remaining_quantity > 0
    assert journal.partials[-1]["reason"] == "TP1"


def test_reconcile_labels_a_profitable_close_as_take_profit() -> None:
    """The old code journalled every remote close as 'SL', including winners."""
    fake = FakeClient()
    fake.current_price = Decimal("106")
    risk = RiskManager()
    trade_plan = plan()
    risk.register_open_position(trade_plan)
    journal = FakeJournal()
    engine = ExecutionEngine(fake, risk, journal)  # type: ignore[arg-type]
    engine.open_trades["BTCUSDT"] = ManagedTrade(trade_plan, "entry-1", remaining_quantity=trade_plan.quantity)

    run(engine.reconcile_orders())

    assert journal.exits[-1]["exit_reason"] == "TP"


def test_execution_rounds_order_to_symbol_filters() -> None:
    fake = FakeClient()
    engine = ExecutionEngine(fake, RiskManager())
    trade_plan = replace(
        plan(),
        quantity=Decimal("1.234567"),
        entry_price=Decimal("100.009"),
        sl_price=Decimal("97.009"),
        tp1_price=Decimal("106.009"),
    )

    result = run(engine.place_trade(trade_plan))

    assert result.accepted
    assert fake.orders[0]["quantity"] == "1.2345"
    assert fake.orders[0]["price"] == "100"
    if engine._monitor_task:
        engine._monitor_task.cancel()


def test_pure_tp_sl_ignores_events_exits_only_at_sl_or_tp(monkeypatch) -> None:
    """PURE_TP_SL: no event exit fires. A winner that rolled back but is still
    between SL (97) and TP (112) stays OPEN; it closes only at SL or the target."""
    monkeypatch.setattr("trading.execution_engine.settings", replace(_base_settings, pure_tp_sl=True))

    # 1) rolled-over winner, price 105 -- peak-guard/scale-outs would have closed it,
    #    but pure mode leaves it open.
    fake = FakeClient()
    fake.current_price = Decimal("105")
    risk = RiskManager()
    tp = plan()
    risk.register_open_position(tp)
    engine = ExecutionEngine(fake, risk, FakeJournal())  # type: ignore[arg-type]
    m = ManagedTrade(tp, "e1", remaining_quantity=tp.quantity)
    m.peak_r = Decimal("2.5")  # had been well in front
    engine.open_trades["BTCUSDT"] = m
    engine.update_market_context("BTCUSDT", {"atr_14": 2})
    run(engine._monitor_once())
    assert "BTCUSDT" in engine.open_trades  # nothing closed it

    # 2) price hits the take-profit target -> closes as TP.
    fake.current_price = Decimal("113")
    run(engine._monitor_once())
    assert "BTCUSDT" not in engine.open_trades
    assert engine.risk_manager.closed_trades[-1].pnl_usd > 0


def test_pure_tp_sl_closes_at_stop_loss(monkeypatch) -> None:
    monkeypatch.setattr("trading.execution_engine.settings", replace(_base_settings, pure_tp_sl=True))
    fake = FakeClient()
    fake.current_price = Decimal("96")  # below SL 97
    risk = RiskManager()
    tp = plan()
    risk.register_open_position(tp)
    journal = FakeJournal()
    engine = ExecutionEngine(fake, risk, journal)  # type: ignore[arg-type]
    engine.open_trades["BTCUSDT"] = ManagedTrade(tp, "e1", remaining_quantity=tp.quantity)
    run(engine._monitor_once())
    assert "BTCUSDT" not in engine.open_trades
    assert journal.exits[-1]["exit_reason"] == "SL"
