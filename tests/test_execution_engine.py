from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

from trading.execution_engine import ExecutionEngine, ManagedTrade
from trading.risk_manager import RiskManager, TradePlan
from utils.binance_client import OrderResult, SymbolFilters


class FakeClient:
    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self.current_price = Decimal("100")
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
        tp1_price=Decimal("106"),
        tp2_price=Decimal("109"),
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
