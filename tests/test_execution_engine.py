from __future__ import annotations

import asyncio
from decimal import Decimal

from trading.execution_engine import ExecutionEngine
from trading.risk_manager import RiskManager, TradePlan
from utils.binance_client import OrderResult


class FakeClient:
    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.cancelled: list[str] = []

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
        return []


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
