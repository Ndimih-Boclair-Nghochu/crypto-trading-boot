from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from learning.journal import Journal
from trading.risk_manager import RiskManager, TradePlan
from utils.binance_client import OrderResult, ResilientBinanceClient
from utils.logger import logger


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    order_id: str | None = None
    status: str = "REJECTED"
    reason: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ManagedTrade:
    plan: TradePlan
    entry_order_id: str
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    tp1_hit: bool = False
    remaining_quantity: Decimal = Decimal("0")
    trailing_stop: Decimal | None = None


class ExecutionEngine:
    def __init__(
        self,
        client: ResilientBinanceClient,
        risk_manager: RiskManager,
        journal: Journal | None = None,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.journal = journal
        self.open_trades: dict[str, ManagedTrade] = {}
        self.lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None

    async def place_trade(self, plan: TradePlan) -> ExecutionResult:
        if not plan.approved or not plan.symbol or not plan.direction:
            return ExecutionResult(False, reason=plan.reason)
        async with self.lock:
            try:
                balance = await self.client.get_usdt_balance()
                if balance <= 0 or (plan.entry_price * plan.quantity) > balance:
                    return ExecutionResult(False, reason="insufficient final Binance balance")

                side = "BUY" if plan.direction == "LONG" else "SELL"
                entry = await self.client.place_order(
                    symbol=plan.symbol,
                    side=side,
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity=_fmt(plan.quantity),
                    price=_fmt(plan.entry_price),
                )
                if not entry.accepted or not entry.order_id:
                    return ExecutionResult(False, status=entry.status, reason=entry.reason, raw=entry.raw)

                filled = await self._wait_for_fill(plan.symbol, entry.order_id, timeout_seconds=30)
                entry_order_id = entry.order_id
                if not filled:
                    await self.client.cancel_order(plan.symbol, entry.order_id)
                    fallback = await self.client.place_order(
                        symbol=plan.symbol,
                        side=side,
                        type="MARKET",
                        quantity=_fmt(plan.quantity),
                    )
                    if not fallback.accepted or not fallback.order_id:
                        return ExecutionResult(False, status=fallback.status, reason=fallback.reason, raw=fallback.raw)
                    entry_order_id = fallback.order_id

                await self._place_protection(plan)
                self.risk_manager.register_open_position(plan)
                self.open_trades[plan.symbol] = ManagedTrade(plan=plan, entry_order_id=entry_order_id, remaining_quantity=plan.quantity)
                if self.journal:
                    await self.journal.log_trade_open(plan, entry_order_id)
                self.start_monitoring()
                return ExecutionResult(True, entry_order_id, "FILLED_OR_MONITORING", raw=entry.raw)
            except Exception as exc:
                logger.exception(f"Trade execution failed closed for {plan.symbol}: {exc}")
                return ExecutionResult(False, reason=str(exc))

    async def _wait_for_fill(self, symbol: str, order_id: str, timeout_seconds: int) -> bool:
        deadline = datetime.now(UTC) + timedelta(seconds=timeout_seconds)
        while datetime.now(UTC) < deadline:
            order = await self.client.get_order(symbol, order_id)
            if order and order.get("status") == "FILLED":
                return True
            await asyncio.sleep(3)
        return False

    async def _place_protection(self, plan: TradePlan) -> OrderResult:
        side = "SELL" if plan.direction == "LONG" else "BUY"
        if plan.direction == "LONG":
            oco_kwargs = {
                "aboveType": "LIMIT_MAKER",
                "abovePrice": _fmt(plan.tp1_price),
                "belowType": "STOP_LOSS_LIMIT",
                "belowStopPrice": _fmt(plan.sl_price),
                "belowPrice": _fmt(plan.sl_price),
                "belowTimeInForce": "GTC",
            }
        else:
            oco_kwargs = {
                "aboveType": "STOP_LOSS_LIMIT",
                "aboveStopPrice": _fmt(plan.sl_price),
                "abovePrice": _fmt(plan.sl_price),
                "aboveTimeInForce": "GTC",
                "belowType": "LIMIT_MAKER",
                "belowPrice": _fmt(plan.tp1_price),
            }
        return await self.client.place_oco_order(
            symbol=plan.symbol,
            side=side,
            quantity=_fmt(plan.quantity / Decimal("2")),
            **oco_kwargs,
        )

    def start_monitoring(self) -> None:
        if self._monitor_task and not self._monitor_task.done():
            return
        self._monitor_task = asyncio.create_task(self.monitor_open_trades())

    async def monitor_open_trades(self) -> None:
        while True:
            try:
                await self._monitor_once()
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"Execution monitor error: {exc}")
                await asyncio.sleep(10)

    async def _monitor_once(self) -> None:
        async with self.lock:
            for symbol, managed in list(self.open_trades.items()):
                plan = managed.plan
                price = await self._latest_price(symbol)
                if price <= 0:
                    continue
                atr_value = Decimal(str(plan.indicator_state.get("latest", {}).get("atr_14", "0") or "0"))
                if atr_value > 0:
                    managed.trailing_stop = self.risk_manager.update_trailing_stop(symbol, price, atr_value)

                if not managed.tp1_hit and self._target_hit(plan.direction or "", price, plan.tp1_price):
                    await self._close_quantity(symbol, plan.direction or "", managed.remaining_quantity / Decimal("2"))
                    managed.remaining_quantity /= Decimal("2")
                    managed.tp1_hit = True
                    if self.journal:
                        await self.journal.log_event("TP1_HIT", "INFO", f"{symbol} TP1 hit", {"price": str(price)})

                if managed.trailing_stop and self._stop_hit(plan.direction or "", price, managed.trailing_stop):
                    await self._close_quantity(symbol, plan.direction or "", managed.remaining_quantity)
                    self.open_trades.pop(symbol, None)
                    self.risk_manager.register_closed_trade(symbol, Decimal("0"), Decimal("0"))
                    if self.journal:
                        await self.journal.log_trade_exit(symbol, price, "TRAILING")
                    continue

                if datetime.now(UTC) - managed.opened_at >= timedelta(hours=4):
                    meaningful_move = abs(price - plan.entry_price) >= atr_value if atr_value > 0 else True
                    if not meaningful_move:
                        await self._close_quantity(symbol, plan.direction or "", managed.remaining_quantity)
                        self.open_trades.pop(symbol, None)
                        self.risk_manager.register_closed_trade(symbol, Decimal("0"), Decimal("0"))
                        if self.journal:
                            await self.journal.log_trade_exit(symbol, price, "TIME_STOP")

    async def _latest_price(self, symbol: str) -> Decimal:
        candles = await self.client.get_ohlcv(symbol, "1m", 1)
        if not candles:
            return Decimal("0")
        return candles[-1].close

    async def _close_quantity(self, symbol: str, direction: str, quantity: Decimal) -> ExecutionResult:
        if quantity <= 0:
            return ExecutionResult(False, reason="quantity <= 0")
        side = "SELL" if direction == "LONG" else "BUY"
        result = await self.client.place_order(symbol=symbol, side=side, type="MARKET", quantity=_fmt(quantity))
        return ExecutionResult(result.accepted, result.order_id, result.status, result.reason, result.raw)

    async def emergency_close_all(self) -> list[ExecutionResult]:
        async with self.lock:
            results = []
            for symbol, managed in list(self.open_trades.items()):
                results.append(await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity))
                self.open_trades.pop(symbol, None)
                if self.journal:
                    await self.journal.log_trade_exit(symbol, Decimal("0"), "MANUAL")
            return results

    async def emergency_close_symbol(self, symbol: str) -> ExecutionResult:
        async with self.lock:
            managed = self.open_trades.get(symbol)
            if not managed:
                return ExecutionResult(False, reason="symbol not managed")
            result = await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity)
            if result.accepted:
                self.open_trades.pop(symbol, None)
                if self.journal:
                    await self.journal.log_trade_exit(symbol, Decimal("0"), "MANUAL")
            return result

    async def reconcile_orders(self) -> None:
        try:
            remote_orders = await self.client.get_open_orders()
            remote_symbols = {order.get("symbol") for order in remote_orders}
            internal_symbols = set(self.open_trades)
            unknown = remote_symbols - internal_symbols
            if unknown and self.journal:
                await self.journal.log_event(
                    "ORDER_RECONCILIATION",
                    "CRITICAL",
                    "Binance has open orders unknown to local state",
                    {"symbols": sorted(symbol for symbol in unknown if symbol)},
                )
        except Exception as exc:
            logger.exception(f"Order reconciliation failed: {exc}")

    def _target_hit(self, direction: str, price: Decimal, target: Decimal) -> bool:
        return price >= target if direction == "LONG" else price <= target

    def _stop_hit(self, direction: str, price: Decimal, stop: Decimal) -> bool:
        return price <= stop if direction == "LONG" else price >= stop


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")
