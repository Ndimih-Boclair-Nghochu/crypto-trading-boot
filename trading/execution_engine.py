from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from learning.journal import Journal
from trading.risk_manager import RiskManager, TradePlan
from utils.alerts import AlertManager
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
        self.alerter = AlertManager()

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
                await self.alerter.send(
                    f"✅ TRADE OPENED: {plan.symbol} {plan.direction}",
                    {
                        "entry": str(plan.entry_price),
                        "sl": str(plan.sl_price),
                        "tp1": str(plan.tp1_price),
                        "size": str(plan.quantity),
                        "strategy": plan.strategy_used,
                        "confidence": f"{plan.lstm_confidence:.0%}",
                    },
                )
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
                    close_qty = managed.remaining_quantity / Decimal("2")
                    await self._close_quantity(symbol, plan.direction or "", close_qty)
                    managed.remaining_quantity /= Decimal("2")
                    if symbol in self.risk_manager.open_positions:
                        self.risk_manager.open_positions[symbol].quantity = managed.remaining_quantity
                    managed.tp1_hit = True
                    partial_pnl = self._calc_pnl(plan, price, close_qty)
                    partial_r = self._calc_r_multiple(plan, price)
                    self.risk_manager.register_closed_trade(symbol, partial_pnl, partial_r, remove_position=False)
                    if self.journal:
                        await self.journal.log_partial_exit(symbol, price, close_qty, partial_pnl, partial_r, "TP1")
                    await self._send_close_alert(symbol, price, "TP1", True)

                if not managed.tp1_hit:
                    hard_sl = plan.sl_price
                    if self._stop_hit(plan.direction or "", price, hard_sl):
                        await self._close_quantity(symbol, plan.direction or "", managed.remaining_quantity)
                        pnl = self._calc_pnl(plan, price, managed.remaining_quantity)
                        r_multiple = self._calc_r_multiple(plan, price)
                        self.open_trades.pop(symbol, None)
                        self.risk_manager.register_closed_trade(symbol, pnl, r_multiple)
                        if self.journal:
                            await self.journal.log_trade_exit(symbol, price, "SL")
                        await self._send_close_alert(symbol, price, "SL", False)
                        continue

                if managed.trailing_stop and self._stop_hit(plan.direction or "", price, managed.trailing_stop):
                    await self._close_quantity(symbol, plan.direction or "", managed.remaining_quantity)
                    pnl = self._calc_pnl(plan, price, managed.remaining_quantity)
                    r_multiple = self._calc_r_multiple(plan, price)
                    self.open_trades.pop(symbol, None)
                    self.risk_manager.register_closed_trade(symbol, pnl, r_multiple)
                    if self.journal:
                        await self.journal.log_trade_exit(symbol, price, "TRAILING")
                    await self._send_close_alert(symbol, price, "TRAILING", self._is_win(plan, price))
                    continue

                if datetime.now(UTC) - managed.opened_at >= timedelta(hours=4):
                    meaningful_move = abs(price - plan.entry_price) >= atr_value if atr_value > 0 else True
                    if not meaningful_move:
                        await self._close_quantity(symbol, plan.direction or "", managed.remaining_quantity)
                        pnl = self._calc_pnl(plan, price, managed.remaining_quantity)
                        r_multiple = self._calc_r_multiple(plan, price)
                        self.open_trades.pop(symbol, None)
                        self.risk_manager.register_closed_trade(symbol, pnl, r_multiple)
                        if self.journal:
                            await self.journal.log_trade_exit(symbol, price, "TIME_STOP")
                        await self._send_close_alert(symbol, price, "TIME_STOP", self._is_win(plan, price))

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

    def _calc_pnl(self, plan: TradePlan, exit_price: Decimal, quantity: Decimal) -> Decimal:
        if plan.direction == "LONG":
            return (exit_price - plan.entry_price) * quantity
        return (plan.entry_price - exit_price) * quantity

    def _calc_r_multiple(self, plan: TradePlan, exit_price: Decimal) -> Decimal:
        risk_per_unit = abs(plan.entry_price - plan.sl_price)
        if risk_per_unit == 0:
            return Decimal("0")
        if plan.direction == "LONG":
            return (exit_price - plan.entry_price) / risk_per_unit
        return (plan.entry_price - exit_price) / risk_per_unit

    async def _send_close_alert(self, symbol: str, price: Decimal, exit_reason: str, is_win: bool) -> None:
        await self.alerter.send(
            f"{'✅ WIN' if is_win else '❌ LOSS'}: {symbol} closed",
            {"exit_price": str(price), "reason": exit_reason, "pnl_approx": "see dashboard"},
        )

    def _is_win(self, plan: TradePlan, price: Decimal) -> bool:
        if plan.direction == "LONG":
            return price >= plan.entry_price
        return price <= plan.entry_price

    async def emergency_close_all(self) -> list[ExecutionResult]:
        async with self.lock:
            results = []
            for symbol, managed in list(self.open_trades.items()):
                results.append(await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity))
                self.open_trades.pop(symbol, None)
                if self.journal:
                    await self.journal.log_trade_exit(symbol, Decimal("0"), "MANUAL")
                await self.alerter.send(
                    f"MANUAL CLOSE: {symbol}",
                    {"reason": "MANUAL", "pnl_approx": "see dashboard"},
                )
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
                await self.alerter.send(
                    f"MANUAL CLOSE: {symbol}",
                    {"reason": "MANUAL", "pnl_approx": "see dashboard"},
                )
            return result

    async def reconcile_orders(self) -> None:
        try:
            remote_orders = await self.client.get_open_orders()
            remote_symbols = {order.get("symbol") for order in remote_orders if order.get("symbol")}
            internal_symbols = set(self.open_trades.keys())
            unknown = remote_symbols - internal_symbols
            if unknown and self.journal:
                await self.journal.log_event(
                    "ORDER_RECONCILIATION",
                    "CRITICAL",
                    "Binance has open orders unknown to local state",
                    {"symbols": sorted(unknown)},
                )
            stale = internal_symbols - remote_symbols
            for symbol in stale:
                managed = self.open_trades.get(symbol)
                if not managed:
                    continue
                try:
                    await self.client.get_open_orders(symbol)
                    price = await self._latest_price(symbol)
                    if price <= 0:
                        price = managed.plan.entry_price
                    pnl = self._calc_pnl(managed.plan, price, managed.remaining_quantity)
                    r_multiple = self._calc_r_multiple(managed.plan, price)
                    self.open_trades.pop(symbol, None)
                    self.risk_manager.register_closed_trade(symbol, pnl, r_multiple)
                    if self.journal:
                        await self.journal.log_trade_exit(symbol, price, "SL")
                        await self.journal.log_event(
                            "RECONCILE_CLOSE",
                            "WARNING",
                            f"{symbol} position found closed on Binance (OCO fired); cleaned up local state",
                            {"symbol": symbol, "approx_exit": str(price)},
                        )
                    logger.warning(f"Reconcile: {symbol} was closed on Binance (OCO fired); local state cleaned up")
                except Exception as exc:
                    logger.warning(f"Reconcile cleanup failed for {symbol}: {exc}")
        except Exception as exc:
            logger.exception(f"Order reconciliation failed: {exc}")

    def _target_hit(self, direction: str, price: Decimal, target: Decimal) -> bool:
        return price >= target if direction == "LONG" else price <= target

    def _stop_hit(self, direction: str, price: Decimal, stop: Decimal) -> bool:
        return price <= stop if direction == "LONG" else price >= stop


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")
