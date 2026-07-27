from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from config import settings
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
    remaining_quantity: Decimal = Decimal("0")
    scaled_out: set[str] = field(default_factory=set)
    protective_stop: Decimal = Decimal("0")
    protective_target: Decimal = Decimal("0")
    peak_r: Decimal = Decimal("0")


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
        # Freshest indicator snapshot per symbol, pushed by the main loop. The
        # monitor previously read ATR out of the plan captured at entry, so a
        # trade opened in one volatility regime kept being managed as if that
        # regime still held hours later.
        self.market_context: dict[str, dict[str, Any]] = {}

    def update_market_context(self, symbol: str, latest: dict[str, Any]) -> None:
        self.market_context[symbol] = latest or {}

    # ------------------------------------------------------------------- entry

    async def place_trade(self, plan: TradePlan) -> ExecutionResult:
        if not plan.approved or not plan.symbol or not plan.direction:
            return ExecutionResult(False, reason=plan.reason)
        async with self.lock:
            try:
                prepared = await self._prepare_plan_for_filters(plan)
                if isinstance(prepared, ExecutionResult):
                    return prepared
                plan = prepared

                side = "BUY" if plan.direction == "LONG" else "SELL"
                affordable = await self._can_afford(plan.symbol, side, plan.quantity, plan.entry_price)
                if affordable is not None:
                    return await self._reject_execution(plan, affordable)

                entry = await self.client.place_order(
                    symbol=plan.symbol,
                    side=side,
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity=_fmt(plan.quantity),
                    price=_fmt(plan.entry_price),
                )
                if not entry.accepted or not entry.order_id:
                    return await self._reject_execution(plan, entry.reason or "entry order rejected", entry.status, entry.raw)

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
                        return await self._reject_execution(
                            plan, fallback.reason or "market fallback rejected", fallback.status, fallback.raw
                        )
                    entry_order_id = fallback.order_id

                # Protection covers the *whole* position and targets the runner's
                # exit, not TP1. Previously the exchange-side OCO was placed for
                # half the quantity, so if this process died the other half sat
                # on Binance with no stop at all.
                target = _protective_target(plan)
                protection = await self._place_protection(plan, plan.quantity, plan.sl_price, target)
                if not protection.accepted:
                    await self._close_quantity(plan.symbol, plan.direction, plan.quantity)
                    return await self._reject_execution(
                        plan,
                        protection.reason or "protective OCO rejected; entry closed",
                        protection.status,
                        protection.raw,
                    )

                self.risk_manager.register_open_position(plan)
                self.open_trades[plan.symbol] = ManagedTrade(
                    plan=plan,
                    entry_order_id=entry_order_id,
                    remaining_quantity=plan.quantity,
                    protective_stop=plan.sl_price,
                    protective_target=target,
                )
                if self.journal:
                    await self.journal.log_trade_open(plan, entry_order_id)
                await self.alerter.send(
                    f"TRADE OPENED: {plan.symbol} {plan.direction}",
                    {
                        "entry": str(plan.entry_price),
                        "sl": str(plan.sl_price),
                        "tp1": str(plan.tp1_price),
                        "target": str(plan.final_target_price),
                        "size": str(plan.quantity),
                        "risk_pct": f"{plan.risk_pct:.3f}%",
                        "conviction": f"{plan.conviction:.2f}",
                    },
                )
                self.start_monitoring()
                return ExecutionResult(True, entry_order_id, "FILLED_OR_MONITORING", raw=entry.raw)
            except Exception as exc:
                logger.exception(f"Trade execution failed closed for {plan.symbol}: {exc}")
                return ExecutionResult(False, reason=str(exc))

    async def _can_afford(self, symbol: str, side: str, quantity: Decimal, price: Decimal) -> str | None:
        """Pre-flight the balance in the asset actually being spent.

        Futures: both sides spend USDT *margin* (notional / leverage), and a
        SELL opens a short rather than needing coins to sell. Spot: a BUY spends
        USDT, a SELL spends the base asset -- checking only USDT there let SHORT
        entries through to Binance, which refused every one with -2010.
        """
        if self.client.is_futures:
            available = await self.client.get_usdt_balance()
            if available <= 0:
                return "USDT margin balance unavailable or zero"
            leverage = Decimal(str(max(1, settings.futures_leverage)))
            margin_needed = (quantity * price) / leverage
            if margin_needed > available:
                return f"insufficient margin: need {margin_needed:.2f} USDT at {leverage}x, have {available:.2f}"
            return None

        if side == "BUY":
            balance = await self.client.get_usdt_balance()
            if balance <= 0:
                return "USDT balance unavailable or zero"
            if quantity * price > balance:
                return f"insufficient USDT: need {quantity * price}, have {balance}"
            return None

        base = _base_asset(symbol)
        free = await self.client.get_asset_free(base)
        if free < quantity:
            return (
                f"insufficient {base} to sell: need {quantity}, have {free}"
                + ("" if settings.shorting_available else " (spot accounts cannot short)")
            )
        return None

    async def _prepare_plan_for_filters(self, plan: TradePlan) -> TradePlan | ExecutionResult:
        if not plan.symbol:
            return ExecutionResult(False, reason="Filter violation: missing symbol")
        filters = await self.client.get_symbol_filters(plan.symbol)
        if not filters:
            return await self._reject_execution(plan, f"Filter violation: missing exchange filters for {plan.symbol}")
        quantity = filters.round_quantity(plan.quantity)
        entry_price = filters.round_price(plan.entry_price)
        ok, reason = filters.validate_order(entry_price, quantity)
        if not ok:
            return await self._reject_execution(plan, f"Filter violation: {reason}", "FILTER_REJECTED")
        if quantity != plan.quantity or entry_price != plan.entry_price:
            logger.info(
                f"Rounded {plan.symbol} order to Binance filters: qty {plan.quantity}->{quantity}, "
                f"entry {plan.entry_price}->{entry_price}"
            )
        return replace(
            plan,
            quantity=quantity,
            entry_price=entry_price,
            sl_price=filters.round_price(plan.sl_price),
            tp1_price=filters.round_price(plan.tp1_price),
            tp2_price=filters.round_price(plan.tp2_price) if plan.tp2_price else None,
            final_target_price=filters.round_price(plan.final_target_price),
        )

    async def _reject_execution(
        self,
        plan: TradePlan,
        reason: str,
        status: str = "REJECTED",
        raw: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        logger.warning(f"Execution rejected for {plan.symbol}: {reason}")
        if self.journal:
            await self.journal.log_event(
                "ORDER_REJECTED", "WARNING", reason, {"symbol": plan.symbol, "status": status, "raw": raw or {}}
            )
        await self.alerter.send(f"ORDER REJECTED: {plan.symbol}", {"reason": reason, "status": status})
        return ExecutionResult(False, status=status, reason=reason, raw=raw)

    async def _wait_for_fill(self, symbol: str, order_id: str, timeout_seconds: int) -> bool:
        deadline = datetime.now(UTC) + timedelta(seconds=timeout_seconds)
        while datetime.now(UTC) < deadline:
            order = await self.client.get_order(symbol, order_id)
            if order and order.get("status") == "FILLED":
                return True
            await asyncio.sleep(3)
        return False

    # -------------------------------------------------------------- protection

    async def _place_protection(
        self, plan: TradePlan, quantity: Decimal, stop_price: Decimal, target_price: Decimal
    ) -> OrderResult:
        side = "SELL" if plan.direction == "LONG" else "BUY"
        filters = await self.client.get_symbol_filters(plan.symbol or "")
        if filters:
            stop_price = filters.round_price(stop_price)
            target_price = filters.round_price(target_price)

        # Futures brackets the position with two closePosition orders instead of
        # a spot OCO. closePosition sizes to the live position, so it needs no
        # quantity and stays correct as scale-outs shrink the position.
        if self.client.is_futures:
            return await self.client.place_futures_protection(
                symbol=plan.symbol or "",
                close_side=side,
                stop_price=_fmt(stop_price),
                target_price=_fmt(target_price),
            )

        if filters:
            quantity = filters.round_quantity(quantity)
            ok, reason = filters.validate_order(target_price, quantity)
            if not ok:
                return OrderResult(False, status="FILTER_REJECTED", reason=f"Filter violation: OCO {reason}")
        if quantity <= 0:
            return OrderResult(False, status="FILTER_REJECTED", reason="OCO quantity is zero")

        if plan.direction == "LONG":
            oco_kwargs = {
                "aboveType": "LIMIT_MAKER",
                "abovePrice": _fmt(target_price),
                "belowType": "STOP_LOSS_LIMIT",
                "belowStopPrice": _fmt(stop_price),
                "belowPrice": _fmt(stop_price),
                "belowTimeInForce": "GTC",
            }
        else:
            oco_kwargs = {
                "aboveType": "STOP_LOSS_LIMIT",
                "aboveStopPrice": _fmt(stop_price),
                "abovePrice": _fmt(stop_price),
                "aboveTimeInForce": "GTC",
                "belowType": "LIMIT_MAKER",
                "belowPrice": _fmt(target_price),
            }
        return await self.client.place_oco_order(symbol=plan.symbol, side=side, quantity=_fmt(quantity), **oco_kwargs)

    async def _replace_protection(self, managed: ManagedTrade, stop_price: Decimal, target_price: Decimal) -> None:
        symbol = managed.plan.symbol or ""
        await self._cancel_resting_orders(symbol)
        result = await self._place_protection(managed.plan, managed.remaining_quantity, stop_price, target_price)
        if result.accepted:
            managed.protective_stop = stop_price
            managed.protective_target = target_price
            return
        logger.warning(f"{symbol}: could not re-place protective OCO ({result.reason}); position is monitor-only")
        if self.journal:
            await self.journal.log_event(
                "PROTECTION_REPLACE_FAILED",
                "CRITICAL",
                f"{symbol} has no exchange-side stop after a protection update",
                {"symbol": symbol, "reason": result.reason},
            )

    async def _cancel_resting_orders(self, symbol: str) -> None:
        try:
            # One bulk cancel on futures; per-order on spot (cancel_all_orders
            # falls back to a loop there).
            await self.client.cancel_all_orders(symbol)
        except Exception as exc:
            logger.warning(f"{symbol}: could not cancel resting orders: {exc}")

    # ----------------------------------------------------------------- monitor

    def start_monitoring(self) -> None:
        if self._monitor_task and not self._monitor_task.done():
            return
        self._monitor_task = asyncio.create_task(self.monitor_open_trades())

    async def monitor_open_trades(self) -> None:
        while True:
            try:
                await self._monitor_once()
                await asyncio.sleep(float(settings.monitor_interval_seconds))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"Execution monitor error: {exc}")
                await asyncio.sleep(float(settings.monitor_interval_seconds))

    async def _monitor_once(self) -> None:
        async with self.lock:
            for symbol, managed in list(self.open_trades.items()):
                plan = managed.plan
                direction = plan.direction or ""
                price = await self._latest_price(symbol)
                if price <= 0:
                    continue

                context = self.market_context.get(symbol, {}) or plan.indicator_state.get("latest", {})
                atr_value = _decimal(context.get("atr_14"), Decimal("0"))
                risk = plan.initial_risk_per_unit or abs(plan.entry_price - plan.sl_price)
                current_r = self._r_multiple(plan, price)
                managed.peak_r = max(managed.peak_r, current_r)

                # Peak-profit guard: once the trade has been at least
                # PEAK_GUARD_ARM_R in front, never let it hand back more than
                # PEAK_GUARD_GIVEBACK_PCT of that best gain. Checked before any
                # other exit so a reversing winner is banked at market instantly.
                if self._peak_guard_triggered(managed, current_r):
                    await self._finalise(symbol, managed, price, "PEAK_GUARD")
                    continue

                new_stop = self.risk_manager.update_protective_stop(symbol, price, atr_value)
                if new_stop and self._stop_moved(managed, new_stop, price):
                    await self._replace_protection(managed, new_stop, managed.protective_target or plan.final_target_price)

                stop_price = new_stop or managed.protective_stop or plan.sl_price
                if self._stop_hit(direction, price, stop_price):
                    reason = "PROFIT_LOCK" if current_r > 0 else ("TRAIL" if managed.peak_r > 0 else "SL")
                    await self._finalise(symbol, managed, price, reason)
                    continue

                if await self._take_scale_outs(symbol, managed, price, risk):
                    continue

                if self._reversal_detected(direction, context, current_r):
                    await self._finalise(symbol, managed, price, "REVERSAL")
                    continue

                if self._target_hit(direction, price, managed.protective_target or plan.final_target_price):
                    await self._finalise(symbol, managed, price, "TP_FINAL")
                    continue

                age = datetime.now(UTC) - managed.opened_at
                if age >= timedelta(hours=float(settings.time_stop_hours)):
                    moved = abs(price - plan.entry_price) >= atr_value if atr_value > 0 else True
                    if not moved:
                        await self._finalise(symbol, managed, price, "TIME_STOP")

    def _peak_guard_triggered(self, managed: ManagedTrade, current_r: Decimal) -> bool:
        """True once a trade that reached a real peak gives back too much of it.

        Measured in R (profit units), so 'give back 5% of the peak' means the
        current profit dropped below 95% of the best profit the trade ever
        showed. Only arms after peak_r >= arm_r so ordinary noise around
        breakeven never trips it.
        """
        if not settings.peak_guard_enabled:
            return False
        arm_r = Decimal(str(settings.peak_guard_arm_r))
        if managed.peak_r < arm_r or managed.peak_r <= 0:
            return False
        giveback = Decimal(str(settings.peak_guard_giveback_pct)) / Decimal("100")
        floor_r = managed.peak_r * (Decimal("1") - giveback)
        return current_r < floor_r

    def _stop_moved(self, managed: ManagedTrade, new_stop: Decimal, price: Decimal) -> bool:
        """Only rewrite the exchange order when the move is worth an API call."""
        current = managed.protective_stop
        if current <= 0:
            return True
        if price <= 0:
            return False
        return abs(new_stop - current) / price > Decimal("0.0015")

    async def _take_scale_outs(self, symbol: str, managed: ManagedTrade, price: Decimal, risk: Decimal) -> bool:
        """Bank part of the position at 1R and 2R, leaving a runner.

        Combined with the profit lock this is what makes 'a winner cannot become
        a loser' true in practice rather than just on the stop chart: by the time
        TP1 fills, realised profit already exceeds the remaining risk.
        """
        plan = managed.plan
        direction = plan.direction or ""
        ladder = (
            ("TP1", plan.tp1_price, Decimal(str(settings.scale_out_tp1_fraction))),
            ("TP2", plan.tp2_price or Decimal("0"), Decimal(str(settings.scale_out_tp2_fraction))),
        )
        for label, level, fraction in ladder:
            if label in managed.scaled_out or level <= 0 or fraction <= 0:
                continue
            if not self._target_hit(direction, price, level):
                continue
            close_qty = (plan.quantity * fraction).quantize(Decimal("0.00000001"))
            close_qty = min(close_qty, managed.remaining_quantity)
            if close_qty <= 0:
                managed.scaled_out.add(label)
                continue

            await self._cancel_resting_orders(symbol)
            result = await self._close_quantity(symbol, direction, close_qty)
            managed.scaled_out.add(label)
            if not result.accepted:
                logger.warning(f"{symbol}: {label} scale-out rejected ({result.reason})")
                await self._replace_protection(managed, managed.protective_stop, managed.protective_target)
                continue

            managed.remaining_quantity -= close_qty
            position = self.risk_manager.open_positions.get(symbol)
            if position:
                position.quantity = managed.remaining_quantity
            partial_pnl = self._pnl(plan, price, close_qty)
            partial_r = self._r_multiple(plan, price)
            self.risk_manager.register_closed_trade(symbol, partial_pnl, partial_r, remove_position=False)
            if self.journal:
                await self.journal.log_partial_exit(symbol, price, close_qty, partial_pnl, partial_r, label)
            await self.alerter.send(
                f"PARTIAL EXIT {label}: {symbol}",
                {"price": str(price), "closed": str(close_qty), "pnl": str(partial_pnl), "r": f"{partial_r:.2f}"},
            )

            if managed.remaining_quantity <= 0:
                self.open_trades.pop(symbol, None)
                self.risk_manager.open_positions.pop(symbol, None)
                if self.journal:
                    await self.journal.log_trade_exit(symbol, price, label)
                return True
            await self._replace_protection(
                managed,
                self.risk_manager.update_protective_stop(symbol, price, _decimal(self.market_context.get(symbol, {}).get("atr_14"), Decimal("0")))
                or managed.protective_stop,
                managed.protective_target or plan.final_target_price,
            )
        return False

    def _reversal_detected(self, direction: str, context: dict[str, Any], current_r: Decimal) -> bool:
        """Momentum has rolled over while the trade is in front: take the money.

        Deliberately requires the trade to already be profitable. Applied to a
        losing position this would just be a discretionary early stop, which the
        stop loss already handles.
        """
        if not settings.reversal_exit_enabled:
            return False
        if current_r < Decimal(str(settings.reversal_exit_min_r)):
            return False
        close = _float(context.get("close"))
        ema_9 = _float(context.get("ema_9"))
        macd_hist = _float(context.get("macd_hist"))
        if close <= 0 or ema_9 <= 0:
            return False
        if direction == "LONG":
            return close < ema_9 and macd_hist < 0
        return close > ema_9 and macd_hist > 0

    async def _finalise(self, symbol: str, managed: ManagedTrade, price: Decimal, reason: str) -> None:
        plan = managed.plan
        await self._cancel_resting_orders(symbol)
        await self._close_quantity(symbol, plan.direction or "", managed.remaining_quantity)
        pnl = self._pnl(plan, price, managed.remaining_quantity)
        r_multiple = self._r_multiple(plan, price)
        self.open_trades.pop(symbol, None)
        self.risk_manager.register_closed_trade(symbol, pnl, r_multiple, bar_seconds=_bar_seconds())
        if self.journal:
            await self.journal.log_trade_exit(symbol, price, reason)
        await self.alerter.send(
            f"{'WIN' if pnl >= 0 else 'LOSS'}: {symbol} closed ({reason})",
            {"exit_price": str(price), "reason": reason, "pnl": str(pnl), "r": f"{r_multiple:.2f}"},
        )

    # -------------------------------------------------------------- primitives

    async def _latest_price(self, symbol: str) -> Decimal:
        candles = await self.client.get_ohlcv(symbol, "1m", 1)
        if not candles:
            return Decimal("0")
        return candles[-1].close

    async def _close_quantity(self, symbol: str, direction: str, quantity: Decimal) -> ExecutionResult:
        if quantity <= 0:
            return ExecutionResult(False, reason="quantity <= 0")
        side = "SELL" if direction == "LONG" else "BUY"
        if self.client.is_futures:
            # Closing a long sells, closing a short buys -- reduceOnly guarantees
            # the order can only shrink the position, never flip it. No
            # base-asset balance to check: the position itself is the collateral.
            filters = await self.client.get_symbol_filters(symbol)
            if filters:
                quantity = filters.round_quantity(quantity)
                if filters.min_qty > 0 and quantity < filters.min_qty:
                    return ExecutionResult(False, status="FILTER_REJECTED", reason=f"quantity {quantity} below minQty {filters.min_qty}")
            if quantity <= 0:
                return ExecutionResult(False, reason="quantity rounds to zero")
            result = await self.client.place_order(
                symbol=symbol, side=side, type="MARKET", quantity=_fmt(quantity), reduceOnly="true"
            )
            return ExecutionResult(result.accepted, result.order_id, result.status, result.reason, result.raw)

        if side == "SELL":
            free = await self.client.get_asset_free(_base_asset(symbol))
            if free < quantity:
                quantity = free
        filters = await self.client.get_symbol_filters(symbol)
        if filters:
            quantity = filters.round_quantity(quantity)
            if filters.min_qty > 0 and quantity < filters.min_qty:
                return ExecutionResult(False, status="FILTER_REJECTED", reason=f"quantity {quantity} below minQty {filters.min_qty}")
        if quantity <= 0:
            return ExecutionResult(False, reason="no free balance to close")
        result = await self.client.place_order(symbol=symbol, side=side, type="MARKET", quantity=_fmt(quantity))
        return ExecutionResult(result.accepted, result.order_id, result.status, result.reason, result.raw)

    def _pnl(self, plan: TradePlan, exit_price: Decimal, quantity: Decimal) -> Decimal:
        if plan.direction == "LONG":
            return (exit_price - plan.entry_price) * quantity
        return (plan.entry_price - exit_price) * quantity

    def _r_multiple(self, plan: TradePlan, exit_price: Decimal) -> Decimal:
        risk = plan.initial_risk_per_unit or abs(plan.entry_price - plan.sl_price)
        if risk == 0:
            return Decimal("0")
        if plan.direction == "LONG":
            return (exit_price - plan.entry_price) / risk
        return (plan.entry_price - exit_price) / risk

    def _target_hit(self, direction: str, price: Decimal, target: Decimal) -> bool:
        if target <= 0:
            return False
        return price >= target if direction == "LONG" else price <= target

    def _stop_hit(self, direction: str, price: Decimal, stop: Decimal) -> bool:
        if stop <= 0:
            return False
        return price <= stop if direction == "LONG" else price >= stop

    # ----------------------------------------------------------------- manual

    async def emergency_close_all(self) -> list[ExecutionResult]:
        async with self.lock:
            results = []
            for symbol, managed in list(self.open_trades.items()):
                price = await self._latest_price(symbol)
                results.append(await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity))
                exit_price = price if price > 0 else managed.plan.entry_price
                self.open_trades.pop(symbol, None)
                self.risk_manager.register_closed_trade(
                    symbol, self._pnl(managed.plan, exit_price, managed.remaining_quantity), self._r_multiple(managed.plan, exit_price)
                )
                if self.journal:
                    await self.journal.log_trade_exit(symbol, exit_price, "MANUAL")
                await self.alerter.send(f"MANUAL CLOSE: {symbol}", {"reason": "MANUAL", "exit_price": str(exit_price)})
            return results

    async def emergency_close_symbol(self, symbol: str) -> ExecutionResult:
        async with self.lock:
            managed = self.open_trades.get(symbol)
            price = await self._latest_price(symbol)
            await self._cancel_resting_orders(symbol)

            if not managed:
                # No local record -- typically a position opened before a
                # restart, managed only by its exchange stop. The Stop button
                # must still work, so flatten whatever the account actually
                # holds on the exchange rather than reporting "not managed".
                return await self._force_flatten_untracked(symbol, price)

            result = await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity)
            if result.accepted:
                exit_price = price if price > 0 else managed.plan.entry_price
                self.open_trades.pop(symbol, None)
                self.risk_manager.register_closed_trade(
                    symbol, self._pnl(managed.plan, exit_price, managed.remaining_quantity), self._r_multiple(managed.plan, exit_price)
                )
                if self.journal:
                    await self.journal.log_trade_exit(symbol, exit_price, "MANUAL")
                await self.alerter.send(f"MANUAL CLOSE: {symbol}", {"reason": "MANUAL", "exit_price": str(exit_price)})
            return result

    async def _force_flatten_untracked(self, symbol: str, price: Decimal) -> ExecutionResult:
        """Sell whatever base asset the account holds for an untracked symbol.

        Spot-only: a long position is simply a holding of the base asset, so
        market-selling the free balance flattens it. Guarantees the Stop button
        does something even for positions the bot never had in memory.
        """
        base = _base_asset(symbol)
        free = await self.client.get_asset_free(base)
        if free <= 0:
            logger.info(f"Manual stop for {symbol}: nothing held on the exchange to close")
            return ExecutionResult(False, reason=f"no {base} balance to close")
        result = await self._close_quantity(symbol, "LONG", free)
        if result.accepted and self.journal:
            exit_price = price if price > 0 else Decimal("0")
            await self.journal.log_event(
                "MANUAL_CLOSE",
                "INFO",
                f"Force-flattened untracked {symbol} ({free} {base}) on manual stop",
                {"symbol": symbol, "qty": str(free), "approx_price": str(exit_price)},
            )
            await self.alerter.send(f"MANUAL CLOSE (untracked): {symbol}", {"qty": str(free), "base": base})
        return result

    # -------------------------------------------------------------- reconcile

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
            for symbol in internal_symbols - remote_symbols:
                managed = self.open_trades.get(symbol)
                if not managed:
                    continue
                try:
                    price = await self._latest_price(symbol)
                    if price <= 0:
                        price = managed.plan.entry_price
                    reason = self._infer_exit_reason(managed.plan, price)
                    pnl = self._pnl(managed.plan, price, managed.remaining_quantity)
                    r_multiple = self._r_multiple(managed.plan, price)
                    self.open_trades.pop(symbol, None)
                    self.risk_manager.register_closed_trade(symbol, pnl, r_multiple, bar_seconds=_bar_seconds())
                    if self.journal:
                        await self.journal.log_trade_exit(symbol, price, reason)
                        await self.journal.log_event(
                            "RECONCILE_CLOSE",
                            "WARNING",
                            f"{symbol} position found closed on Binance; cleaned up local state as {reason}",
                            {"symbol": symbol, "approx_exit": str(price), "reason": reason},
                        )
                    logger.warning(f"Reconcile: {symbol} closed on Binance ({reason}); local state cleaned up")
                except Exception as exc:
                    logger.warning(f"Reconcile cleanup failed for {symbol}: {exc}")
        except Exception as exc:
            logger.exception(f"Order reconciliation failed: {exc}")

    def _infer_exit_reason(self, plan: TradePlan, exit_price: Decimal) -> str:
        """Name the exit by where price actually is relative to the plan.

        Every remotely-closed position used to be journalled as "SL", including
        take-profit fills. That produced a stop-loss bucket averaging -0.23R --
        arithmetically impossible for a real stop, and it made the exit
        distribution unreadable, which is the one statistic the exit logic has
        to be tuned against.
        """
        r_multiple = self._r_multiple(plan, exit_price)
        if r_multiple >= Decimal("0.9"):
            return "TP"
        if r_multiple > 0:
            return "PROFIT_LOCK"
        if r_multiple <= Decimal("-0.9"):
            return "SL"
        return "TRAIL"


def _protective_target(plan: TradePlan) -> Decimal:
    """The price the exchange-side OCO aims at: the runner's exit, not TP1."""
    for level in (plan.final_target_price, plan.tp2_price, plan.tp1_price):
        if level and level > 0:
            return level
    return Decimal("0")


def _bar_seconds() -> int:
    return _TIMEFRAME_SECONDS.get(settings.primary_timeframe, 3600)


_TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _base_asset(symbol: str) -> str:
    for quote in ("USDT", "BUSD", "USDC", "BTC", "ETH"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol


def _decimal(value: Any, default: Decimal) -> Decimal:
    try:
        if value is None:
            return default
        return Decimal(str(value))
    except Exception:
        return default


def _float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")
