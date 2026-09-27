from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from config import Settings, settings
from trading.strategy_engine import TradeSignal
from utils.logger import logger


@dataclass(frozen=True)
class ChecklistItem:
    name: str
    passed: bool
    detail: str


@dataclass
class ClosedTradeStats:
    pnl_usd: Decimal
    r_multiple: Decimal
    closed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    symbol: str | None = None


@dataclass
class OpenPosition:
    symbol: str
    direction: str
    entry_price: Decimal
    quantity: Decimal
    sl_price: Decimal
    opened_at: datetime
    risk_pct: Decimal
    # Distance from entry to the *original* stop. Every R figure downstream is
    # measured against this, so it must not drift as the stop ratchets up.
    initial_risk_per_unit: Decimal = Decimal("0")
    highest_price: Decimal | None = None
    lowest_price: Decimal | None = None
    profit_locked: bool = False


@dataclass(frozen=True)
class TradeCandidate:
    signal: TradeSignal
    entry_price: Decimal | float | str
    atr: Decimal | float | str
    account_balance: Decimal | float | str
    available_balance: Decimal | float | str
    next_support: Decimal | float | str | None = None
    next_resistance: Decimal | float | str | None = None
    conviction: float = 0.0


@dataclass(frozen=True)
class TradePlan:
    approved: bool
    reason: str
    checklist: list[ChecklistItem]
    symbol: str | None = None
    direction: str | None = None
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    sl_price: Decimal = Decimal("0")
    tp1_price: Decimal = Decimal("0")
    tp2_price: Decimal | None = None
    final_target_price: Decimal = Decimal("0")
    initial_risk_per_unit: Decimal = Decimal("0")
    strategy_used: str | None = None
    regime_at_entry: str | None = None
    lstm_confidence: float = 0.0
    rl_confidence: float = 0.0
    confluence_score: float = 0.0
    conviction: float = 0.0
    indicator_state: dict[str, Any] = field(default_factory=dict)
    risk_pct: Decimal = Decimal("0")
    reward_risk: Decimal = Decimal("0")


class RiskManager:
    def __init__(self, cfg: Settings = settings) -> None:
        self.settings = cfg
        self.lock = asyncio.Lock()
        self.open_positions: dict[str, OpenPosition] = {}
        self.closed_trades: list[ClosedTradeStats] = []
        self.peak_equity = Decimal("0")
        self.current_equity = Decimal("0")
        self.daily_realized: dict[date, Decimal] = {}
        self.weekly_realized: dict[tuple[int, int], Decimal] = {}
        self.circuit_breaker_active = False
        self.cooldown_until: dict[str, datetime] = {}
        self._override_max_risk = float(cfg.max_risk_per_trade_pct)
        self._override_max_daily = float(cfg.max_daily_loss_pct)
        self._override_max_weekly = float(cfg.max_weekly_loss_pct)
        self._override_max_trades = int(cfg.max_concurrent_trades)

    # ------------------------------------------------------------------- plan

    async def calculate(self, candidate: TradeCandidate) -> TradePlan:
        async with self.lock:
            try:
                return self._calculate_locked(candidate)
            except Exception as exc:
                logger.exception(f"Risk manager failed closed: {exc}")
                return TradePlan(False, f"risk manager error: {exc}", [])

    def _calculate_locked(self, candidate: TradeCandidate) -> TradePlan:
        self._load_overrides()
        signal = candidate.signal
        entry = _d(candidate.entry_price)
        atr_value = _d(candidate.atr)
        balance = _d(candidate.account_balance)
        available = _d(candidate.available_balance)
        conviction = max(0.0, min(1.0, float(candidate.conviction)))
        checklist: list[ChecklistItem] = []

        self._update_equity(balance)
        checklist.append(ChecklistItem("Confidence gate passed", signal.direction in {"LONG", "SHORT"}, signal.direction))
        checklist.append(ChecklistItem("No circuit breaker active", not self.circuit_breaker_active, str(self.circuit_breaker_active)))
        checklist.append(
            ChecklistItem("Daily loss limit not hit", not self.daily_loss_limit_hit(balance), f"{self._today_realized_pct(balance):.4f}%")
        )
        checklist.append(
            ChecklistItem("Weekly loss limit not hit", not self.weekly_loss_limit_hit(balance), f"{self._week_realized_pct(balance):.4f}%")
        )
        checklist.append(
            ChecklistItem(
                "Max concurrent trades not exceeded",
                len(self.open_positions) < self._override_max_trades,
                f"{len(self.open_positions)}/{self._override_max_trades}",
            )
        )

        if signal.direction not in {"LONG", "SHORT"}:
            return self._blocked("confidence gate failed", checklist)
        if self.circuit_breaker_active:
            return self._blocked("drawdown circuit breaker active", checklist)
        if self.daily_loss_limit_hit(balance):
            return self._blocked("daily loss limit hit", checklist)
        if self.weekly_loss_limit_hit(balance):
            return self._blocked("weekly loss limit hit", checklist)
        if len(self.open_positions) >= self._override_max_trades:
            return self._blocked("max concurrent open trades exceeded", checklist)
        if entry <= 0 or atr_value <= 0 or balance <= 0:
            checklist.append(ChecklistItem("Position size calculated and within limits", False, "invalid entry/ATR/balance"))
            return self._blocked("invalid price, ATR, or balance", checklist)

        atr_risk = _d(self.settings.stop_atr_multiple) * atr_value
        if atr_risk <= 0:
            return self._blocked("invalid stop distance", checklist)

        long = signal.direction == "LONG"
        sign = Decimal("1") if long else Decimal("-1")

        # Structure-aware stop: anchor the SL just beyond the nearest support
        # (long) / resistance (short) so only a genuine level-break stops the
        # trade, not ordinary noise. Fall back to the ATR stop, and only adopt a
        # structural stop that sits a sane distance away (0.5x-2x the ATR stop) so
        # it is neither hair-trigger tight nor recklessly wide. Risk-per-unit is
        # then whatever the actual stop distance is, and TP levels scale from it.
        sl_price = entry - sign * atr_risk
        stop_level = _optional_decimal(candidate.next_support if long else candidate.next_resistance)
        if stop_level is not None:
            buffer = Decimal("0.25") * atr_value
            struct_sl = stop_level - sign * buffer
            struct_risk = abs(entry - struct_sl)
            on_loss_side = struct_sl < entry if long else struct_sl > entry
            if on_loss_side and Decimal("0.5") * atr_risk <= struct_risk <= Decimal("2.0") * atr_risk:
                sl_price = struct_sl

        risk_per_unit = abs(entry - sl_price)
        # Floor the stop distance: in very low-volatility conditions the ATR /
        # structural stop can land a fraction of a percent from entry, which gets
        # picked off by ordinary noise and makes the dollar risk (and the trade)
        # trivially small. Never place the stop closer than STOP_MIN_PCT of price.
        min_dist = entry * _d(self.settings.stop_min_pct) / Decimal("100")
        if min_dist > 0 and risk_per_unit < min_dist:
            risk_per_unit = min_dist
            sl_price = entry - sign * min_dist
        if risk_per_unit <= 0:
            return self._blocked("invalid stop distance", checklist)
        tp1_price = entry + sign * _d(self.settings.tp1_r_multiple) * risk_per_unit
        tp2_price = entry + sign * _d(self.settings.tp2_r_multiple) * risk_per_unit
        final_target = entry + sign * _d(self.settings.final_target_r_multiple) * risk_per_unit

        # Structure caps the runner: if the next resistance (or support) sits
        # inside the arithmetic target, that level is the realistic exit.
        structural = _optional_decimal(candidate.next_resistance if long else candidate.next_support)
        if structural is not None:
            beyond_tp2 = structural > tp2_price if long else structural < tp2_price
            inside_target = structural < final_target if long else structural > final_target
            if beyond_tp2 and inside_target:
                final_target = structural

        reward_risk = abs(final_target - entry) / risk_per_unit
        min_rr = _d(self.settings.min_reward_risk)
        checklist.append(ChecklistItem("Stop loss set and valid", sl_price > 0, str(sl_price)))
        checklist.append(ChecklistItem(f"Final target at least {min_rr:.1f}R", reward_risk >= min_rr, f"{reward_risk:.2f}R"))
        if sl_price <= 0:
            return self._blocked("invalid stop loss", checklist)
        if reward_risk < min_rr:
            return self._blocked(f"final target below {min_rr:.1f}R", checklist)

        risk_pct = self._risk_pct_for_conviction(conviction)
        quantity = self._position_size(balance, entry, risk_per_unit, risk_pct)
        if signal.regime_at_entry == "HIGH_VOLATILITY":
            quantity *= Decimal("0.5")
        notional = quantity * entry
        trade_risk_pct = (quantity * risk_per_unit / balance) * Decimal("100") if balance else Decimal("0")
        total_risk_pct = self._open_risk_pct() + trade_risk_pct

        checklist.append(
            ChecklistItem(
                "Position size calculated and within limits",
                quantity > 0 and trade_risk_pct <= _d(self._override_max_risk) + Decimal("0.0001"),
                f"qty={quantity}, risk={trade_risk_pct:.4f}%, conviction={conviction:.2f}",
            )
        )
        checklist.append(
            ChecklistItem(
                "Max portfolio risk not exceeded",
                total_risk_pct <= _d(self.settings.max_portfolio_risk_pct),
                f"{total_risk_pct:.4f}%",
            )
        )
        # Spot must cover the whole notional; futures only the margin
        # (notional / leverage). Using the spot rule on futures would reject
        # every leveraged trade whose notional exceeds the wallet.
        if getattr(self.settings, "is_futures", False):
            leverage = _d(max(1, int(getattr(self.settings, "futures_leverage", 1))))
            capital_needed = notional / leverage
            capital_label = f"margin={capital_needed:.2f} at {leverage}x"
        else:
            capital_needed = notional
            capital_label = f"notional={notional}"
        checklist.append(ChecklistItem("Binance account has sufficient balance", capital_needed <= available, capital_label))

        failed = [item for item in checklist if not item.passed]
        if failed:
            return self._blocked(failed[0].name, checklist)

        return TradePlan(
            approved=True,
            reason="approved",
            checklist=checklist,
            symbol=signal.symbol,
            direction=signal.direction,
            quantity=quantity,
            entry_price=entry,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            final_target_price=final_target,
            initial_risk_per_unit=risk_per_unit,
            strategy_used=signal.strategy_used,
            regime_at_entry=signal.regime_at_entry,
            lstm_confidence=signal.lstm_confidence,
            rl_confidence=signal.rl_confidence,
            confluence_score=signal.confluence_score,
            conviction=conviction,
            indicator_state=signal.indicator_state,
            risk_pct=trade_risk_pct,
            reward_risk=reward_risk,
        )

    # ------------------------------------------------------------------ sizing

    def _risk_pct_for_conviction(self, conviction: float) -> Decimal:
        """Map conviction onto the configured risk band.

        A marginal setup that barely cleared the gate and an overwhelming one
        used to stake exactly the same fraction of the account. The curve is
        convex (gamma > 1) so size grows slowly through the middle of the range
        and only approaches the maximum when the evidence genuinely is strong.
        """
        low = _d(self.settings.risk_min_pct)
        high = _d(self.settings.risk_max_pct)
        if high < low:
            low, high = high, low
        gamma = max(0.1, float(self.settings.risk_curve_gamma))
        shaped = Decimal(str(round(max(0.0, min(1.0, conviction)) ** gamma, 8)))
        scaled = low + (high - low) * shaped
        return min(scaled, _d(self._override_max_risk))

    def _position_size(self, balance: Decimal, entry: Decimal, risk_per_unit: Decimal, risk_pct: Decimal) -> Decimal:
        if risk_per_unit <= 0 or entry <= 0:
            return Decimal("0")
        risk_fraction = risk_pct / Decimal("100")

        kelly_fraction = self._half_kelly_risk_fraction()
        if kelly_fraction is not None:
            # Both quantities are "fraction of equity risked per trade", so the
            # comparison is dimensionally sound. The previous implementation
            # compared a risk-based size against a notional-allocation size,
            # which are different units and produced arbitrary results.
            risk_fraction = min(risk_fraction, kelly_fraction)

        by_risk = (balance * risk_fraction) / risk_per_unit
        by_notional_cap = (balance * _d(self.settings.max_position_pct) / Decimal("100")) / entry
        return max(Decimal("0"), min(by_risk, by_notional_cap)).quantize(Decimal("0.00000001"))

    def _half_kelly_risk_fraction(self) -> Decimal | None:
        """Half-Kelly expressed as a fraction of equity to risk per trade."""
        trades = [t for t in self.closed_trades[-50:] if t.r_multiple is not None]
        if len(trades) < 20:
            return None
        wins = [t.r_multiple for t in trades if t.r_multiple > 0]
        losses = [-t.r_multiple for t in trades if t.r_multiple < 0]
        if not wins or not losses:
            return None
        win_rate = Decimal(len(wins)) / Decimal(len(trades))
        avg_win = sum(wins, Decimal("0")) / Decimal(len(wins))
        avg_loss = sum(losses, Decimal("0")) / Decimal(len(losses))
        if avg_win <= 0 or avg_loss <= 0:
            return None
        payoff = avg_win / avg_loss
        kelly = (win_rate * payoff - (Decimal("1") - win_rate)) / payoff
        return max(Decimal("0"), kelly / Decimal("2"))

    # ----------------------------------------------------------- position life

    def register_open_position(self, plan: TradePlan) -> None:
        if not plan.approved or not plan.symbol or not plan.direction:
            return
        self.open_positions[plan.symbol] = OpenPosition(
            symbol=plan.symbol,
            direction=plan.direction,
            entry_price=plan.entry_price,
            quantity=plan.quantity,
            sl_price=plan.sl_price,
            opened_at=datetime.now(UTC),
            risk_pct=plan.risk_pct,
            initial_risk_per_unit=plan.initial_risk_per_unit or abs(plan.entry_price - plan.sl_price),
            highest_price=plan.entry_price,
            lowest_price=plan.entry_price,
        )

    def register_closed_trade(
        self,
        symbol: str,
        pnl_usd: Decimal | float | str,
        r_multiple: Decimal | float | str,
        remove_position: bool = True,
        bar_seconds: int | None = None,
    ) -> None:
        pnl = _d(pnl_usd)
        r_value = _d(r_multiple)
        closed_at = datetime.now(UTC)
        if remove_position:
            self.open_positions.pop(symbol, None)
        self.closed_trades.append(ClosedTradeStats(pnl, r_value, closed_at, symbol))
        self.daily_realized[closed_at.date()] = self.daily_realized.get(closed_at.date(), Decimal("0")) + pnl
        week_key = closed_at.isocalendar()[:2]
        self.weekly_realized[week_key] = self.weekly_realized.get(week_key, Decimal("0")) + pnl

        # Re-entering a symbol seconds after being stopped out of it is how the
        # account paid the spread repeatedly on the same failing idea. Losses
        # start a cooldown; wins do not, so a working setup can be re-taken.
        if pnl < 0 and self.settings.loss_cooldown_bars > 0:
            seconds = bar_seconds or 3600
            self.cooldown_until[symbol] = closed_at + timedelta(seconds=seconds * self.settings.loss_cooldown_bars)

    def cooldown_reason(self, symbol: str) -> str | None:
        until = self.cooldown_until.get(symbol)
        if not until:
            return None
        now = datetime.now(UTC)
        if now >= until:
            self.cooldown_until.pop(symbol, None)
            return None
        remaining = int((until - now).total_seconds() // 60)
        return f"cooling off after a loss on {symbol} for another {remaining} minute(s)"

    def update_protective_stop(
        self,
        symbol: str,
        current_price: Decimal | float | str,
        atr_value: Decimal | float | str,
    ) -> Decimal | None:
        """Ratchet the stop forward. It never moves backwards.

        Three rules stack, and the tightest of them wins:

        * once the trade has been `profit_lock_arm_r` in front, the stop sits at
          entry plus the round-trip cost, so the position can no longer end red;
        * past `trail_arm_r`, the stop keeps all but `profit_lock_give_back` of
          the best excursion so far;
        * an ATR chandelier off the extreme, for when volatility collapses and
          the R-based rule would leave too much room.
        """
        position = self.open_positions.get(symbol)
        if not position:
            return None
        price = _d(current_price)
        atr_d = _d(atr_value)
        risk = position.initial_risk_per_unit or abs(position.entry_price - position.sl_price)
        if risk <= 0 or price <= 0:
            return position.sl_price

        long = position.direction == "LONG"
        if long:
            position.highest_price = max(position.highest_price or price, price)
            extreme = position.highest_price
            mfe_r = (extreme - position.entry_price) / risk
        else:
            position.lowest_price = min(position.lowest_price or price, price)
            extreme = position.lowest_price
            mfe_r = (position.entry_price - extreme) / risk

        candidates: list[Decimal] = [position.sl_price]
        sign = Decimal("1") if long else Decimal("-1")

        arm_r = _d(self.settings.profit_lock_arm_r)
        if mfe_r >= arm_r:
            cost_buffer = position.entry_price * _d(self.settings.taker_fee_rate) * Decimal("2")
            candidates.append(position.entry_price + sign * cost_buffer)
            position.profit_locked = True

        trail_arm = _d(self.settings.trail_arm_r)
        if mfe_r >= trail_arm:
            give_back = _d(self.settings.profit_lock_give_back)
            locked_r = max(Decimal("0"), mfe_r - give_back)
            candidates.append(position.entry_price + sign * locked_r * risk)
            if atr_d > 0:
                chandelier = extreme - sign * _d(self.settings.trail_atr_multiple) * atr_d
                candidates.append(chandelier)

        position.sl_price = max(candidates) if long else min(candidates)
        return position.sl_price

    def unrealized_r(self, symbol: str, current_price: Decimal | float | str) -> Decimal:
        position = self.open_positions.get(symbol)
        if not position:
            return Decimal("0")
        risk = position.initial_risk_per_unit or abs(position.entry_price - position.sl_price)
        if risk <= 0:
            return Decimal("0")
        price = _d(current_price)
        if position.direction == "LONG":
            return (price - position.entry_price) / risk
        return (position.entry_price - price) / risk

    # ------------------------------------------------------------------ limits

    def daily_loss_limit_hit(self, balance: Decimal) -> bool:
        return self._today_realized_pct(balance) <= -self._override_max_daily

    def weekly_loss_limit_hit(self, balance: Decimal) -> bool:
        return self._week_realized_pct(balance) <= -self._override_max_weekly

    def circuit_breaker_hit(self, equity: Decimal | float | str) -> bool:
        current = _d(equity)
        self._update_equity(current)
        return self.circuit_breaker_active

    def _today_realized_pct(self, balance: Decimal) -> float:
        pnl = self.daily_realized.get(datetime.now(UTC).date(), Decimal("0"))
        return float((pnl / balance) * Decimal("100")) if balance > 0 else 0.0

    def _week_realized_pct(self, balance: Decimal) -> float:
        week_key = datetime.now(UTC).isocalendar()[:2]
        pnl = self.weekly_realized.get(week_key, Decimal("0"))
        return float((pnl / balance) * Decimal("100")) if balance > 0 else 0.0

    def _open_risk_pct(self) -> Decimal:
        return sum((position.risk_pct for position in self.open_positions.values()), Decimal("0"))

    def _update_equity(self, equity: Decimal) -> None:
        if equity <= 0:
            return
        self.current_equity = equity
        if self.peak_equity <= 0:
            self.peak_equity = equity
        else:
            self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity <= 0:
            return
        # Pure TP/SL mode disables the drawdown circuit breaker entirely: it must
        # neither close trades nor block new entries. Keep it disarmed.
        if self.settings.pure_tp_sl:
            self.circuit_breaker_active = False
            return
        drawdown_pct = (self.peak_equity - equity) / self.peak_equity * Decimal("100")
        if drawdown_pct >= _d(self.settings.drawdown_circuit_breaker_pct):
            if not self.circuit_breaker_active:
                logger.critical(f"Circuit breaker armed at {drawdown_pct:.2f}% drawdown")
            self.circuit_breaker_active = True
        elif self.circuit_breaker_active and drawdown_pct <= _d(self.settings.circuit_breaker_reset_pct):
            logger.warning(f"Circuit breaker cleared; drawdown recovered to {drawdown_pct:.2f}%")
            self.circuit_breaker_active = False

    def _blocked(self, reason: str, checklist: list[ChecklistItem]) -> TradePlan:
        return TradePlan(False, reason, checklist)

    def _load_overrides(self) -> None:
        override_path = self.settings.runtime_dir / "risk_overrides.json"
        if not override_path.exists():
            self._override_max_risk = float(self.settings.max_risk_per_trade_pct)
            self._override_max_daily = float(self.settings.max_daily_loss_pct)
            self._override_max_weekly = float(self.settings.max_weekly_loss_pct)
            self._override_max_trades = int(self.settings.max_concurrent_trades)
            return
        try:
            overrides = json.loads(override_path.read_text(encoding="utf-8"))
            self._override_max_risk = float(overrides.get("max_risk_per_trade_pct", self.settings.max_risk_per_trade_pct))
            self._override_max_daily = float(overrides.get("max_daily_loss_pct", self.settings.max_daily_loss_pct))
            self._override_max_weekly = float(overrides.get("max_weekly_loss_pct", self.settings.max_weekly_loss_pct))
            self._override_max_trades = int(overrides.get("max_concurrent_trades", self.settings.max_concurrent_trades))
        except Exception as exc:
            logger.warning(f"Risk override load skipped: {exc}")


def _d(value: Decimal | float | str | int) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal value: {value}") from exc


def _optional_decimal(value: Decimal | float | str | None) -> Decimal | None:
    if value in (None, ""):
        return None
    return _d(value)
