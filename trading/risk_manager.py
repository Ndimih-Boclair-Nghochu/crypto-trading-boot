from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
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


@dataclass
class OpenPosition:
    symbol: str
    direction: str
    entry_price: Decimal
    quantity: Decimal
    sl_price: Decimal
    opened_at: datetime
    risk_pct: Decimal
    highest_price: Decimal | None = None
    lowest_price: Decimal | None = None


@dataclass(frozen=True)
class TradeCandidate:
    signal: TradeSignal
    entry_price: Decimal | float | str
    atr: Decimal | float | str
    account_balance: Decimal | float | str
    available_balance: Decimal | float | str
    next_support: Decimal | float | str | None = None
    next_resistance: Decimal | float | str | None = None


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
    strategy_used: str | None = None
    regime_at_entry: str | None = None
    lstm_confidence: float = 0.0
    rl_confidence: float = 0.0
    confluence_score: float = 0.0
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
        self._override_max_risk = float(cfg.max_risk_per_trade_pct)
        self._override_max_daily = float(cfg.max_daily_loss_pct)
        self._override_max_weekly = float(cfg.max_weekly_loss_pct)
        self._override_max_trades = int(cfg.max_concurrent_trades)
        self._override_confidence = float(cfg.confidence_threshold)

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
        checklist: list[ChecklistItem] = []

        self._update_equity(balance)
        checklist.append(ChecklistItem("Confidence gate passed", signal.direction in {"LONG", "SHORT"}, signal.direction))
        checklist.append(ChecklistItem("No circuit breaker active", not self.circuit_breaker_active, str(self.circuit_breaker_active)))
        checklist.append(
            ChecklistItem(
                "Daily loss limit not hit",
                not self.daily_loss_limit_hit(balance),
                f"{self._today_realized_pct(balance):.4f}%",
            )
        )
        checklist.append(
            ChecklistItem(
                "Weekly loss limit not hit",
                not self.weekly_loss_limit_hit(balance),
                f"{self._week_realized_pct(balance):.4f}%",
            )
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

        stop_distance = Decimal("1.5") * atr_value
        if signal.direction == "LONG":
            sl_price = entry - stop_distance
            tp_distance = max(Decimal("1.5") * atr_value, Decimal("2") * stop_distance)
            tp1_price = entry + tp_distance
            tp2_price = _optional_decimal(candidate.next_resistance) or entry + Decimal("3") * stop_distance
        else:
            sl_price = entry + stop_distance
            tp_distance = max(Decimal("1.5") * atr_value, Decimal("2") * stop_distance)
            tp1_price = entry - tp_distance
            tp2_price = _optional_decimal(candidate.next_support) or entry - Decimal("3") * stop_distance

        risk_per_unit = abs(entry - sl_price)
        reward_per_unit = abs(tp1_price - entry)
        reward_risk = reward_per_unit / risk_per_unit if risk_per_unit > 0 else Decimal("0")
        checklist.append(ChecklistItem("Stop loss set and valid", sl_price > 0, str(sl_price)))
        checklist.append(ChecklistItem("R:R >= 2:1", reward_risk >= Decimal("2"), str(reward_risk)))
        if sl_price <= 0:
            return self._blocked("invalid stop loss", checklist)
        if reward_risk < Decimal("2"):
            return self._blocked("reward:risk below 2:1", checklist)

        quantity = self._position_size(balance, entry, atr_value, risk_per_unit)
        if signal.regime_at_entry == "HIGH_VOLATILITY":
            quantity *= Decimal("0.5")
        notional = quantity * entry
        trade_risk_pct = (quantity * risk_per_unit / balance) * Decimal("100") if balance else Decimal("0")
        total_risk_pct = self._open_risk_pct() + trade_risk_pct

        checklist.append(
            ChecklistItem(
                "Position size calculated and within limits",
                quantity > 0 and trade_risk_pct <= _d(self._override_max_risk),
                f"qty={quantity}, risk={trade_risk_pct:.4f}%",
            )
        )
        checklist.append(
            ChecklistItem(
                "Max portfolio risk not exceeded",
                total_risk_pct <= _d(self.settings.max_portfolio_risk_pct),
                f"{total_risk_pct:.4f}%",
            )
        )
        checklist.append(ChecklistItem("Binance account has sufficient balance", notional <= available, f"notional={notional}"))

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
            strategy_used=signal.strategy_used,
            regime_at_entry=signal.regime_at_entry,
            lstm_confidence=signal.lstm_confidence,
            rl_confidence=signal.rl_confidence,
            confluence_score=signal.confluence_score,
            indicator_state=signal.indicator_state,
            risk_pct=trade_risk_pct,
            reward_risk=reward_risk,
        )

    def _position_size(self, balance: Decimal, entry: Decimal, atr_value: Decimal, risk_per_unit: Decimal) -> Decimal:
        risk_fraction = _d(self._override_max_risk) / Decimal("100")
        fixed_fractional = (balance * risk_fraction) / risk_per_unit
        atr_based = (balance * Decimal("0.01")) / (Decimal("2") * atr_value)
        half_kelly = self._half_kelly_fraction()
        if half_kelly is None:
            kelly_quantity = fixed_fractional
        elif half_kelly <= 0:
            kelly_quantity = Decimal("0")
        else:
            kelly_quantity = (balance * half_kelly) / entry
        return min(fixed_fractional, atr_based, kelly_quantity).quantize(Decimal("0.00000001"))

    def _half_kelly_fraction(self) -> Decimal | None:
        trades = self.closed_trades[-50:]
        if len(trades) < 10:
            return None
        wins = [t.pnl_usd for t in trades if t.pnl_usd > 0]
        losses = [-t.pnl_usd for t in trades if t.pnl_usd < 0]
        if not wins or not losses:
            return Decimal("0")
        win_rate = Decimal(len(wins)) / Decimal(len(trades))
        avg_win = sum(wins, Decimal("0")) / Decimal(len(wins))
        avg_loss = sum(losses, Decimal("0")) / Decimal(len(losses))
        if avg_win <= 0:
            return Decimal("0")
        kelly = (win_rate * avg_win - (Decimal("1") - win_rate) * avg_loss) / avg_win
        return max(Decimal("0"), kelly * Decimal("0.5"))

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
            highest_price=plan.entry_price,
            lowest_price=plan.entry_price,
        )

    def register_closed_trade(self, symbol: str, pnl_usd: Decimal | float | str, r_multiple: Decimal | float | str) -> None:
        pnl = _d(pnl_usd)
        closed_at = datetime.now(UTC)
        self.open_positions.pop(symbol, None)
        self.closed_trades.append(ClosedTradeStats(pnl, _d(r_multiple), closed_at))
        self.daily_realized[closed_at.date()] = self.daily_realized.get(closed_at.date(), Decimal("0")) + pnl
        week_key = closed_at.isocalendar()[:2]
        self.weekly_realized[week_key] = self.weekly_realized.get(week_key, Decimal("0")) + pnl

    def update_trailing_stop(self, symbol: str, current_price: Decimal | float | str, atr_value: Decimal | float | str) -> Decimal | None:
        position = self.open_positions.get(symbol)
        if not position:
            return None
        price = _d(current_price)
        atr_d = _d(atr_value)
        if position.direction == "LONG":
            position.highest_price = max(position.highest_price or price, price)
            if price - position.entry_price >= atr_d:
                position.sl_price = max(position.sl_price, position.highest_price - atr_d)
        else:
            position.lowest_price = min(position.lowest_price or price, price)
            if position.entry_price - price >= atr_d:
                position.sl_price = min(position.sl_price, position.lowest_price + atr_d)
        return position.sl_price

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
        self.current_equity = equity
        if self.peak_equity <= 0:
            self.peak_equity = equity
        else:
            self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0 and equity < self.peak_equity:
            drawdown_pct = (self.peak_equity - equity) / self.peak_equity * Decimal("100")
            if drawdown_pct >= _d(self.settings.drawdown_circuit_breaker_pct):
                self.circuit_breaker_active = True

    def _blocked(self, reason: str, checklist: list[ChecklistItem]) -> TradePlan:
        return TradePlan(False, reason, checklist)

    def _load_overrides(self) -> None:
        override_path = self.settings.runtime_dir / "risk_overrides.json"
        if not override_path.exists():
            self._override_max_risk = float(self.settings.max_risk_per_trade_pct)
            self._override_max_daily = float(self.settings.max_daily_loss_pct)
            self._override_max_weekly = float(self.settings.max_weekly_loss_pct)
            self._override_max_trades = int(self.settings.max_concurrent_trades)
            self._override_confidence = float(self.settings.confidence_threshold)
            return
        try:
            overrides = json.loads(override_path.read_text(encoding="utf-8"))
            self._override_max_risk = float(
                overrides.get("max_risk_per_trade_pct", self.settings.max_risk_per_trade_pct)
            )
            self._override_max_daily = float(overrides.get("max_daily_loss_pct", self.settings.max_daily_loss_pct))
            self._override_max_weekly = float(overrides.get("max_weekly_loss_pct", self.settings.max_weekly_loss_pct))
            self._override_max_trades = int(overrides.get("max_concurrent_trades", self.settings.max_concurrent_trades))
            self._override_confidence = float(overrides.get("confidence_threshold", self.settings.confidence_threshold))
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
