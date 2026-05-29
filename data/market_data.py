from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class Candle:
    symbol: str
    timeframe: str
    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: int

    def is_valid(self) -> bool:
        return all(value > 0 for value in (self.open, self.high, self.low, self.close, self.volume))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open_time": self.open_time,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "close_time": self.close_time,
        }

    @property
    def open_dt(self) -> datetime:
        return datetime.fromtimestamp(self.open_time / 1000, tz=UTC)


@dataclass(frozen=True)
class OrderBookLevel:
    price: Decimal
    quantity: Decimal


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def empty(cls, symbol: str) -> "OrderBookSnapshot":
        return cls(symbol=symbol, bids=[], asks=[])

    @property
    def spread(self) -> Decimal:
        if not self.bids or not self.asks:
            return Decimal("0")
        return self.asks[0].price - self.bids[0].price

    @property
    def bid_ask_imbalance(self) -> Decimal:
        bid_qty = sum(level.quantity for level in self.bids)
        ask_qty = sum(level.quantity for level in self.asks)
        total = bid_qty + ask_qty
        if total == 0:
            return Decimal("0")
        return (bid_qty - ask_qty) / total


@dataclass(frozen=True)
class RecentTrade:
    symbol: str
    trade_id: int
    price: Decimal
    quantity: Decimal
    trade_time: int
    is_buyer_maker: bool


@dataclass
class MarketMeta:
    fear_greed: float | None = None
    btc_dominance: float | None = None
    funding_rates: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_interest: dict[str, dict[str, Any]] = field(default_factory=dict)
    liquidations: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    updated_at: datetime | None = None


class RollingMarketStore:
    def __init__(self, max_candles: int = 500) -> None:
        self.max_candles = max_candles
        self.candles: dict[str, dict[str, deque[Candle]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_candles))
        )
        self.order_books: dict[str, OrderBookSnapshot] = {}
        self.recent_trades: dict[str, deque[RecentTrade]] = defaultdict(lambda: deque(maxlen=500))
        self.meta = MarketMeta()
        self.lock = asyncio.Lock()

    async def replace_candles(self, symbol: str, timeframe: str, candles: list[Candle]) -> None:
        valid = [c for c in candles if c.is_valid()]
        async with self.lock:
            bucket = self.candles[symbol][timeframe]
            bucket.clear()
            bucket.extend(valid[-self.max_candles :])

    async def add_candle(self, candle: Candle) -> None:
        if not candle.is_valid():
            return
        async with self.lock:
            bucket = self.candles[candle.symbol][candle.timeframe]
            if bucket and bucket[-1].open_time == candle.open_time:
                bucket[-1] = candle
            else:
                bucket.append(candle)

    async def set_order_book(self, snapshot: OrderBookSnapshot) -> None:
        async with self.lock:
            self.order_books[snapshot.symbol] = snapshot

    async def add_recent_trades(self, symbol: str, trades: list[RecentTrade]) -> None:
        async with self.lock:
            self.recent_trades[symbol].extend(trades)

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            return {
                "candles": {
                    symbol: {tf: list(values) for tf, values in frames.items()}
                    for symbol, frames in self.candles.items()
                },
                "order_books": dict(self.order_books),
                "recent_trades": {symbol: list(values) for symbol, values in self.recent_trades.items()},
                "meta": self.meta,
            }

    async def get_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        async with self.lock:
            return list(self.candles[symbol][timeframe])


def resample_4h_from_1h(symbol: str, candles_1h: list[Candle]) -> list[Candle]:
    """Build closed 4h candles from 1h candles when the direct feed is unavailable."""
    grouped: dict[int, list[Candle]] = defaultdict(list)
    for candle in candles_1h:
        dt = candle.open_dt
        bucket_hour = (dt.hour // 4) * 4
        bucket_dt = dt.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
        grouped[int(bucket_dt.timestamp() * 1000)].append(candle)

    out: list[Candle] = []
    for open_time in sorted(grouped):
        group = sorted(grouped[open_time], key=lambda c: c.open_time)
        if len(group) < 4:
            continue
        out.append(
            Candle(
                symbol=symbol,
                timeframe="4h",
                open_time=open_time,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum((c.volume for c in group), Decimal("0")),
                close_time=group[-1].close_time,
            )
        )
    return out
