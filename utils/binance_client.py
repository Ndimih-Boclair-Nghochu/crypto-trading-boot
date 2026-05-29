from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypeVar

import requests

from config import Settings, settings
from data.market_data import Candle, OrderBookLevel, OrderBookSnapshot, RecentTrade
from utils.logger import logger

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None

try:
    from binance import AsyncClient
except Exception:  # pragma: no cover
    AsyncClient = None


T = TypeVar("T")


def _default(value: T | Callable[[], T]) -> T:
    return value() if callable(value) else value


def safe_api_call(default: T | Callable[[], T]) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(self: "ResilientBinanceClient", *args: Any, **kwargs: Any) -> T:
            try:
                return await fn(self, *args, **kwargs)
            except Exception as exc:
                logger.exception(f"Binance API call failed: {fn.__name__}: {exc}")
                self.connected = False
                return _default(default)

        return wrapper

    return decorator


@dataclass
class OrderResult:
    accepted: bool
    order_id: str | None = None
    status: str = "UNKNOWN"
    raw: dict[str, Any] | None = None
    reason: str | None = None


class ResilientBinanceClient:
    """Async Binance wrapper with testnet-first safety and fail-closed behavior."""

    def __init__(self, cfg: Settings = settings) -> None:
        self.settings = cfg
        self.client: Any | None = None
        self.session: Any | None = None
        self.connected = False
        self._init_lock = asyncio.Lock()
        self._health_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        async with self._init_lock:
            if not self.settings.use_testnet:
                self.settings.assert_live_trading_allowed()
            if aiohttp and self.session is None:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
            if AsyncClient and self.client is None:
                self.client = await AsyncClient.create(
                    api_key=self.settings.binance_api_key,
                    api_secret=self.settings.binance_secret,
                    testnet=self.settings.use_testnet,
                )
                if self.settings.use_testnet:
                    self.client.API_URL = self.settings.binance_spot_base_url + "/api"
            self.connected = await self.ping()

    async def close(self) -> None:
        if self._health_task:
            self._health_task.cancel()
        if self.client:
            await self.client.close_connection()
        if self.session:
            await self.session.close()

    async def _public_get(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{base_url}{path}"
        if aiohttp and self.session:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        return await asyncio.to_thread(lambda: requests.get(url, params=params, timeout=15).json())

    @safe_api_call(False)
    async def ping(self) -> bool:
        if self.client:
            await self.client.ping()
        else:
            await self._public_get(self.settings.binance_spot_base_url, "/api/v3/ping")
        self.connected = True
        return True

    def start_health_check(self, interval_seconds: int = 60) -> None:
        if self._health_task and not self._health_task.done():
            return
        self._health_task = asyncio.create_task(self._health_loop(interval_seconds))

    async def _health_loop(self, interval_seconds: int) -> None:
        backoff = 1
        while True:
            ok = await self.ping()
            if ok:
                backoff = 1
                await asyncio.sleep(interval_seconds)
                continue
            logger.warning(f"Binance disconnected; reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
            await self.initialize()

    @safe_api_call(list)
    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> list[Candle]:
        if self.client:
            raw = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        else:
            raw = await self._public_get(
                self.settings.binance_spot_base_url,
                "/api/v3/klines",
                {"symbol": symbol, "interval": interval, "limit": limit},
            )
        candles = [
            Candle(
                symbol=symbol,
                timeframe=interval,
                open_time=int(row[0]),
                open=Decimal(str(row[1])),
                high=Decimal(str(row[2])),
                low=Decimal(str(row[3])),
                close=Decimal(str(row[4])),
                volume=Decimal(str(row[5])),
                close_time=int(row[6]),
            )
            for row in raw
        ]
        return [c for c in candles if c.is_valid()]

    @safe_api_call(list)
    async def get_historical_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[Candle]:
        candles: list[Candle] = []
        cursor = start_time_ms
        while cursor < end_time_ms:
            if self.client:
                raw = await self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=cursor,
                    endTime=end_time_ms,
                    limit=limit,
                )
            else:
                raw = await self._public_get(
                    self.settings.binance_spot_base_url,
                    "/api/v3/klines",
                    {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_time_ms, "limit": limit},
                )
            if not raw:
                break
            batch = [
                Candle(
                    symbol=symbol,
                    timeframe=interval,
                    open_time=int(row[0]),
                    open=Decimal(str(row[1])),
                    high=Decimal(str(row[2])),
                    low=Decimal(str(row[3])),
                    close=Decimal(str(row[4])),
                    volume=Decimal(str(row[5])),
                    close_time=int(row[6]),
                )
                for row in raw
            ]
            candles.extend(c for c in batch if c.is_valid())
            next_cursor = int(raw[-1][6]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            await asyncio.sleep(0.1)
        return candles

    @safe_api_call(lambda: OrderBookSnapshot.empty(""))
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBookSnapshot:
        if self.client:
            raw = await self.client.get_order_book(symbol=symbol, limit=limit)
        else:
            raw = await self._public_get(
                self.settings.binance_spot_base_url,
                "/api/v3/depth",
                {"symbol": symbol, "limit": limit},
            )
        return OrderBookSnapshot(
            symbol=symbol,
            bids=[OrderBookLevel(Decimal(str(price)), Decimal(str(qty))) for price, qty in raw.get("bids", [])[:limit]],
            asks=[OrderBookLevel(Decimal(str(price)), Decimal(str(qty))) for price, qty in raw.get("asks", [])[:limit]],
        )

    @safe_api_call(list)
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[RecentTrade]:
        if self.client:
            raw = await self.client.get_recent_trades(symbol=symbol, limit=limit)
        else:
            raw = await self._public_get(
                self.settings.binance_spot_base_url,
                "/api/v3/trades",
                {"symbol": symbol, "limit": limit},
            )
        return [
            RecentTrade(
                symbol=symbol,
                trade_id=int(row["id"]),
                price=Decimal(str(row["price"])),
                quantity=Decimal(str(row["qty"])),
                trade_time=int(row["time"]),
                is_buyer_maker=bool(row.get("isBuyerMaker", False)),
            )
            for row in raw
        ]

    @safe_api_call(None)
    async def get_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        raw = await self._public_get(
            self.settings.binance_futures_base_url,
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1},
        )
        return raw[0] if raw else None

    @safe_api_call(None)
    async def get_open_interest(self, symbol: str) -> dict[str, Any] | None:
        return await self._public_get(
            self.settings.binance_futures_base_url,
            "/fapi/v1/openInterest",
            {"symbol": symbol},
        )

    @safe_api_call(Decimal("0"))
    async def get_usdt_balance(self) -> Decimal:
        if not self.client:
            logger.warning("Private balance API requires python-binance client.")
            return Decimal("0")
        account = await self.client.get_account()
        for balance in account.get("balances", []):
            if balance.get("asset") == "USDT":
                return Decimal(str(balance.get("free", "0")))
        return Decimal("0")

    @safe_api_call(lambda: OrderResult(False, reason="order API failed"))
    async def place_order(self, **kwargs: Any) -> OrderResult:
        if not self.client:
            return OrderResult(False, reason="python-binance is required for authenticated order placement")
        raw = await self.client.create_order(**kwargs)
        return OrderResult(True, str(raw.get("orderId")), str(raw.get("status", "NEW")), raw=raw)

    @safe_api_call(False)
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if not self.client:
            return False
        await self.client.cancel_order(symbol=symbol, orderId=order_id)
        return True

    @safe_api_call(None)
    async def get_order(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        if not self.client:
            return None
        return await self.client.get_order(symbol=symbol, orderId=order_id)

    @safe_api_call(lambda: OrderResult(False, reason="OCO API failed"))
    async def place_oco_order(self, **kwargs: Any) -> OrderResult:
        if not self.client:
            return OrderResult(False, reason="python-binance is required for OCO order placement")
        if hasattr(self.client, "create_order_list_oco"):
            raw = await self.client.create_order_list_oco(**kwargs)
        elif hasattr(self.client, "create_oco_order"):
            raw = await self.client.create_oco_order(**_legacy_oco_kwargs(kwargs))
        else:
            return OrderResult(False, reason="python-binance client does not expose an OCO helper")
        order_list_id = raw.get("orderListId") or raw.get("listOrderStatus")
        return OrderResult(True, str(order_list_id), str(raw.get("listOrderStatus", "EXECUTING")), raw=raw)

    @safe_api_call(list)
    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if not self.client:
            return []
        if symbol:
            return await self.client.get_open_orders(symbol=symbol)
        return await self.client.get_open_orders()


def _legacy_oco_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "price" in kwargs and "stopPrice" in kwargs:
        return kwargs
    side = kwargs.get("side")
    if side == "SELL":
        return {
            "symbol": kwargs["symbol"],
            "side": side,
            "quantity": kwargs["quantity"],
            "price": kwargs.get("abovePrice"),
            "stopPrice": kwargs.get("belowStopPrice"),
            "stopLimitPrice": kwargs.get("belowPrice"),
            "stopLimitTimeInForce": kwargs.get("belowTimeInForce", "GTC"),
        }
    return {
        "symbol": kwargs["symbol"],
        "side": side,
        "quantity": kwargs["quantity"],
        "price": kwargs.get("belowPrice"),
        "stopPrice": kwargs.get("aboveStopPrice"),
        "stopLimitPrice": kwargs.get("abovePrice"),
        "stopLimitTimeInForce": kwargs.get("aboveTimeInForce", "GTC"),
    }
