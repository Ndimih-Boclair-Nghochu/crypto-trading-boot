from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
                reason = self._format_api_exception(exc) if hasattr(self, "_format_api_exception") else str(exc)
                logger.exception(f"Binance API call failed: {fn.__name__}: {reason}")
                self.connected = False
                fallback = _default(default)
                if hasattr(fallback, "accepted") and hasattr(fallback, "reason"):
                    return type(fallback)(
                        False,
                        status=getattr(fallback, "status", "REJECTED"),
                        reason=reason,
                        raw={"exception": type(exc).__name__},
                    )
                return fallback

        return wrapper

    return decorator


@dataclass
class OrderResult:
    accepted: bool
    order_id: str | None = None
    status: str = "UNKNOWN"
    raw: dict[str, Any] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    tick_size: Decimal = Decimal("0")
    step_size: Decimal = Decimal("0")
    min_notional: Decimal = Decimal("0")
    min_qty: Decimal = Decimal("0")
    max_qty: Decimal = Decimal("0")

    def round_price(self, price: Decimal | float | str) -> Decimal:
        return round_tick_size(price, self.tick_size)

    def round_quantity(self, quantity: Decimal | float | str) -> Decimal:
        return round_step_size(quantity, self.step_size)

    def validate_order(self, price: Decimal, quantity: Decimal) -> tuple[bool, str | None]:
        if self.min_qty > 0 and quantity < self.min_qty:
            return False, f"quantity {quantity} below minQty {self.min_qty}"
        if self.max_qty > 0 and quantity > self.max_qty:
            return False, f"quantity {quantity} above maxQty {self.max_qty}"
        notional = price * quantity
        if self.min_notional > 0 and notional < self.min_notional:
            return False, f"notional {notional} below minNotional {self.min_notional}"
        return True, None


class ResilientBinanceClient:
    """Async Binance wrapper with testnet-first safety and fail-closed behavior."""

    def __init__(self, cfg: Settings = settings) -> None:
        self.settings = cfg
        self.client: Any | None = None
        self.session: Any | None = None
        self.connected = False
        self._init_lock = asyncio.Lock()
        self._health_task: asyncio.Task[None] | None = None
        self._exchange_info_lock = asyncio.Lock()
        self.exchange_info: dict[str, dict[str, Any]] = {}
        self.exchange_info_updated_at: datetime | None = None
        self.exchange_info_ttl = timedelta(hours=24)
        self.used_weight_1m = 0
        self.weight_limit_1m = 6000
        self._retry_after_seconds: int | None = None

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
            await self.update_exchange_info()

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
                self._update_weight_from_headers(response.headers)
                response.raise_for_status()
                return await response.json()
        response = await asyncio.to_thread(lambda: requests.get(url, params=params, timeout=15))
        self._update_weight_from_headers(response.headers)
        response.raise_for_status()
        return response.json()

    def _update_weight_from_headers(self, headers: Any) -> None:
        try:
            used = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
            if used is not None:
                self.used_weight_1m = int(used)
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            self._retry_after_seconds = int(retry_after) if retry_after else None
        except Exception as exc:
            logger.warning(f"Could not parse Binance rate headers: {exc}")

    def _capture_client_response_headers(self) -> None:
        response = getattr(self.client, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            self._update_weight_from_headers(headers)

    async def _throttle_if_needed(self) -> None:
        threshold = int(self.weight_limit_1m * 0.8)
        if self._retry_after_seconds:
            await asyncio.sleep(max(self._retry_after_seconds, 1))
            self._retry_after_seconds = None
            return
        if self.used_weight_1m >= threshold:
            overage = max(self.used_weight_1m - threshold, 0)
            sleep_seconds = min(60, max(1, 1 + overage // 250))
            logger.warning(
                f"Binance request weight high ({self.used_weight_1m}/{self.weight_limit_1m}); throttling {sleep_seconds}s"
            )
            await asyncio.sleep(sleep_seconds)

    def _historical_sleep_seconds(self) -> float:
        ratio = self.used_weight_1m / self.weight_limit_1m if self.weight_limit_1m else 0
        if ratio >= 0.95:
            return 15.0
        if ratio >= 0.90:
            return 5.0
        if ratio >= 0.80:
            return 1.0
        return 0.1

    def _format_api_exception(self, exc: Exception) -> str:
        code = getattr(exc, "code", None)
        status_code = getattr(exc, "status_code", None)
        message = getattr(exc, "message", None) or str(exc)
        if code is not None:
            return f"Binance APIError {code}: {message}"
        if status_code is not None:
            return f"Binance HTTP {status_code}: {message}"
        return message

    @safe_api_call(dict)
    async def update_exchange_info(self, force: bool = False) -> dict[str, dict[str, Any]]:
        async with self._exchange_info_lock:
            now = datetime.now(UTC)
            if (
                not force
                and self.exchange_info
                and self.exchange_info_updated_at
                and now - self.exchange_info_updated_at < self.exchange_info_ttl
            ):
                return self.exchange_info
            raw = await self._public_get(self.settings.binance_spot_base_url, "/api/v3/exchangeInfo")
            self.exchange_info = {item["symbol"]: item for item in raw.get("symbols", []) if "symbol" in item}
            self.exchange_info_updated_at = now
            for limit in raw.get("rateLimits", []):
                if (
                    limit.get("rateLimitType") == "REQUEST_WEIGHT"
                    and limit.get("interval") == "MINUTE"
                    and int(limit.get("intervalNum", 1)) == 1
                ):
                    self.weight_limit_1m = int(limit.get("limit", self.weight_limit_1m))
            return self.exchange_info

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters | None:
        await self.update_exchange_info()
        symbol_info = self.exchange_info.get(symbol)
        if not symbol_info:
            return None
        by_type = {item.get("filterType"): item for item in symbol_info.get("filters", [])}
        price_filter = by_type.get("PRICE_FILTER", {})
        lot_filter = by_type.get("LOT_SIZE", {})
        min_notional_filter = by_type.get("MIN_NOTIONAL", {})
        notional_filter = by_type.get("NOTIONAL", {})
        min_notional = min_notional_filter.get("minNotional") or notional_filter.get("minNotional") or "0"
        return SymbolFilters(
            symbol=symbol,
            tick_size=Decimal(str(price_filter.get("tickSize", "0"))),
            step_size=Decimal(str(lot_filter.get("stepSize", "0"))),
            min_notional=Decimal(str(min_notional)),
            min_qty=Decimal(str(lot_filter.get("minQty", "0"))),
            max_qty=Decimal(str(lot_filter.get("maxQty", "0"))),
        )

    @safe_api_call(False)
    async def ping(self) -> bool:
        if self.client:
            await self.client.ping()
            self._capture_client_response_headers()
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
            self._capture_client_response_headers()
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
                self._capture_client_response_headers()
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
            await asyncio.sleep(self._historical_sleep_seconds())
        return candles

    @safe_api_call(lambda: OrderBookSnapshot.empty(""))
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBookSnapshot:
        if self.client:
            raw = await self.client.get_order_book(symbol=symbol, limit=limit)
            self._capture_client_response_headers()
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
            self._capture_client_response_headers()
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
        await self._throttle_if_needed()
        account = await self.client.get_account()
        self._capture_client_response_headers()
        for balance in account.get("balances", []):
            if balance.get("asset") == "USDT":
                return Decimal(str(balance.get("free", "0")))
        return Decimal("0")

    @safe_api_call(lambda: OrderResult(False, reason="order API failed"))
    async def place_order(self, **kwargs: Any) -> OrderResult:
        if not self.client:
            return OrderResult(False, reason="python-binance is required for authenticated order placement")
        await self._throttle_if_needed()
        raw = await self.client.create_order(**kwargs)
        self._capture_client_response_headers()
        return OrderResult(True, str(raw.get("orderId")), str(raw.get("status", "NEW")), raw=raw)

    @safe_api_call(False)
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if not self.client:
            return False
        await self._throttle_if_needed()
        await self.client.cancel_order(symbol=symbol, orderId=order_id)
        self._capture_client_response_headers()
        return True

    @safe_api_call(None)
    async def get_order(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        if not self.client:
            return None
        await self._throttle_if_needed()
        result = await self.client.get_order(symbol=symbol, orderId=order_id)
        self._capture_client_response_headers()
        return result

    @safe_api_call(lambda: OrderResult(False, reason="OCO API failed"))
    async def place_oco_order(self, **kwargs: Any) -> OrderResult:
        if not self.client:
            return OrderResult(False, reason="python-binance is required for OCO order placement")
        await self._throttle_if_needed()
        if hasattr(self.client, "create_order_list_oco"):
            raw = await self.client.create_order_list_oco(**kwargs)
        elif hasattr(self.client, "create_oco_order"):
            raw = await self.client.create_oco_order(**_legacy_oco_kwargs(kwargs))
        else:
            return OrderResult(False, reason="python-binance client does not expose an OCO helper")
        self._capture_client_response_headers()
        order_list_id = raw.get("orderListId") or raw.get("listOrderStatus")
        return OrderResult(True, str(order_list_id), str(raw.get("listOrderStatus", "EXECUTING")), raw=raw)

    @safe_api_call(list)
    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if not self.client:
            return []
        await self._throttle_if_needed()
        if symbol:
            result = await self.client.get_open_orders(symbol=symbol)
        else:
            result = await self.client.get_open_orders()
        self._capture_client_response_headers()
        return result


def round_step_size(quantity: Decimal | float | str, step_size: Decimal | float | str) -> Decimal:
    quantity_d = Decimal(str(quantity))
    step_d = Decimal(str(step_size))
    if step_d <= 0:
        return quantity_d
    return (quantity_d // step_d) * step_d


def round_tick_size(price: Decimal | float | str, tick_size: Decimal | float | str) -> Decimal:
    price_d = Decimal(str(price))
    tick_d = Decimal(str(tick_size))
    if tick_d <= 0:
        return price_d
    return (price_d // tick_d) * tick_d


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
