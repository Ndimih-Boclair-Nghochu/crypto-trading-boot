from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from config import Settings, settings
from data.market_data import RollingMarketStore, resample_4h_from_1h
from utils.binance_client import ResilientBinanceClient
from utils.logger import logger

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import websockets
except Exception:  # pragma: no cover
    websockets = None


class MarketDataPipeline:
    def __init__(
        self,
        client: ResilientBinanceClient,
        cfg: Settings = settings,
        store: RollingMarketStore | None = None,
    ) -> None:
        self.client = client
        self.settings = cfg
        self.store = store or RollingMarketStore(max_candles=500)
        self._tasks: list[asyncio.Task[Any]] = []
        self._last_fng_fetch: datetime | None = None
        self._last_dominance_fetch: datetime | None = None

    async def start_streams(self) -> None:
        for symbol in self.settings.symbols:
            self._tasks.append(asyncio.create_task(self._poll_symbol(symbol), name=f"poll-{symbol}"))
        self._tasks.append(asyncio.create_task(self._meta_loop(), name="meta-loop"))
        if websockets:
            self._tasks.append(asyncio.create_task(self._liquidation_stream(), name="liquidations"))
        else:
            logger.warning("websockets package unavailable; liquidation stream disabled.")

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def update_all(self) -> dict[str, Any]:
        await asyncio.gather(*(self._update_symbol(symbol) for symbol in self.settings.symbols), return_exceptions=True)
        await self._refresh_hourly_meta(force=False)
        return await self.store.snapshot()

    async def _poll_symbol(self, symbol: str) -> None:
        while True:
            await self._update_symbol(symbol)
            await asyncio.sleep(30)

    async def _update_symbol(self, symbol: str) -> None:
        tasks = [
            self._fetch_timeframe(symbol, timeframe)
            for timeframe in self.settings.timeframes
            if timeframe.lower() != "4h"
        ]
        tasks.append(self._fetch_order_book(symbol))
        tasks.append(self._fetch_recent_trades(symbol))
        tasks.append(self._fetch_futures_meta(symbol))
        await asyncio.gather(*tasks, return_exceptions=True)

        if "4H" in self.settings.timeframes or "4h" in self.settings.timeframes:
            direct = await self.client.get_ohlcv(symbol, "4h", 500)
            if direct:
                await self.store.replace_candles(symbol, "4h", direct)
            else:
                candles_1h = await self.store.get_candles(symbol, "1h")
                await self.store.replace_candles(symbol, "4h", resample_4h_from_1h(symbol, candles_1h))

    async def _fetch_timeframe(self, symbol: str, timeframe: str) -> None:
        candles = await self.client.get_ohlcv(symbol, timeframe.lower(), 500)
        if candles:
            await self.store.replace_candles(symbol, timeframe.lower(), candles)

    async def _fetch_order_book(self, symbol: str) -> None:
        snapshot = await self.client.get_order_book(symbol, 20)
        if snapshot.symbol:
            await self.store.set_order_book(snapshot)

    async def _fetch_recent_trades(self, symbol: str) -> None:
        trades = await self.client.get_recent_trades(symbol, 100)
        await self.store.add_recent_trades(symbol, trades)

    async def _fetch_futures_meta(self, symbol: str) -> None:
        funding, open_interest = await asyncio.gather(
            self.client.get_funding_rate(symbol),
            self.client.get_open_interest(symbol),
            return_exceptions=True,
        )
        async with self.store.lock:
            if isinstance(funding, dict):
                self.store.meta.funding_rates[symbol] = funding
            if isinstance(open_interest, dict):
                self.store.meta.open_interest[symbol] = open_interest
            self.store.meta.updated_at = datetime.now(UTC)

    async def _meta_loop(self) -> None:
        while True:
            await self._refresh_hourly_meta(force=False)
            await asyncio.sleep(300)

    async def _refresh_hourly_meta(self, force: bool = False) -> None:
        now = datetime.now(UTC)
        should_fetch_fng = force or not self._last_fng_fetch or now - self._last_fng_fetch >= timedelta(hours=1)
        should_fetch_dom = force or not self._last_dominance_fetch or now - self._last_dominance_fetch >= timedelta(hours=1)
        if should_fetch_fng:
            value = await self._fetch_fear_greed()
            async with self.store.lock:
                self.store.meta.fear_greed = value
                self.store.meta.updated_at = now
            self._last_fng_fetch = now
        if should_fetch_dom:
            value = await self._fetch_btc_dominance()
            async with self.store.lock:
                self.store.meta.btc_dominance = value
                self.store.meta.updated_at = now
            self._last_dominance_fetch = now

    async def _fetch_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any] | None:
        try:
            if aiohttp:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        return await response.json()
            return await asyncio.to_thread(lambda: requests.get(url, params=params, timeout=15).json())
        except Exception as exc:
            logger.warning(f"Public data fetch failed for {url}: {exc}")
            return None

    async def _fetch_fear_greed(self) -> float | None:
        raw = await self._fetch_json("https://api.alternative.me/fng/", {"limit": 1, "format": "json"})
        try:
            return float(raw["data"][0]["value"])  # type: ignore[index]
        except Exception:
            return None

    async def _fetch_btc_dominance(self) -> float | None:
        params = {"fsym": "BTC", "tsyms": "USD"}
        if self.settings.cryptocompare_api_key:
            params["api_key"] = self.settings.cryptocompare_api_key
        raw = await self._fetch_json("https://min-api.cryptocompare.com/data/top/mktcapfull", {"limit": 20, "tsym": "USD"})
        try:
            entries = raw["Data"]  # type: ignore[index]
            total = sum(float(item["RAW"]["USD"]["MKTCAP"]) for item in entries if "RAW" in item)
            btc = next(float(item["RAW"]["USD"]["MKTCAP"]) for item in entries if item["CoinInfo"]["Name"] == "BTC")
            return (btc / total) * 100 if total else None
        except Exception:
            return None

    async def _liquidation_stream(self) -> None:
        url = f"{self.settings.binance_futures_ws_base_url}/!forceOrder@arr"
        backoff = 1
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore[union-attr]
                    backoff = 1
                    async for message in ws:
                        async with self.store.lock:
                            self.store.meta.liquidations.append({"raw": message, "received_at": datetime.now(UTC).isoformat()})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Liquidation stream disconnected: {exc}; reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
