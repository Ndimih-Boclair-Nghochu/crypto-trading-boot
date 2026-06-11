from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from config import Settings
from utils.binance_client import ResilientBinanceClient, SymbolFilters, round_step_size, round_tick_size


def run(coro):
    return asyncio.run(coro)


def test_round_step_and_tick_size_use_decimal_flooring() -> None:
    assert round_step_size(Decimal("1.234567"), Decimal("0.0001")) == Decimal("1.2345")
    assert round_tick_size(Decimal("100.009"), Decimal("0.01")) == Decimal("100.00")


def test_symbol_filters_validate_min_qty_and_notional() -> None:
    filters = SymbolFilters(
        symbol="BTCUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.0001"),
        min_notional=Decimal("10"),
        min_qty=Decimal("0.001"),
    )

    assert filters.validate_order(Decimal("100"), Decimal("0.0009")) == (
        False,
        "quantity 0.0009 below minQty 0.001",
    )
    assert filters.validate_order(Decimal("100"), Decimal("0.001")) == (
        False,
        "notional 0.100 below minNotional 10",
    )
    assert filters.validate_order(Decimal("10000"), Decimal("0.001")) == (True, None)


def test_cached_exchange_info_produces_symbol_filters_without_network() -> None:
    client = ResilientBinanceClient(Settings())
    client.exchange_info = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0002", "maxQty": "100"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
            ],
        }
    }
    client.exchange_info_updated_at = datetime.now(UTC)

    filters = run(client.get_symbol_filters("BTCUSDT"))

    assert filters == SymbolFilters(
        symbol="BTCUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.0001"),
        min_notional=Decimal("10"),
        min_qty=Decimal("0.0002"),
        max_qty=Decimal("100"),
    )


def test_weight_headers_drive_historical_sleep_backoff() -> None:
    client = ResilientBinanceClient(Settings())
    client._update_weight_from_headers({"X-MBX-USED-WEIGHT-1M": "5400"})

    assert client.used_weight_1m == 5400
    assert client._historical_sleep_seconds() == 5.0
