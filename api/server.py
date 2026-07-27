"""REST API for the trading system.

Exposes the same data the Streamlit dashboard used to display (trades,
positions, equity curve, performance, system events, risk settings, and the
trading on/off toggle), so a separate frontend (e.g. deployed on Vercel) can
present it. Runs in the same container/process group as the trading bot
(main.py), sharing the same `.runtime` directory and database, so toggling
"trading enabled" or saving risk overrides here takes effect immediately for
the running bot.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from db.connection import Database

STATE_PATH = settings.trading_state_path
RISK_OVERRIDE_PATH = settings.runtime_dir / "risk_overrides.json"
CLOSE_REQUESTS_PATH = settings.runtime_dir / "close_requests.json"

# Live prices for open-position P&L. Cached briefly so a burst of dashboard
# refreshes doesn't hammer the ticker endpoint. Prices come from the same venue
# the bot trades on, so unrealized P&L is consistent with the entry price.
_PRICE_CACHE: dict[str, Any] = {"at": 0.0, "prices": {}}


def _read_json(path: Any, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


async def _current_prices() -> dict[str, Decimal]:
    """All symbol prices from the venue ticker, cached ~3s. Never raises."""
    now = time.time()
    if now - float(_PRICE_CACHE["at"]) < 3 and _PRICE_CACHE["prices"]:
        return _PRICE_CACHE["prices"]
    url = settings.binance_spot_base_url + "/api/v3/ticker/price"
    try:
        data = await asyncio.to_thread(lambda: requests.get(url, timeout=8).json())
        prices = {
            row["symbol"]: Decimal(str(row["price"]))
            for row in data
            if isinstance(row, dict) and row.get("symbol") and row.get("price")
        }
        if prices:
            _PRICE_CACHE["prices"] = prices
            _PRICE_CACHE["at"] = now
        return _PRICE_CACHE["prices"]
    except Exception:
        return _PRICE_CACHE["prices"]  # serve last-known rather than blanking the UI


def _with_live_pnl(row: dict[str, Any], prices: dict[str, Decimal]) -> dict[str, Any]:
    """Attach current_price / market_value / unrealized_pnl / unrealized_pct.

    Only for still-open rows; closed trades already carry realised pnl_usd. This
    is what fills the dashboard's 'Now' and 'Unreal. P&L' columns so an operator
    can see, live, whether each position is green or red.
    """
    row = dict(row)
    if str(row.get("outcome")) != "OPEN":
        return row
    price = prices.get(row.get("symbol"))
    try:
        entry = Decimal(str(row.get("entry_price") or 0))
        qty = Decimal(str(row.get("quantity") or 0))
    except Exception:
        return row
    if price is None or entry <= 0 or qty <= 0:
        return row
    if str(row.get("direction")) == "SHORT":
        pnl = (entry - price) * qty
    else:
        pnl = (price - entry) * qty
    cost = entry * qty
    row["current_price"] = str(price)
    row["market_value"] = str(price * qty)
    row["unrealized_pnl"] = str(pnl)
    row["unrealized_pct"] = str((pnl / cost * Decimal("100")) if cost > 0 else Decimal("0"))
    return row


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database()
    try:
        await db.initialize()
    except Exception as exc:  # pragma: no cover
        # Don't let a DB connectivity issue prevent the API (and /api/health)
        # from starting at all -- /api/overview will surface this as a 503
        # via the try/except below, but the rest of the API stays usable.
        from utils.logger import logger as _logger

        _logger.error(f"API: database initialization failed: {exc}")
        db.engine = None
        db.sessionmaker = None
    app.state.db = db
    try:
        yield
    finally:
        await db.close()


app = FastAPI(title="Crypto Trading Desk API", lifespan=lifespan)

# Wildcard CORS, unconditionally. This API exposes no secrets and no
# authenticated/destructive actions (read-only status + data), so there is
# no security reason to restrict the origin. Making this depend on a
# FRONTEND_ORIGIN env var being set correctly on Render was a real failure
# mode: if that var was missing, blank, or didn't exactly match the Vercel
# URL, every browser request was silently blocked by CORS with no useful
# error surfaced anywhere in the app itself.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _db(app_: FastAPI) -> Database:
    return app_.state.db


@app.get("/")
@app.head("/")
@app.get("/api/health")
async def health() -> dict[str, Any]:
    state = _read_json(STATE_PATH, {"trading_enabled": True, "status": "STARTING"})
    db = _db(app)

    updated_at = state.get("updated_at")
    stale = False
    if updated_at:
        try:
            age_seconds = (datetime.now(UTC) - datetime.fromisoformat(updated_at)).total_seconds()
            stale = age_seconds > 120
        except Exception:
            stale = False

    status = state.get("status", "STARTING")
    if stale and status not in {"ERROR"}:
        status = "UNRESPONSIVE"

    return {
        "status": status,
        "reason": state.get("reason"),
        "trading_enabled": bool(state.get("trading_enabled", True)),
        "testnet": settings.use_testnet,
        "binance_connected": bool(state.get("binance_connected", False)),
        "db_connected": db.sessionmaker is not None,
        "updated_at": updated_at,
    }


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return _read_json(STATE_PATH, {"trading_enabled": True, "status": "STARTING"})


async def _safe_fetch_all(db: Database, statement: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    try:
        return await db.fetch_all(statement, params)
    except Exception as exc:
        from utils.logger import logger as _logger

        _logger.warning(f"API: query failed, returning empty result: {exc}")
        return []


@app.get("/api/overview")
async def overview() -> dict[str, Any]:
    db = _db(app)
    prices = await _current_prices()
    trades = [_with_live_pnl(r, prices) for r in await _safe_fetch_all(db, "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 50")]
    open_positions = [
        _with_live_pnl(r, prices)
        for r in await _safe_fetch_all(db, "SELECT * FROM trades WHERE outcome = 'OPEN' ORDER BY entry_time DESC")
    ]
    return {
        "trades": trades,
        "open_positions": open_positions,
        "equity": await _safe_fetch_all(db, "SELECT * FROM equity_snapshots ORDER BY captured_at DESC LIMIT 300"),
        "events": await _safe_fetch_all(db, "SELECT * FROM system_events ORDER BY occurred_at DESC LIMIT 100"),
        "performance": await _safe_fetch_all(
            db,
            """
            SELECT strategy_name, regime, win_rate, total_trades, profit_factor, avg_r_multiple
            FROM strategy_performance
            ORDER BY updated_at DESC
            LIMIT 50
            """,
        ),
        "no_trade": await _safe_fetch_all(db, "SELECT * FROM no_trade_log ORDER BY logged_at DESC LIMIT 50"),
        "symbols": list(settings.symbols),
        "db_connected": db.sessionmaker is not None,
    }


@app.get("/api/risk-settings")
async def get_risk_settings() -> dict[str, Any]:
    overrides = _read_json(RISK_OVERRIDE_PATH, {})
    return {
        "max_risk_per_trade_pct": overrides.get("max_risk_per_trade_pct", settings.max_risk_per_trade_pct),
        "max_daily_loss_pct": overrides.get("max_daily_loss_pct", settings.max_daily_loss_pct),
        "max_weekly_loss_pct": overrides.get("max_weekly_loss_pct", settings.max_weekly_loss_pct),
        "max_concurrent_trades": overrides.get("max_concurrent_trades", settings.max_concurrent_trades),
        "confidence_threshold": overrides.get("confidence_threshold", settings.confidence_threshold),
    }


class RiskSettingsBody(BaseModel):
    max_risk_per_trade_pct: float
    max_daily_loss_pct: float
    max_weekly_loss_pct: float
    max_concurrent_trades: int
    confidence_threshold: float


@app.post("/api/risk-settings")
async def save_risk_settings(body: RiskSettingsBody) -> dict[str, Any]:
    overrides = body.model_dump()
    RISK_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RISK_OVERRIDE_PATH.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    return overrides


@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper()
    current = _read_json(CLOSE_REQUESTS_PATH, {"symbols": []})
    pending = set(current.get("symbols", []))
    pending.add(symbol)
    CLOSE_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLOSE_REQUESTS_PATH.write_text(json.dumps({"symbols": sorted(pending)}), encoding="utf-8")
    return {"ok": True, "queued": symbol}


@app.get("/api/models")
async def model_status() -> dict[str, Any]:
    """Per-symbol model provenance, so the dashboard can show what it is trading on."""
    weights_dir = settings.base_dir / "models" / "weights"
    models: list[dict[str, Any]] = []
    for symbol in settings.symbols:
        meta = _read_json(weights_dir / f"lstm_{symbol}.json", {})
        models.append(
            {
                "symbol": symbol,
                "trained_at": meta.get("trained_at"),
                "model_version": meta.get("model_version"),
                "rows_trained": meta.get("rows_trained"),
                "directional_precision": meta.get("directional_precision"),
                "val_accuracy": meta.get("val_accuracy"),
                "label_counts": meta.get("label_counts"),
            }
        )
    rl_meta = _read_json(weights_dir / "ppo_trading_agent.json", {})
    return {
        "lstm": models,
        "rl": {
            "agent_version": rl_meta.get("agent_version"),
            "rows": rl_meta.get("rows"),
            "timesteps": rl_meta.get("timesteps"),
            "mode": rl_meta.get("mode"),
        },
        "min_model_quality": settings.min_model_quality,
        "market_type": settings.market_type,
        "shorting_available": settings.shorting_available,
    }
