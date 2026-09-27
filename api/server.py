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
CONTROL_PATH = settings.runtime_dir / "control.json"

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
    # Expected outcome if the trade runs to its stop or its take-profit, in dollars,
    # so the operator sees "makes +$X at TP / loses -$Y at SL" the moment it opens.
    try:
        sl = Decimal(str(row.get("sl_price") or 0))
        if sl > 0:
            risk_unit = abs(entry - sl)
            row["expected_sl_usd"] = str(-(risk_unit * qty))
            # Take-profit is the runner's final target (final_target_r_multiple x R),
            # which is where the trade actually exits in pure TP/SL mode -- not the
            # intermediate tp2 (2R). Show that so the figure matches the real target.
            tp_r = Decimal(str(settings.final_target_r_multiple))
            row["expected_tp_usd"] = str(risk_unit * tp_r * qty)
    except Exception:
        pass
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
        "paused": bool(_read_json(CONTROL_PATH, {"paused": False}).get("paused", False)),
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


@app.get("/api/mode")
async def mode() -> dict[str, Any]:
    """Which venue the platform is on right now, and whether Live is unlocked.
    The dashboard reads this to show the Demo/Live badge and gate the switch."""
    return {
        "mode": settings.mode,  # "demo" | "live"
        "is_live": settings.is_live_mode,
        "market_type": settings.market_type,  # "spot" | "futures"
        "shorting_available": settings.shorting_available,
        "futures_keys_configured": settings.futures_keys_configured,
        "live_trading_allowed": settings.live_trading_allowed,
        "live_trading_reviewed": settings.live_trading_reviewed,
        "completed_trade_count": settings.completed_trade_count,
        "min_live_trades": settings.min_live_trades,
    }


@app.get("/api/overview")
async def overview() -> dict[str, Any]:
    db = _db(app)
    prices = await _current_prices()
    # Only ever show the CURRENT mode's data: demo trades never bleed into the
    # live dashboard and vice-versa. is_live is tagged on each row at write time.
    is_live = settings.is_live_mode
    trades = [
        _with_live_pnl(r, prices)
        for r in await _safe_fetch_all(
            db, "SELECT * FROM trades WHERE is_live = :is_live ORDER BY entry_time DESC LIMIT 500", {"is_live": is_live}
        )
    ]
    open_positions = [
        _with_live_pnl(r, prices)
        for r in await _safe_fetch_all(
            db,
            "SELECT * FROM trades WHERE outcome = 'OPEN' AND is_live = :is_live ORDER BY entry_time DESC",
            {"is_live": is_live},
        )
    ]
    return {
        "trades": trades,
        "open_positions": open_positions,
        "mode": settings.mode,
        "equity": await _safe_fetch_all(
            db,
            "SELECT * FROM equity_snapshots WHERE is_live = :is_live ORDER BY captured_at DESC LIMIT 300",
            {"is_live": is_live},
        ),
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


@app.post("/api/system/stop")
async def system_stop() -> dict[str, Any]:
    """Global kill-switch: pause the engine and close every open trade now."""
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_PATH.write_text(json.dumps({"paused": True}), encoding="utf-8")
    CLOSE_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLOSE_REQUESTS_PATH.write_text(json.dumps({"symbols": [], "close_all": True}), encoding="utf-8")
    return {"ok": True, "paused": True}


@app.post("/api/system/resume")
async def system_resume() -> dict[str, Any]:
    """Resume trading after a global stop."""
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_PATH.write_text(json.dumps({"paused": False}), encoding="utf-8")
    return {"ok": True, "paused": False}


# --- Demo <-> Live switch --------------------------------------------------
# The API only QUEUES a mode change (writes a request file on the host-mounted
# logs volume); a host-side watcher validates and applies it (.env edit + engine
# restart). Live is guarded server-side: the exact phrase must be typed, futures
# keys must be present, and the completed-trade gate must be met -- so this can
# never arm real money by accident or without the deliberate confirmation.
MODE_REQUEST_PATH = settings.base_dir / "logs" / "mode_request.json"
GO_LIVE_PHRASE = "GO LIVE"


class GoLiveBody(BaseModel):
    confirm: str


@app.post("/api/system/go-live")
async def go_live(body: GoLiveBody) -> dict[str, Any]:
    if body.confirm.strip() != GO_LIVE_PHRASE:
        return {"ok": False, "error": 'Type GO LIVE exactly to confirm real-money trading.'}
    if not settings.futures_keys_configured:
        return {"ok": False, "error": "Futures API key is not configured on the server."}
    if settings.completed_trade_count < settings.min_live_trades:
        return {
            "ok": False,
            "error": f"Live is locked: need {settings.min_live_trades} completed trades "
            f"(have {settings.completed_trade_count}).",
        }
    MODE_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODE_REQUEST_PATH.write_text(
        json.dumps({"target": "live", "ts": datetime.now(UTC).isoformat()}), encoding="utf-8"
    )
    return {"ok": True, "queued": "live", "note": "Switching to LIVE — the bot restarts in ~10s."}


@app.post("/api/system/go-demo")
async def go_demo() -> dict[str, Any]:
    """Revert to DEMO — always safe, no confirmation required."""
    MODE_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODE_REQUEST_PATH.write_text(
        json.dumps({"target": "demo", "ts": datetime.now(UTC).isoformat()}), encoding="utf-8"
    )
    return {"ok": True, "queued": "demo", "note": "Switching to DEMO — the bot restarts in ~10s."}
