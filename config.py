from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dependency fallback
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"
RUNTIME_DIR.mkdir(exist_ok=True)

if load_dotenv:
    load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)


def _csv(name: str, default: Iterable[str]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return tuple(default)
    return tuple(part.strip().upper() for part in raw.split(",") if part.strip())


def _database_url() -> str:
    raw = os.getenv("DATABASE_URL")
    if not raw:
        if _bool("ALLOW_LOCALHOST_DB_FALLBACK", False):
            # Opt-in only, for genuine local development without a .env
            # file. Never silently used in a real deployment: if
            # DATABASE_URL is missing there, that's a misconfiguration that
            # should fail loudly and immediately, not connect to a
            # localhost Postgres that can never exist in a container.
            return "postgresql+asyncpg://botuser:strongpassword@localhost:5432/crypto_bot"
        raise RuntimeError(
            "DATABASE_URL is not set. This must be configured as an environment "
            "variable / secret on whatever platform this is running on:\n"
            "  - Render: service -> Environment -> add DATABASE_URL\n"
            "  - Fly.io: `fly secrets set DATABASE_URL=\"...\"` (then redeploy/restart "
            "the machine -- secrets set on an already-running machine do not "
            "retroactively apply until it restarts)\n"
            "For local development only, set ALLOW_LOCALHOST_DB_FALLBACK=true to use "
            "a localhost default instead of setting DATABASE_URL explicitly."
        )
    # Hosting providers (Render, Heroku, Fly, etc.) inject DATABASE_URL using
    # the plain "postgres://" or "postgresql://" scheme, which SQLAlchemy's
    # async engine + asyncpg driver cannot use directly. Normalize the scheme
    # so the same env var works without manual edits in the provider's
    # dashboard.
    if raw.startswith("postgres://"):
        raw = "postgresql+asyncpg://" + raw[len("postgres://"):]
    elif raw.startswith("postgresql://"):
        raw = "postgresql+asyncpg://" + raw[len("postgresql://"):]
    return raw


@dataclass(frozen=True)
class Settings:
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_secret: str = field(default_factory=lambda: os.getenv("BINANCE_SECRET", ""))
    use_testnet: bool = field(default_factory=lambda: _bool("USE_TESTNET", True))
    live_trading_reviewed: bool = field(default_factory=lambda: _bool("LIVE_TRADING_REVIEWED", False))
    testnet_trade_count: int = field(default_factory=lambda: _int("TESTNET_TRADE_COUNT", 0))

    database_url: str = field(default_factory=_database_url)

    symbols: tuple[str, ...] = field(
        default_factory=lambda: _csv("SYMBOLS", ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"))
    )
    timeframes: tuple[str, ...] = field(
        default_factory=lambda: _csv("TIMEFRAMES", ("1m", "5m", "15m", "1h", "4h", "1d"))
    )

    max_risk_per_trade_pct: float = field(default_factory=lambda: _float("MAX_RISK_PER_TRADE_PCT", 1.0))
    max_daily_loss_pct: float = field(default_factory=lambda: _float("MAX_DAILY_LOSS_PCT", 4.0))
    max_weekly_loss_pct: float = field(default_factory=lambda: _float("MAX_WEEKLY_LOSS_PCT", 8.0))
    max_concurrent_trades: int = field(default_factory=lambda: _int("MAX_CONCURRENT_TRADES", 3))
    confidence_threshold: float = field(default_factory=lambda: _float("CONFIDENCE_THRESHOLD", 0.70))
    max_portfolio_risk_pct: float = field(default_factory=lambda: _float("MAX_PORTFOLIO_RISK_PCT", 3.0))
    drawdown_circuit_breaker_pct: float = field(default_factory=lambda: _float("DRAWDOWN_CIRCUIT_BREAKER_PCT", 10.0))

    cryptocompare_api_key: str = field(default_factory=lambda: os.getenv("CRYPTOCOMPARE_API_KEY", ""))
    economic_calendar_api_url: str = field(default_factory=lambda: os.getenv("ECONOMIC_CALENDAR_API_URL", ""))
    require_economic_calendar: bool = field(default_factory=lambda: _bool("REQUIRE_ECONOMIC_CALENDAR", False))

    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    base_dir: Path = BASE_DIR
    runtime_dir: Path = RUNTIME_DIR
    trading_state_path: Path = RUNTIME_DIR / "trading_state.json"

    @property
    def live_trading_allowed(self) -> bool:
        return self.use_testnet or (self.live_trading_reviewed and self.testnet_trade_count >= 100)

    def assert_live_trading_allowed(self) -> None:
        if not self.live_trading_allowed:
            raise RuntimeError(
                "Live trading is locked. Run at least 100 testnet trades and set "
                "LIVE_TRADING_REVIEWED=true before USE_TESTNET=false."
            )

    @property
    def binance_spot_base_url(self) -> str:
        return "https://testnet.binance.vision" if self.use_testnet else "https://api.binance.com"

    @property
    def binance_futures_base_url(self) -> str:
        return "https://testnet.binancefuture.com" if self.use_testnet else "https://fapi.binance.com"

    @property
    def binance_ws_base_url(self) -> str:
        return "wss://testnet.binance.vision/ws" if self.use_testnet else "wss://stream.binance.com:9443/ws"

    @property
    def binance_futures_ws_base_url(self) -> str:
        return "wss://stream.binancefuture.com/ws"


settings = Settings()
