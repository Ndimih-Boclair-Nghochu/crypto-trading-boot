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
    confidence_threshold: float = field(default_factory=lambda: _float("CONFIDENCE_THRESHOLD", 0.45))
    max_portfolio_risk_pct: float = field(default_factory=lambda: _float("MAX_PORTFOLIO_RISK_PCT", 3.0))
    max_position_pct: float = field(default_factory=lambda: _float("MAX_POSITION_PCT", 5.0))
    drawdown_circuit_breaker_pct: float = field(default_factory=lambda: _float("DRAWDOWN_CIRCUIT_BREAKER_PCT", 10.0))
    # Once tripped, the breaker used to latch forever with no code path that
    # cleared it, so a single bad reading stopped trading until someone noticed
    # and restarted the process. It now re-arms when equity recovers to within
    # this percentage of the peak.
    circuit_breaker_reset_pct: float = field(default_factory=lambda: _float("CIRCUIT_BREAKER_RESET_PCT", 5.0))

    # --- venue -------------------------------------------------------------
    # "spot" or "futures". On spot there is nothing to sell, so SHORT setups are
    # rejected by the gate instead of being sent to Binance to be refused.
    market_type: str = field(default_factory=lambda: os.getenv("MARKET_TYPE", "spot").strip().lower())
    taker_fee_rate: float = field(default_factory=lambda: _float("TAKER_FEE_RATE", 0.001))

    # --- signal quality ----------------------------------------------------
    lstm_min_margin: float = field(default_factory=lambda: _float("LSTM_MIN_MARGIN", 0.15))
    min_model_quality: float = field(default_factory=lambda: _float("MIN_MODEL_QUALITY", 0.40))
    min_confluence_score: float = field(default_factory=lambda: _float("MIN_CONFLUENCE_SCORE", 60.0))
    min_indicator_agreement: int = field(default_factory=lambda: _int("MIN_INDICATOR_AGREEMENT", 4))
    min_conviction: float = field(default_factory=lambda: _float("MIN_CONVICTION", 0.20))
    rl_veto_confidence: float = field(default_factory=lambda: _float("RL_VETO_CONFIDENCE", 0.55))

    # --- position sizing ---------------------------------------------------
    # Risk scales between these two bounds with conviction, instead of every
    # trade betting the same fraction regardless of how good the setup is.
    risk_min_pct: float = field(default_factory=lambda: _float("RISK_MIN_PCT", 0.25))
    risk_max_pct: float = field(default_factory=lambda: _float("RISK_MAX_PCT", 1.25))
    risk_curve_gamma: float = field(default_factory=lambda: _float("RISK_CURVE_GAMMA", 1.5))

    # --- entries -----------------------------------------------------------
    primary_timeframe: str = field(default_factory=lambda: os.getenv("PRIMARY_TIMEFRAME", "1h").strip().lower())
    entry_on_closed_candle: bool = field(default_factory=lambda: _bool("ENTRY_ON_CLOSED_CANDLE", True))
    loss_cooldown_bars: int = field(default_factory=lambda: _int("LOSS_COOLDOWN_BARS", 2))

    # --- exits -------------------------------------------------------------
    stop_atr_multiple: float = field(default_factory=lambda: _float("STOP_ATR_MULTIPLE", 1.5))
    # How far ahead the training labels look. Together with stop_atr_multiple
    # this makes a label mean exactly what the trade does: "did price reach +1R
    # before -1R within this many bars".
    label_horizon: int = field(default_factory=lambda: _int("LABEL_HORIZON", 12))
    tp1_r_multiple: float = field(default_factory=lambda: _float("TP1_R_MULTIPLE", 1.0))
    tp2_r_multiple: float = field(default_factory=lambda: _float("TP2_R_MULTIPLE", 2.0))
    final_target_r_multiple: float = field(default_factory=lambda: _float("FINAL_TARGET_R_MULTIPLE", 4.0))
    scale_out_tp1_fraction: float = field(default_factory=lambda: _float("SCALE_OUT_TP1_FRACTION", 0.40))
    scale_out_tp2_fraction: float = field(default_factory=lambda: _float("SCALE_OUT_TP2_FRACTION", 0.30))
    # Profit lock: once a trade has been this far in front, its stop is moved to
    # entry plus costs and never moves back. This is what stops a winner from
    # completing the round trip into a loss.
    profit_lock_arm_r: float = field(default_factory=lambda: _float("PROFIT_LOCK_ARM_R", 0.25))
    profit_lock_give_back: float = field(default_factory=lambda: _float("PROFIT_LOCK_GIVE_BACK", 0.50))
    trail_arm_r: float = field(default_factory=lambda: _float("TRAIL_ARM_R", 1.0))
    trail_atr_multiple: float = field(default_factory=lambda: _float("TRAIL_ATR_MULTIPLE", 2.5))
    reversal_exit_enabled: bool = field(default_factory=lambda: _bool("REVERSAL_EXIT_ENABLED", True))
    reversal_exit_min_r: float = field(default_factory=lambda: _float("REVERSAL_EXIT_MIN_R", 0.15))
    time_stop_hours: float = field(default_factory=lambda: _float("TIME_STOP_HOURS", 8.0))

    # --- training / learning ----------------------------------------------
    # Testnet keeps only ~1100 hourly candles and prices them on its own
    # matching engine. Historical training therefore reads mainnet's public
    # kline endpoint (no key required) while orders still route to whichever
    # venue USE_TESTNET selects.
    training_data_source: str = field(
        default_factory=lambda: os.getenv("TRAINING_DATA_SOURCE", "mainnet").strip().lower()
    )
    training_days: int = field(default_factory=lambda: _int("TRAINING_DAYS", 730))
    training_interval: str = field(default_factory=lambda: os.getenv("TRAINING_INTERVAL", "1h").strip().lower())
    learning_interval_minutes: int = field(default_factory=lambda: _int("LEARNING_INTERVAL_MINUTES", 360))
    min_retrain_rows: int = field(default_factory=lambda: _int("MIN_RETRAIN_ROWS", 1500))
    min_rl_update_rows: int = field(default_factory=lambda: _int("MIN_RL_UPDATE_ROWS", 500))

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
    def shorting_available(self) -> bool:
        return self.market_type == "futures"

    @property
    def timeframe_preference(self) -> tuple[str, ...]:
        """Analysis timeframes in priority order, primary first."""
        rest = tuple(tf for tf in ("1h", "15m", "4h", "5m", "1m", "1d") if tf != self.primary_timeframe)
        return (self.primary_timeframe, *rest)

    @property
    def binance_spot_base_url(self) -> str:
        return "https://testnet.binance.vision" if self.use_testnet else "https://api.binance.com"

    @property
    def training_data_base_url(self) -> str:
        """Where historical klines come from, independent of where orders go."""
        if self.training_data_source == "venue":
            return self.binance_spot_base_url
        return "https://api.binance.com"

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
