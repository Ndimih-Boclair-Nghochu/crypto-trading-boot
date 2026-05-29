from __future__ import annotations

try:
    from sqlalchemy import CheckConstraint, DateTime, Integer, Numeric, String, Text, UniqueConstraint, func
    from sqlalchemy.dialects.postgresql import JSONB, UUID
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
    from sqlalchemy.schema import ForeignKey
except Exception:  # pragma: no cover
    DeclarativeBase = object  # type: ignore[assignment]
    Mapped = object  # type: ignore[assignment]


if DeclarativeBase is not object:

    class Base(DeclarativeBase):
        pass


    class Trade(Base):
        __tablename__ = "trades"
        __table_args__ = (
            CheckConstraint("direction IN ('LONG', 'SHORT')", name="ck_trades_direction"),
            CheckConstraint("outcome IN ('WIN', 'LOSS', 'BREAKEVEN', 'OPEN')", name="ck_trades_outcome"),
            CheckConstraint(
                "exit_reason IN ('TP1','TP2','SL','TRAILING','TIME_STOP','MANUAL','CIRCUIT_BREAKER')",
                name="ck_trades_exit_reason",
            ),
        )

        trade_id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
        symbol: Mapped[str] = mapped_column(String(20), nullable=False)
        direction: Mapped[str] = mapped_column(String(10), nullable=False)
        entry_price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
        exit_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
        sl_price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
        tp1_price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
        tp2_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
        quantity: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
        entry_time: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
        exit_time: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
        pnl_usd: Mapped[float | None] = mapped_column(Numeric(20, 8))
        pnl_pct: Mapped[float | None] = mapped_column(Numeric(10, 4))
        r_multiple: Mapped[float | None] = mapped_column(Numeric(10, 4))
        strategy_used: Mapped[str] = mapped_column(String(50), nullable=False)
        regime_at_entry: Mapped[str] = mapped_column(String(30), nullable=False)
        lstm_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
        rl_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
        confluence_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
        outcome: Mapped[str | None] = mapped_column(String(15))
        exit_reason: Mapped[str | None] = mapped_column(String(20))
        binance_order_id: Mapped[str | None] = mapped_column(String(50))
        created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

        snapshots: Mapped[list["MarketSnapshot"]] = relationship(back_populates="trade")


    class MarketSnapshot(Base):
        __tablename__ = "market_snapshots"

        snapshot_id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
        trade_id: Mapped[str | None] = mapped_column(UUID(as_uuid=True), ForeignKey("trades.trade_id", ondelete="CASCADE"))
        symbol: Mapped[str] = mapped_column(String(20), nullable=False)
        captured_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
        timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
        indicators: Mapped[dict] = mapped_column(JSONB, nullable=False)
        regime: Mapped[str] = mapped_column(String(30), nullable=False)
        raw_candles: Mapped[dict | None] = mapped_column(JSONB)

        trade: Mapped[Trade | None] = relationship(back_populates="snapshots")


    class NoTradeLog(Base):
        __tablename__ = "no_trade_log"

        log_id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
        symbol: Mapped[str] = mapped_column(String(20), nullable=False)
        logged_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
        regime: Mapped[str | None] = mapped_column(String(30))
        lstm_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
        confluence_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
        gate_failed: Mapped[str] = mapped_column(String(100), nullable=False)
        indicator_state: Mapped[dict | None] = mapped_column(JSONB)


    class StrategyPerformance(Base):
        __tablename__ = "strategy_performance"
        __table_args__ = (UniqueConstraint("strategy_name", "regime", "period_start", name="uq_strategy_regime_period"),)

        perf_id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
        strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
        regime: Mapped[str] = mapped_column(String(30), nullable=False)
        period_start: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
        period_end: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)
        total_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
        wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
        losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
        win_rate: Mapped[float | None] = mapped_column(Numeric(5, 4))
        avg_r_multiple: Mapped[float | None] = mapped_column(Numeric(10, 4))
        profit_factor: Mapped[float | None] = mapped_column(Numeric(10, 4))
        max_drawdown: Mapped[float | None] = mapped_column(Numeric(10, 4))
        updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())


    class EquitySnapshot(Base):
        __tablename__ = "equity_snapshots"

        snapshot_id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
        captured_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
        balance_usdt: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
        open_pnl: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False, default=0)
        total_equity: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
        peak_equity: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
        drawdown_pct: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)


    class SystemEvent(Base):
        __tablename__ = "system_events"
        __table_args__ = (CheckConstraint("severity IN ('INFO','WARNING','CRITICAL')", name="ck_events_severity"),)

        event_id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
        event_type: Mapped[str] = mapped_column(String(50), nullable=False)
        severity: Mapped[str] = mapped_column(String(10), nullable=False)
        message: Mapped[str] = mapped_column(Text, nullable=False)
        context: Mapped[dict | None] = mapped_column(JSONB)
        occurred_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

else:
    Base = object  # type: ignore[assignment]
