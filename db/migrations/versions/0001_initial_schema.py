from __future__ import annotations

from alembic import op

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE EXTENSION IF NOT EXISTS pgcrypto;

        CREATE TABLE trades (
            trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            direction VARCHAR(10) NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
            entry_price NUMERIC(20, 8) NOT NULL,
            exit_price NUMERIC(20, 8),
            sl_price NUMERIC(20, 8) NOT NULL,
            tp1_price NUMERIC(20, 8) NOT NULL,
            tp2_price NUMERIC(20, 8),
            quantity NUMERIC(20, 8) NOT NULL,
            entry_time TIMESTAMPTZ NOT NULL,
            exit_time TIMESTAMPTZ,
            pnl_usd NUMERIC(20, 8),
            pnl_pct NUMERIC(10, 4),
            r_multiple NUMERIC(10, 4),
            strategy_used VARCHAR(50) NOT NULL,
            regime_at_entry VARCHAR(30) NOT NULL,
            lstm_confidence NUMERIC(5, 4),
            rl_confidence NUMERIC(5, 4),
            confluence_score NUMERIC(5, 2),
            outcome VARCHAR(15) CHECK (outcome IN ('WIN', 'LOSS', 'BREAKEVEN', 'OPEN')),
            exit_reason VARCHAR(20) CHECK (exit_reason IN ('TP1','TP2','SL','TRAILING','TIME_STOP','MANUAL','CIRCUIT_BREAKER')),
            binance_order_id VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE market_snapshots (
            snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            trade_id UUID REFERENCES trades(trade_id) ON DELETE CASCADE,
            symbol VARCHAR(20) NOT NULL,
            captured_at TIMESTAMPTZ NOT NULL,
            timeframe VARCHAR(5) NOT NULL,
            indicators JSONB NOT NULL,
            regime VARCHAR(30) NOT NULL,
            raw_candles JSONB
        );

        CREATE TABLE no_trade_log (
            log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            regime VARCHAR(30),
            lstm_confidence NUMERIC(5, 4),
            confluence_score NUMERIC(5, 2),
            gate_failed VARCHAR(100) NOT NULL,
            indicator_state JSONB
        );

        CREATE TABLE strategy_performance (
            perf_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            strategy_name VARCHAR(50) NOT NULL,
            regime VARCHAR(30) NOT NULL,
            period_start TIMESTAMPTZ NOT NULL,
            period_end TIMESTAMPTZ NOT NULL,
            total_trades INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,
            win_rate NUMERIC(5, 4),
            avg_r_multiple NUMERIC(10, 4),
            profit_factor NUMERIC(10, 4),
            max_drawdown NUMERIC(10, 4),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (strategy_name, regime, period_start)
        );

        CREATE TABLE equity_snapshots (
            snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            balance_usdt NUMERIC(20, 8) NOT NULL,
            open_pnl NUMERIC(20, 8) NOT NULL DEFAULT 0,
            total_equity NUMERIC(20, 8) NOT NULL,
            peak_equity NUMERIC(20, 8) NOT NULL,
            drawdown_pct NUMERIC(10, 4) NOT NULL
        );

        CREATE TABLE system_events (
            event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            event_type VARCHAR(50) NOT NULL,
            severity VARCHAR(10) NOT NULL CHECK (severity IN ('INFO','WARNING','CRITICAL')),
            message TEXT NOT NULL,
            context JSONB,
            occurred_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX idx_trades_symbol_time ON trades(symbol, entry_time DESC);
        CREATE INDEX idx_trades_outcome ON trades(outcome);
        CREATE INDEX idx_trades_strategy ON trades(strategy_used, regime_at_entry);
        CREATE INDEX idx_snapshots_trade ON market_snapshots(trade_id);
        CREATE INDEX idx_equity_time ON equity_snapshots(captured_at DESC);
        CREATE INDEX idx_no_trade_symbol ON no_trade_log(symbol, logged_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS system_events;
        DROP TABLE IF EXISTS equity_snapshots;
        DROP TABLE IF EXISTS strategy_performance;
        DROP TABLE IF EXISTS no_trade_log;
        DROP TABLE IF EXISTS market_snapshots;
        DROP TABLE IF EXISTS trades;
        """
    )
