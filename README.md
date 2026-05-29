# Crypto AI Trading System

Production-oriented autonomous crypto trading system for Binance testnet-first operation. It includes async market ingestion, technical analysis, AI/RL decision gates, risk controls, PostgreSQL journaling, learning jobs, execution monitoring, and a Streamlit dashboard.

This code is financial automation software, not a profit guarantee. It defaults to Binance Testnet and refuses live trading unless `TESTNET_TRADE_COUNT >= 100` and `LIVE_TRADING_REVIEWED=true`.

## Option A: Docker

1. Install Docker Desktop.
2. Copy `.env.example` to `.env` and fill in Binance API keys. PostgreSQL defaults already match `docker-compose.yml`.
3. Run `docker-compose up -d postgres`.
4. Run `docker-compose run bot python main.py --mode=migrate`.
5. Run `docker-compose run bot python main.py --mode=download_data`.
6. Run `docker-compose run bot python main.py --mode=train_models`.
7. Run `docker-compose up`.
8. Open `http://localhost:8501` and toggle trading ON for testnet trading.
9. After 100+ testnet trades and manual review, set `USE_TESTNET=false`, `TESTNET_TRADE_COUNT=100`, and `LIVE_TRADING_REVIEWED=true`.

## Option B: Manual Setup

1. Install PostgreSQL 15+ and create a database named `crypto_bot`.
2. Install Python 3.11+.
3. Run `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and fill in `BINANCE_API_KEY`, `BINANCE_SECRET`, and `DATABASE_URL`.
5. Run `alembic -c db/migrations/alembic.ini upgrade head`.
6. Run `python main.py --mode=download_data`.
7. Run `python main.py --mode=train_models`.
8. In a second terminal run `streamlit run dashboard.py`.
9. Run `python main.py`.

## Safety Model

- Every Binance call is wrapped and logged.
- Models fail closed to `NO_TRADE`.
- The confidence gate is mandatory before risk and execution.
- Circuit breakers are non-bypassable.
- PostgreSQL writes run through a background journal queue and cannot crash the trading loop.
- Live trading requires 100 recorded testnet trades plus an explicit manual review flag.

## Useful Commands

```bash
python main.py --mode=migrate
python main.py --mode=download_data
python main.py --mode=train_models
python main.py
pytest
```
