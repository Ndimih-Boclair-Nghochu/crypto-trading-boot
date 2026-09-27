#!/usr/bin/env bash
# Deliberate DEMO <-> LIVE switch for the NBN trading bot.
#   ./mode.sh status   -> show current mode
#   ./mode.sh demo     -> safe testnet/spot paper money (default, always reversible)
#   ./mode.sh live     -> REAL MONEY futures (guarded: key check + typed confirmation)
set -euo pipefail
cd "$(dirname "$0")"
ENVF=.env

set_kv() { # key value : replace or append in .env
  if grep -q "^$1=" "$ENVF"; then sed -i "s|^$1=.*|$1=$2|" "$ENVF"; else echo "$1=$2" >> "$ENVF"; fi
}

status() {
  echo "== .env flags =="
  grep -E "^USE_TESTNET|^MARKET_TYPE|^LIVE_TRADING_REVIEWED" "$ENVF" || true
  echo "== live /api/mode =="
  curl -s http://localhost:10000/api/mode || true
  echo
}

case "${1:-status}" in
  status) status ;;

  demo)
    echo ">> Switching to DEMO (testnet spot, paper money -- always safe)..."
    set_kv USE_TESTNET true
    set_kv MARKET_TYPE spot
    set_kv LIVE_TRADING_REVIEWED false
    sudo docker compose up -d --force-recreate app
    sleep 12; status
    echo ">> DEMO active. No real money at risk."
    ;;

  live)
    echo "************************************************************"
    echo "*  LIVE mode trades REAL MONEY on FUTURES, with leverage.  *"
    echo "*  Losses are real and can liquidate. The edge is UNPROVEN. *"
    echo "************************************************************"
    if ! grep -qE "^BINANCE_FUTURES_API_KEY=.+" "$ENVF" || ! grep -qE "^BINANCE_FUTURES_SECRET=.+" "$ENVF"; then
      echo "ABORT: futures API key/secret are not set in .env."; exit 1
    fi
    read -r -p 'Type exactly  GO LIVE  to confirm real-money trading: ' confirm
    if [ "$confirm" != "GO LIVE" ]; then echo "Aborted -- still in demo."; exit 1; fi
    set_kv USE_TESTNET false
    set_kv MARKET_TYPE futures
    set_kv LIVE_TRADING_REVIEWED true
    sudo docker compose up -d --force-recreate app
    sleep 12; status
    echo ">> LIVE requested. Watch the dashboard badge (should read LIVE - REAL MONEY) and the logs."
    echo ">> To revert instantly:  ./mode.sh demo"
    ;;

  *) echo "usage: $0 [status|demo|live]"; exit 1 ;;
esac
