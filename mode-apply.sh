#!/usr/bin/env bash
# Non-interactive mode apply. Called ONLY by mode-watcher after the API has
# already validated the typed confirmation + gate. Not for direct live use.
set -euo pipefail
cd /home/ubuntu/crypto-trading-boot
ENVF=.env
set_kv(){ if grep -q "^$1=" "$ENVF"; then sed -i "s|^$1=.*|$1=$2|" "$ENVF"; else echo "$1=$2">>"$ENVF"; fi; }
case "${1:-}" in
  demo)
    set_kv USE_TESTNET true;  set_kv MARKET_TYPE spot;    set_kv LIVE_TRADING_REVIEWED false ;;
  live)
    grep -qE "^BINANCE_FUTURES_API_KEY=.+" "$ENVF" || { echo "abort: no futures key"; exit 1; }
    set_kv USE_TESTNET false; set_kv MARKET_TYPE futures; set_kv LIVE_TRADING_REVIEWED true ;;
  *) echo "usage: mode-apply.sh [demo|live]"; exit 1 ;;
esac
sudo docker compose up -d --force-recreate app
echo "$(date -u) applied mode=$1"
