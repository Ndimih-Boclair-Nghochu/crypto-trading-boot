#!/usr/bin/env bash
# Runs the trading bot and the REST API side by side in the same container so
# they share the same filesystem (.runtime/trading_state.json, risk
# overrides, etc.) and the same Postgres database.
#
# The API binds to $PORT (set by Render) so it satisfies the platform's
# port-scan requirement and is what a separately-deployed frontend (e.g. on
# Vercel) talks to. The trading bot has no HTTP interface and runs as a
# background process.
set -euo pipefail

PORT="${PORT:-8000}"

echo "Starting trading bot (main.py)..."
python main.py &
BOT_PID=$!

echo "Starting API (uvicorn) on 0.0.0.0:${PORT}..."
uvicorn api.server:app --host 0.0.0.0 --port "${PORT}" &
API_PID=$!

# If either process exits, bring down the whole container so the platform
# restarts it (and the other process) cleanly.
terminate() {
    kill "${BOT_PID}" "${API_PID}" 2>/dev/null || true
}
trap terminate TERM INT

wait -n "${BOT_PID}" "${API_PID}"
EXIT_CODE=$?
terminate
exit "${EXIT_CODE}"
