#!/usr/bin/env bash
# Runs the trading bot and the REST API side by side in the same container so
# they share the same filesystem (.runtime/trading_state.json, risk
# overrides, etc.) and the same Postgres database.
#
# The API binds to $PORT (set by Render) so it satisfies the platform's
# port-scan requirement and is what a separately-deployed frontend (e.g. on
# Vercel) talks to.
#
# The trading bot is supervised in a restart loop: if it crashes (bad API
# keys, transient network errors, etc.) it is relaunched automatically after
# a short delay, WITHOUT taking down the API/dashboard. This keeps the
# dashboard reachable (and able to show what went wrong) even while the bot
# is recovering.
set -uo pipefail

PORT="${PORT:-8000}"

supervise_bot() {
    while true; do
        echo "Starting trading bot (main.py)..."
        python main.py
        EXIT_CODE=$?
        echo "Trading bot exited with code ${EXIT_CODE}; restarting in 10s..."
        sleep 10
    done
}

supervise_bot &
BOT_SUPERVISOR_PID=$!

terminate() {
    kill "${BOT_SUPERVISOR_PID}" 2>/dev/null || true
}
trap terminate TERM INT

echo "Starting API (uvicorn) on 0.0.0.0:${PORT}..."
uvicorn api.server:app --host 0.0.0.0 --port "${PORT}"
API_EXIT_CODE=$?
terminate
exit "${API_EXIT_CODE}"
