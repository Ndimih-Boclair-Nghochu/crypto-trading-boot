#!/usr/bin/env bash
# Runs the trading bot and the Streamlit dashboard side by side in the same
# container so they share the same filesystem (.runtime/trading_state.json,
# risk overrides, etc.) and the same Postgres database.
#
# The dashboard binds to $PORT (set by Render) so it satisfies the platform's
# port-scan requirement and serves as the system's web frontend. The trading
# bot has no HTTP interface and runs as a background process.
set -euo pipefail

PORT="${PORT:-8501}"

echo "Starting trading bot (main.py)..."
python main.py &
BOT_PID=$!

echo "Starting dashboard (Streamlit) on 0.0.0.0:${PORT}..."
streamlit run dashboard.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false &
DASH_PID=$!

# If either process exits, bring down the whole container so the platform
# restarts it (and the other process) cleanly.
terminate() {
    kill "${BOT_PID}" "${DASH_PID}" 2>/dev/null || true
}
trap terminate TERM INT

wait -n "${BOT_PID}" "${DASH_PID}"
EXIT_CODE=$?
terminate
exit "${EXIT_CODE}"
