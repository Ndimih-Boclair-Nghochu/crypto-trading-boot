#!/usr/bin/env bash
# Polls for a mode-switch request written by the API (dashboard toggle) into the
# host-mounted logs volume, then applies it. Runs as a systemd service.
set -uo pipefail
DIR=/home/ubuntu/crypto-trading-boot
REQ="$DIR/logs/mode_request.json"
LOG="$DIR/logs/mode-watcher.log"
echo "$(date -u) mode-watcher started" >> "$LOG"
while true; do
  if [ -f "$REQ" ]; then
    target=$(python3 -c "import json;print(json.load(open('$REQ')).get('target',''))" 2>/dev/null || echo "")
    rm -f "$REQ"
    case "$target" in
      demo|live)
        echo "$(date -u) request: $target" >> "$LOG"
        "$DIR/mode-apply.sh" "$target" >> "$LOG" 2>&1 || echo "$(date -u) apply failed" >> "$LOG"
        ;;
      *) echo "$(date -u) ignored request target='$target'" >> "$LOG" ;;
    esac
  fi
  sleep 5
done
