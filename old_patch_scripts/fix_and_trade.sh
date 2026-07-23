#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say() { echo -e "\n=== $* ==="; }

say "1/7 Sanity: testnet + keys"
grep -q '^USE_TESTNET=true' .env || { echo "ABORT: USE_TESTNET is not true."; exit 1; }
grep -qE '^BINANCE_API_KEY=.+' .env || { echo "ABORT: BINANCE_API_KEY empty."; exit 1; }
echo "ok"

say "2/7 Build context guard"
cat > .dockerignore <<'EOF'
.git
frontend
logs
models/weights
.runtime
__pycache__
*.pyc
.env
EOF
echo "ok"

say "3/7 LAB_MODE patch"
cat > lab_mode_patch.py <<'PYEOF'
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parent
def patch(rel, reps):
    p = BASE / rel; src = p.read_text(encoding="utf-8")
    for old, new in reps:
        if new in src: continue
        if old not in src: sys.exit(f"FAILED anchor in {rel}: {old[:90]}")
        src = src.replace(old, new, 1)
    p.write_text(src, encoding="utf-8"); print("  patched", rel)
patch("config.py", [(
 '    testnet_trade_count: int = field(default_factory=lambda: _int("TESTNET_TRADE_COUNT", 0))',
 '    testnet_trade_count: int = field(default_factory=lambda: _int("TESTNET_TRADE_COUNT", 0))\n'
 '    lab_mode: bool = field(default_factory=lambda: _bool("LAB_MODE", False))')])
patch("models/confidence_gate.py", [
(' base_threshold = self._confidence_threshold()\n        threshold = base_threshold * 0.85 if extreme_regime else base_threshold'.lstrip(),
 'base_threshold = self._confidence_threshold()\n        threshold = base_threshold * 0.85 if extreme_regime else base_threshold\n'
 '\n        # LAB_MODE: testnet-only relaxation so the execution path can be\n'
 '        # exercised end to end. ANDed with use_testnet -- a loosened gate\n'
 '        # must never apply to real money.\n'
 '        lab = bool(getattr(self.settings, "lab_mode", False)) and self.settings.use_testnet\n'
 '        min_confluence = 35 if lab else 65\n'
 '        min_agreement = 2 if lab else 4'),
('        expected_action = "BUY" if direction == "LONG" else "SELL"\n'
 '        if rl_decision.action != expected_action:\n'
 '            reasons.append(f"RL action {rl_decision.action} does not match {expected_action}")',
 '        expected_action = "BUY" if direction == "LONG" else "SELL"\n'
 '        # A HOLD from an undertrained RL agent is tolerated in lab mode;\n'
 '        # an outright opposite call still vetoes.\n'
 '        rl_conflicts = (\n'
 '            rl_decision.action not in {expected_action, "HOLD"} if lab\n'
 '            else rl_decision.action != expected_action\n'
 '        )\n'
 '        if rl_conflicts:\n'
 '            reasons.append(f"RL action {rl_decision.action} does not match {expected_action}")'),
('        if float(confluence.get("score", 0) or 0) < 65:\n            reasons.append("multi-timeframe confluence below 65")',
 '        if float(confluence.get("score", 0) or 0) < min_confluence:\n            reasons.append(f"multi-timeframe confluence below {min_confluence}")'),
('        if agreement < 4:', '        if agreement < min_agreement:'),
])
print("patch done")
PYEOF
python3 lab_mode_patch.py
python3 -c "import ast;[ast.parse(open(f,encoding='utf-8').read()) for f in ['config.py','models/confidence_gate.py']];print('  syntax OK')"

say "4/7 Enable LAB_MODE"
grep -q '^LAB_MODE=' .env && sed -i 's/^LAB_MODE=.*/LAB_MODE=true/' .env || echo 'LAB_MODE=true' >> .env
grep '^LAB_MODE=' .env

say "5/7 Rebuild"
sudo docker compose up -d --build
echo "waiting for API..."
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:10000/api/health || true)
  [ "$code" = "200" ] && { echo "API up after ${i}0s"; break; }
  sleep 10
done
[ "$code" = "200" ] || { echo "ABORT: API never came up."; sudo docker compose logs --tail 60 app; exit 1; }

say "6/7 Model weights"
if ! ls models/weights/*.pt >/dev/null 2>&1 || [ ! -f models/weights/ppo_trading_agent.zip ]; then
  echo "Weights missing -> training now (several minutes)..."
  sudo docker compose exec -T app python main.py --mode download_data
  sudo docker compose exec -T app python main.py --mode train_models
  if ! ls models/weights/*.pt >/dev/null 2>&1; then
    echo "ABORT: training produced no weights. Relevant log lines:"
    sudo docker compose logs app 2>&1 | grep -iE "training failed|import|weights" | tail -30
    exit 1
  fi
  sudo docker compose restart app; sleep 30
fi
ls -la models/weights/

say "7/7 Loosen threshold"
curl -s -X POST http://127.0.0.1/api/risk-settings -H 'Content-Type: application/json' \
 -d '{"max_risk_per_trade_pct":1,"max_daily_loss_pct":4,"max_weekly_loss_pct":8,"max_concurrent_trades":3,"confidence_threshold":0.55}'
echo
curl -s http://127.0.0.1/api/health; echo
echo -e "\nDONE. Dashboard: http://$(curl -s http://checkip.amazonaws.com)"
