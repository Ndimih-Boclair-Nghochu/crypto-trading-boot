#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say(){ echo -e "\n=== $* ==="; }

say "1/5 Apply code patches (notional cap + learning cadence)"
python3 - <<'PYEOF'
from pathlib import Path
import ast
c = Path("config.py"); s = c.read_text(encoding="utf-8")
anc = '    max_portfolio_risk_pct: float = field(default_factory=lambda: _float("MAX_PORTFOLIO_RISK_PCT", 3.0))'
if "max_position_pct" not in s:
    s = s.replace(anc, anc + '\n    max_position_pct: float = field(default_factory=lambda: _float("MAX_POSITION_PCT", 5.0))', 1)
    c.write_text(s, encoding="utf-8")
ast.parse(c.read_text(encoding="utf-8"))
r = Path("trading/risk_manager.py"); t = r.read_text(encoding="utf-8")
old = ('        else:\n            kelly_quantity = (balance * half_kelly) / entry\n'
 '        return min(fixed_fractional, atr_based, kelly_quantity).quantize(Decimal("0.00000001"))')
new = ('        else:\n            kelly_quantity = (balance * half_kelly) / entry\n'
 '        # Hard cap per-trade notional to a small fraction of equity, so many\n'
 '        # positions can run concurrently and no single trade can drain capital.\n'
 '        max_pos_fraction = _d(getattr(self.settings, "max_position_pct", 5.0)) / Decimal("100")\n'
 '        capped = (balance * max_pos_fraction) / entry if entry > 0 else Decimal("0")\n'
 '        return min(fixed_fractional, atr_based, kelly_quantity, capped).quantize(Decimal("0.00000001"))')
if "max_pos_fraction" not in t:
    assert old in t, "risk_manager anchor missing"; r.write_text(t.replace(old,new,1), encoding="utf-8")
ast.parse(r.read_text(encoding="utf-8"))
m = Path("main.py"); u = m.read_text(encoding="utf-8")
if "self._learning_at" not in u:
    u = u.replace("        self._reconcile_at = datetime.now(UTC)\n",
                  "        self._reconcile_at = datetime.now(UTC)\n        self._learning_at = datetime.now(UTC)\n", 1)
    old_rec = ('                    if datetime.now(UTC) >= self._reconcile_at:\n'
 '                        await self.execution.reconcile_orders()\n'
 '                        self._reconcile_at = datetime.now(UTC) + timedelta(minutes=5)')
    new_rec = old_rec + ('\n                    if datetime.now(UTC) >= self._learning_at:\n'
 '                        await self.learning.run()\n'
 '                        self._learning_at = datetime.now(UTC) + timedelta(seconds=60)')
    assert old_rec in u, "main.py anchor missing"; m.write_text(u.replace(old_rec,new_rec,1), encoding="utf-8")
ast.parse(m.read_text(encoding="utf-8"))
print("  patched OK")
PYEOF

say "2/5 Tune risk knobs in .env (high volume, tiny per-trade risk)"
setenv(){ grep -q "^$1=" .env && sed -i "s|^$1=.*|$1=$2|" .env || echo "$1=$2" >> .env; }
setenv MAX_RISK_PER_TRADE_PCT 0.5
setenv MAX_CONCURRENT_TRADES 15
setenv MAX_PORTFOLIO_RISK_PCT 8
setenv MAX_POSITION_PCT 4
setenv DRAWDOWN_CIRCUIT_BREAKER_PCT 25
setenv LAB_MODE true
grep -E "^(MAX_|LAB_|DRAWDOWN)" .env

say "3/5 Rebuild"
sudo docker compose up -d --build
echo "waiting for API..."
for i in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:10000/api/health)" = "200" ] && { echo "up"; break; }
  sleep 5
done

say "4/5 Set gate: confidence 0.55, concurrent 15"
curl -s -X POST http://127.0.0.1/api/risk-settings -H 'Content-Type: application/json' \
 -d '{"max_risk_per_trade_pct":0.5,"max_daily_loss_pct":4,"max_weekly_loss_pct":8,"max_concurrent_trades":15,"confidence_threshold":0.55}'; echo

say "5/5 Verify"
curl -s http://127.0.0.1/api/health; echo
