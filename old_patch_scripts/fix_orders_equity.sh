#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say() { echo -e "\n=== $* ==="; }

say "1/4 Apply OCO + equity patches"
python3 - <<'PYEOF'
from pathlib import Path
import ast, sys
p1 = Path("utils/binance_client.py"); s1 = p1.read_text(encoding="utf-8")
old1 = ('        elif hasattr(self.client, "create_oco_order"):\n'
        '            raw = await self.client.create_oco_order(**_legacy_oco_kwargs(kwargs))')
new1 = ('        elif hasattr(self.client, "create_oco_order"):\n'
        '            # python-binance >=1.0.37 routes create_oco_order to the NEW\n'
        '            # POST /api/v3/orderList/oco endpoint, which REQUIRES aboveType/\n'
        '            # belowType. Legacy stopPrice params caused "APIError -1102:\n'
        '            # aboveType was not sent". Forward the new-style kwargs as-is.\n'
        '            raw = await self.client.create_oco_order(**kwargs)')
if new1 not in s1:
    if old1 not in s1: sys.exit("OCO anchor not found")
    s1 = s1.replace(old1, new1, 1); p1.write_text(s1, encoding="utf-8")
ast.parse(s1); print("  OK utils/binance_client.py")
p2 = Path("main.py"); s2 = p2.read_text(encoding="utf-8")
old2 = ('            balance = await self.client.get_usdt_balance()\n'
        '            open_pnl = Decimal("0")\n'
        '            for position in self.risk.open_positions.values():\n'
        '                current_price = await self._get_current_price(position.symbol)\n'
        '                if current_price <= 0:\n'
        '                    continue\n'
        '                if position.direction == "LONG":\n'
        '                    open_pnl += position.quantity * (current_price - position.entry_price)\n'
        '                else:\n'
        '                    open_pnl += position.quantity * (position.entry_price - current_price)\n'
        '            total_equity = Decimal(str(balance)) + open_pnl')
new2 = ('            balance = await self.client.get_usdt_balance()\n'
        '            # Equity = free USDT + market value of coins held from the bot''s OWN\n'
        '            # positions. The old formula added only unrealized PnL, omitting cost\n'
        '            # basis -- so a buy dropped equity by the full notional (phantom ~25%\n'
        '            # drawdown that latched the circuit breaker). Pre-funded testnet coins\n'
        '            # are excluded: only tracked positions count.\n'
        '            holdings_value = Decimal("0")\n'
        '            open_pnl = Decimal("0")\n'
        '            for position in self.risk.open_positions.values():\n'
        '                current_price = await self._get_current_price(position.symbol)\n'
        '                if current_price <= 0:\n'
        '                    continue\n'
        '                if position.direction == "LONG":\n'
        '                    holdings_value += position.quantity * current_price\n'
        '                    open_pnl += position.quantity * (current_price - position.entry_price)\n'
        '                else:\n'
        '                    holdings_value -= position.quantity * current_price\n'
        '                    open_pnl += position.quantity * (position.entry_price - current_price)\n'
        '            total_equity = Decimal(str(balance)) + holdings_value')
if new2 not in s2:
    if old2 not in s2: sys.exit("equity anchor not found")
    s2 = s2.replace(old2, new2, 1); p2.write_text(s2, encoding="utf-8")
ast.parse(s2); print("  OK main.py")
PYEOF

say "2/4 Rebuild"
sudo docker compose up -d --build

say "3/4 Restart app (clears in-memory peak_equity + latched breaker)"
sudo docker compose restart app
echo "waiting for API..."
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:10000/api/health || true)
  [ "$code" = "200" ] && { echo "API up"; break; }
  sleep 5
done

say "4/4 Health"
curl -s http://127.0.0.1/api/health; echo
