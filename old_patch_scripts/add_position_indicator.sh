#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say() { echo -e "\n=== $* ==="; }

say "1/3 Backend: enrich open positions with live price + P&L"
python3 - <<'PYEOF'
from pathlib import Path
import ast, sys
p = Path("api/server.py"); s = p.read_text(encoding="utf-8")
if "_enrich_open_positions" in s:
    print("  already patched"); sys.exit(0)
s = s.replace("import json\nfrom contextlib import asynccontextmanager",
              "import asyncio\nimport json\nfrom contextlib import asynccontextmanager", 1)
s = s.replace("from config import settings\n",
              "import requests\n\nfrom config import settings\n", 1)
helpers = '''def _fetch_prices(symbols: set[str]) -> dict[str, float]:
    """Public ticker prices on the same network the bot trades (testnet vs live)."""
    base = settings.binance_spot_base_url.rstrip("/")
    out: dict[str, float] = {}
    for symbol in symbols:
        try:
            resp = requests.get(f"{base}/api/v3/ticker/price", params={"symbol": symbol}, timeout=5)
            resp.raise_for_status()
            out[symbol] = float(resp.json()["price"])
        except Exception:
            continue
    return out


async def _enrich_open_positions(positions: list[dict[str, Any]]) -> None:
    symbols = {str(p.get("symbol")) for p in positions if p.get("symbol")}
    if not symbols:
        return
    prices = await asyncio.to_thread(_fetch_prices, symbols)
    for p in positions:
        price = prices.get(str(p.get("symbol")))
        if not price:
            continue
        try:
            entry = float(p.get("entry_price") or 0)
            qty = float(p.get("quantity") or 0)
        except (TypeError, ValueError):
            continue
        direction = str(p.get("direction", "")).upper()
        pnl = (price - entry) * qty if direction == "LONG" else (entry - price) * qty
        cost = entry * qty
        p["current_price"] = price
        p["market_value"] = price * qty
        p["unrealized_pnl"] = pnl
        p["unrealized_pct"] = (pnl / cost * 100.0) if cost else 0.0


@app.get("/api/overview")
'''
s = s.replace('@app.get("/api/overview")\n', helpers, 1)
old = ('async def overview() -> dict[str, Any]:\n    db = _db(app)\n    return {\n'
       '        "trades": await _safe_fetch_all(db, "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 50"),\n'
       '        "open_positions": await _safe_fetch_all(\n'
       '            db, "SELECT * FROM trades WHERE outcome = \'OPEN\' ORDER BY entry_time DESC"\n        ),')
new = ('async def overview() -> dict[str, Any]:\n    db = _db(app)\n'
       '    open_positions = await _safe_fetch_all(\n'
       '        db, "SELECT * FROM trades WHERE outcome = \'OPEN\' ORDER BY entry_time DESC"\n    )\n'
       '    await _enrich_open_positions(open_positions)\n    return {\n'
       '        "trades": await _safe_fetch_all(db, "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 50"),\n'
       '        "open_positions": open_positions,')
if old not in s: sys.exit("overview anchor not found")
s = s.replace(old, new, 1)
ast.parse(s); p.write_text(s, encoding="utf-8"); print("  OK api/server.py")
PYEOF
echo ">>> Now pull the two frontend file changes (git pull) OR edit them to match, then continue."

say "2/3 Rebuild backend container"
sudo docker compose up -d --build
sudo docker compose restart app

say "3/3 Rebuild frontend"
cd ~/crypto-trading-boot/frontend
npm run build
sudo systemctl restart crypto-frontend
echo "DONE."
