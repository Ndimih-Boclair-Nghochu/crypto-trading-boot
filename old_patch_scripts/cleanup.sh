#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say(){ echo -e "\n=== $* ==="; }

say "1/5 Remove any phantom (exit_price=0) trades"
sudo docker compose exec -T postgres psql -U botuser -d crypto_bot \
  -c "DELETE FROM trades WHERE exit_price = 0;" \
  -c "SELECT count(*) AS trades_left FROM trades;"

say "2/5 Sell the orphaned ADA back to USDT"
sudo docker compose exec -T app python - <<'PYEOF'
import asyncio, math
from binance import AsyncClient
from config import settings
async def main():
    c = AsyncClient(settings.binance_api_key, settings.binance_secret, testnet=settings.use_testnet)
    try:
        info = await c.get_symbol_info("ADAUSDT")
        step = float(next(f["stepSize"] for f in info["filters"] if f["filterType"]=="LOT_SIZE"))
        acct = await c.get_account()
        free = next((float(b["free"]) for b in acct["balances"] if b["asset"]=="ADA"), 0.0)
        qty = round(math.floor(free/step)*step, 8)
        print("ADA free:", free, "step:", step, "selling:", qty)
        if qty > 0:
            r = await c.order_market_sell(symbol="ADAUSDT", quantity=qty)
            print("sold:", r.get("status"), "executedQty:", r.get("executedQty"), "gotUSDT:", r.get("cummulativeQuoteQty"))
        else:
            print("nothing to sell")
        usdt = next((b["free"] for b in (await c.get_account())["balances"] if b["asset"]=="USDT"), "0")
        print("USDT now:", usdt)
    finally:
        await c.close_connection()
asyncio.run(main())
PYEOF

say "3/5 Clear stale equity history (removes fake peak/drawdown)"
sudo docker compose exec -T postgres psql -U botuser -d crypto_bot \
  -c "DELETE FROM equity_snapshots;"

say "4/5 Restart app so peak re-baselines to current equity"
sudo docker compose restart app
sleep 40

say "5/5 Verify"
curl -s http://127.0.0.1/api/health; echo
sudo docker compose exec -T postgres psql -U botuser -d crypto_bot \
  -c "SELECT round(balance_usdt,2) AS usdt, round(total_equity,2) AS equity, round(drawdown_pct,2) AS dd FROM equity_snapshots ORDER BY captured_at DESC LIMIT 2;" \
  -c "SELECT outcome, count(*), round(sum(pnl_usd),2) AS pnl FROM trades GROUP BY outcome;"
