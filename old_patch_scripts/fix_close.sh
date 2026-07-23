#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say(){ echo -e "\n=== $* ==="; }

say "1/6 Patch _close_quantity + add get_asset_free"
python3 - <<'PYEOF'
from pathlib import Path
import ast
bc = Path("utils/binance_client.py"); s = bc.read_text(encoding="utf-8")
anchor = '                return Decimal(str(balance.get("free", "0")))\n        return Decimal("0")\n\n    @safe_api_call(lambda: OrderResult(False, reason="order API failed"))\n    async def place_order'
addition = ('                return Decimal(str(balance.get("free", "0")))\n        return Decimal("0")\n\n'
 '    @safe_api_call(lambda: Decimal("0"))\n'
 '    async def get_asset_free(self, asset: str) -> Decimal:\n'
 '        if not self.client:\n'
 '            return Decimal("0")\n'
 '        await self._throttle_if_needed()\n'
 '        account = await self.client.get_account()\n'
 '        self._capture_client_response_headers()\n'
 '        for balance in account.get("balances", []):\n'
 '            if balance.get("asset") == asset:\n'
 '                return Decimal(str(balance.get("free", "0")))\n'
 '        return Decimal("0")\n\n'
 '    @safe_api_call(lambda: OrderResult(False, reason="order API failed"))\n    async def place_order')
if "async def get_asset_free" not in s:
    assert anchor in s, "binance_client anchor missing"
    bc.write_text(s.replace(anchor, addition, 1), encoding="utf-8")
ast.parse(bc.read_text(encoding="utf-8"))

ee = Path("trading/execution_engine.py"); t = ee.read_text(encoding="utf-8")
old = ('    async def _close_quantity(self, symbol: str, direction: str, quantity: Decimal) -> ExecutionResult:\n'
 '        if quantity <= 0:\n'
 '            return ExecutionResult(False, reason="quantity <= 0")\n'
 '        filters = await self.client.get_symbol_filters(symbol)\n'
 '        if filters:\n'
 '            quantity = filters.round_quantity(quantity)\n'
 '            if filters.min_qty > 0 and quantity < filters.min_qty:\n'
 '                return ExecutionResult(False, status="FILTER_REJECTED", reason=f"quantity {quantity} below minQty {filters.min_qty}")\n'
 '        side = "SELL" if direction == "LONG" else "BUY"\n'
 '        result = await self.client.place_order(symbol=symbol, side=side, type="MARKET", quantity=_fmt(quantity))\n'
 '        return ExecutionResult(result.accepted, result.order_id, result.status, result.reason, result.raw)')
new = ('    async def _close_quantity(self, symbol: str, direction: str, quantity: Decimal) -> ExecutionResult:\n'
 '        if quantity <= 0:\n'
 '            return ExecutionResult(False, reason="quantity <= 0")\n'
 '        try:\n'
 '            for order in await self.client.get_open_orders(symbol):\n'
 '                oid = order.get("orderId")\n'
 '                if oid is not None:\n'
 '                    await self.client.cancel_order(symbol, str(oid))\n'
 '        except Exception as exc:\n'
 '            logger.warning(f"{symbol}: could not cancel resting orders before close: {exc}")\n'
 '        side = "SELL" if direction == "LONG" else "BUY"\n'
 '        if side == "SELL":\n'
 '            base = symbol[:-4] if symbol.endswith("USDT") else symbol\n'
 '            free = await self.client.get_asset_free(base)\n'
 '            if free < quantity:\n'
 '                quantity = free\n'
 '        filters = await self.client.get_symbol_filters(symbol)\n'
 '        if filters:\n'
 '            quantity = filters.round_quantity(quantity)\n'
 '            if filters.min_qty > 0 and quantity < filters.min_qty:\n'
 '                return ExecutionResult(False, status="FILTER_REJECTED", reason=f"quantity {quantity} below minQty {filters.min_qty}")\n'
 '        if quantity <= 0:\n'
 '            return ExecutionResult(False, reason="no free balance to close")\n'
 '        result = await self.client.place_order(symbol=symbol, side=side, type="MARKET", quantity=_fmt(quantity))\n'
 '        return ExecutionResult(result.accepted, result.order_id, result.status, result.reason, result.raw)')
if "could not cancel resting orders" not in t:
    assert old in t, "execution_engine anchor missing"
    ee.write_text(t.replace(old, new, 1), encoding="utf-8")
ast.parse(ee.read_text(encoding="utf-8"))
print("  patched OK")
PYEOF

say "2/6 Rebuild"
sudo docker compose up -d --build
sleep 20

say "3/6 Cancel all resting orders + flatten stray coin holdings back to USDT"
sudo docker compose exec -T app python - <<'PYEOF'
import asyncio, math
from binance import AsyncClient
from config import settings
BASES = [s[:-4] for s in settings.symbols if s.endswith("USDT")]
async def main():
    c = AsyncClient(settings.binance_api_key, settings.binance_secret, testnet=settings.use_testnet)
    try:
        for sym in settings.symbols:
            for o in await c.get_open_orders(symbol=sym):
                try: await c.cancel_order(symbol=sym, orderId=o["orderId"])
                except Exception as e: print("cancel skip", sym, e)
        acct = await c.get_account()
        free = {b["asset"]: float(b["free"]) for b in acct["balances"]}
        for base in BASES:
            sym = base + "USDT"
            amt = free.get(base, 0.0)
            info = await c.get_symbol_info(sym)
            step = float(next(f["stepSize"] for f in info["filters"] if f["filterType"]=="LOT_SIZE"))
            minq = float(next(f["minQty"] for f in info["filters"] if f["filterType"]=="LOT_SIZE"))
            qty = round(math.floor(amt/step)*step, 8)
            # only dump clearly bot-sized leftovers, not tiny dust
            if qty >= minq and qty*float((await c.get_symbol_ticker(symbol=sym))["price"]) > 10:
                try:
                    r = await c.order_market_sell(symbol=sym, quantity=qty)
                    print("sold", sym, r.get("executedQty"), "-> USDT", r.get("cummulativeQuoteQty"))
                except Exception as e:
                    print("sell skip", sym, e)
        usdt = next((b["free"] for b in (await c.get_account())["balances"] if b["asset"]=="USDT"), "0")
        print("USDT now:", usdt)
    finally:
        await c.close_connection()
asyncio.run(main())
PYEOF

say "4/6 Drop phantom trades + stale equity history"
sudo docker compose exec -T postgres psql -U botuser -d crypto_bot \
  -c "DELETE FROM trades WHERE exit_price = 0;" \
  -c "DELETE FROM equity_snapshots;"

say "5/6 Restart so drawdown peak re-baselines"
sudo docker compose restart app
sleep 40

say "6/6 Verify"
curl -s http://127.0.0.1/api/health; echo
sudo docker compose exec -T postgres psql -U botuser -d crypto_bot \
  -c "SELECT round(total_equity,2) AS equity, round(drawdown_pct,2) AS dd FROM equity_snapshots ORDER BY captured_at DESC LIMIT 1;" \
  -c "SELECT outcome, count(*), round(sum(pnl_usd),2) AS pnl FROM trades GROUP BY outcome;"
