#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say(){ echo -e "\n=== $* ==="; }

say "1/5 Backend patches"
python3 - <<'PYEOF'
from pathlib import Path
import ast
ee = Path("trading/execution_engine.py"); t = ee.read_text(encoding="utf-8")
t = t.replace(
'''            for symbol, managed in list(self.open_trades.items()):
                results.append(await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity))
                self.open_trades.pop(symbol, None)
                if self.journal:
                    await self.journal.log_trade_exit(symbol, Decimal("0"), "MANUAL")''',
'''            for symbol, managed in list(self.open_trades.items()):
                price = await self._latest_price(symbol)
                results.append(await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity))
                self.open_trades.pop(symbol, None)
                if self.journal:
                    exit_price = price if price > 0 else managed.plan.entry_price
                    await self.journal.log_trade_exit(symbol, exit_price, "MANUAL")''', 1)
t = t.replace(
'''            result = await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity)
            if result.accepted:
                self.open_trades.pop(symbol, None)
                if self.journal:
                    await self.journal.log_trade_exit(symbol, Decimal("0"), "MANUAL")''',
'''            price = await self._latest_price(symbol)
            result = await self._close_quantity(symbol, managed.plan.direction or "", managed.remaining_quantity)
            if result.accepted:
                self.open_trades.pop(symbol, None)
                if self.journal:
                    exit_price = price if price > 0 else managed.plan.entry_price
                    await self.journal.log_trade_exit(symbol, exit_price, "MANUAL")''', 1)
ee.write_text(t, encoding="utf-8"); ast.parse(t)

m = Path("main.py"); u = m.read_text(encoding="utf-8")
if "if symbol in self.execution.open_trades:" not in u:
    u = u.replace(
'''                    for symbol in settings.symbols:
                        candles_by_tf = snapshot["candles"].get(symbol, {})''',
'''                    for symbol in settings.symbols:
                        if symbol in self.execution.open_trades:
                            continue
                        candles_by_tf = snapshot["candles"].get(symbol, {})''', 1)
if "_process_close_requests" not in u:
    u = u.replace("                    await self._write_equity_snapshot()",
                  "                    await self._process_close_requests()\n                    await self._write_equity_snapshot()", 1)
    u = u.replace(
'''    async def learning_tick(self) -> None:
        await self.learning.record_state()''',
'''    async def learning_tick(self) -> None:
        await self.learning.record_state()

    async def _process_close_requests(self) -> None:
        path = settings.runtime_dir / "close_requests.json"
        if not path.exists():
            return
        try:
            symbols = json.loads(path.read_text(encoding="utf-8")).get("symbols", [])
        except Exception:
            return
        if not symbols:
            return
        for symbol in symbols:
            try:
                result = await self.execution.emergency_close_symbol(symbol)
                await self.journal.log_event("MANUAL_CLOSE", "INFO", f"Manual stop requested for {symbol}", {"accepted": result.accepted, "reason": result.reason})
            except Exception as exc:
                logger.warning(f"Manual close failed for {symbol}: {exc}")
        try:
            path.write_text(json.dumps({"symbols": []}), encoding="utf-8")
        except Exception:
            pass''', 1)
m.write_text(u, encoding="utf-8"); ast.parse(u)

s = Path("api/server.py"); a = s.read_text(encoding="utf-8")
if "CLOSE_REQUESTS_PATH" not in a:
    a = a.replace('RISK_OVERRIDE_PATH = settings.runtime_dir / "risk_overrides.json"',
                  'RISK_OVERRIDE_PATH = settings.runtime_dir / "risk_overrides.json"\nCLOSE_REQUESTS_PATH = settings.runtime_dir / "close_requests.json"', 1)
    a = a.replace(
'''        "trades": await _safe_fetch_all(db, "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 50"),
        "open_positions": open_positions,''',
'''        "trades": _trades,
        "open_positions": open_positions,''', 1)
    a = a.replace(
'''    await _enrich_open_positions(open_positions)
    return {''',
'''    await _enrich_open_positions(open_positions)
    _trades = await _safe_fetch_all(db, "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 50")
    await _enrich_open_positions([t for t in _trades if str(t.get("outcome")) == "OPEN"])
    return {''', 1)
    a += '''

@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper()
    current = _read_json(CLOSE_REQUESTS_PATH, {"symbols": []})
    pending = set(current.get("symbols", []))
    pending.add(symbol)
    CLOSE_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLOSE_REQUESTS_PATH.write_text(json.dumps({"symbols": sorted(pending)}), encoding="utf-8")
    return {"ok": True, "queued": symbol}
'''
    s.write_text(a, encoding="utf-8")
ast.parse(Path("api/server.py").read_text(encoding="utf-8"))
print("  backend OK")
PYEOF

say "2/5 Frontend patches"
python3 - <<'PYEOF'
from pathlib import Path
api = Path("frontend/lib/api.ts"); s = api.read_text(encoding="utf-8")
if "unrealized_pnl" not in s:
    s = s.replace("  outcome: string;\n  entry_time: string;\n  exit_time?: string | null;\n};",
"""  outcome: string;
  entry_time: string;
  exit_time?: string | null;
  current_price?: number | string | null;
  market_value?: number | string | null;
  unrealized_pnl?: number | string | null;
  unrealized_pct?: number | string | null;
};""", 1)
if "closePosition" not in s:
    s = s.replace('  health: () => request<TradingState>("/api/health"),',
'  health: () => request<TradingState>("/api/health"),\n  closePosition: (symbol: string) =>\n    request<{ ok: boolean; queued: string }>(`/api/positions/${symbol}/close`, { method: "POST" }),', 1)
api.write_text(s, encoding="utf-8")

pg = Path("frontend/app/page.tsx"); p = pg.read_text(encoding="utf-8")
if "stopping" not in p:
    p = p.replace("  const [savingRisk, setSavingRisk] = useState(false);",
                  "  const [savingRisk, setSavingRisk] = useState(false);\n  const [stopping, setStopping] = useState<Record<string, boolean>>({});", 1)
    p = p.replace(
"""    } finally {
      setSavingRisk(false);
    }
  };""",
"""    } finally {
      setSavingRisk(false);
    }
  };

  const handleStop = async (symbol: string) => {
    setStopping((s) => ({ ...s, [symbol]: true }));
    try {
      await api.closePosition(symbol);
      setActionError(null);
      await refresh();
    } catch {
      setActionError(`Could not stop ${symbol} — is the backend reachable?`);
    } finally {
      setStopping((s) => ({ ...s, [symbol]: false }));
    }
  };""", 1)
    p = p.replace("                      <th>Unreal. P&amp;L</th>\n                      <th>Opened</th>",
                  "                      <th>Unreal. P&amp;L</th>\n                      <th>Opened</th>\n                      <th>Action</th>", 1)
    p = p.replace(
"""                          <td>{fmtTime(p.entry_time)}</td>
                        </tr>
                      );
                    })}""",
"""                          <td>{fmtTime(p.entry_time)}</td>
                          <td>
                            <button className="btn btn--danger" disabled={!!stopping[p.symbol]} onClick={() => handleStop(p.symbol)}>
                              {stopping[p.symbol] ? "Stopping…" : "Stop"}
                            </button>
                          </td>
                        </tr>
                      );
                    })}""", 1)
    p = p.replace(
'''                        <td>{fmt(t.entry_price, 4)}</td>
                        <td>{t.exit_price !== null && t.exit_price !== undefined ? fmt(t.exit_price, 4) : "—"}</td>
                        <td className={num(t.pnl_usd) >= 0 ? "up" : "down"}>{fmtSigned(t.pnl_usd)}</td>''',
'''                        <td>{fmt(t.entry_price, 4)}</td>
                        <td>
                          {t.outcome === "OPEN" && t.current_price != null
                            ? fmt(t.current_price, 4)
                            : t.exit_price !== null && t.exit_price !== undefined
                            ? fmt(t.exit_price, 4)
                            : "—"}
                        </td>
                        {t.outcome === "OPEN" && t.unrealized_pnl != null ? (
                          <td className={num(t.unrealized_pnl) >= 0 ? "up" : "down"}>{fmtSigned(t.unrealized_pnl)}</td>
                        ) : (
                          <td className={num(t.pnl_usd) >= 0 ? "up" : "down"}>{fmtSigned(t.pnl_usd)}</td>
                        )}''', 1)
    pg.write_text(p, encoding="utf-8")
print("  frontend OK")
PYEOF

say "3/5 Rebuild backend"
sudo docker compose up -d --build
for i in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:10000/api/health)" = "200" ] && break; sleep 5; done

say "4/5 Rebuild frontend"
cd ~/crypto-trading-boot/frontend && npm run build && sudo systemctl restart crypto-frontend
cd ~/crypto-trading-boot

say "5/5 Verify"
curl -s -o /dev/null -w 'nginx /api/health: %{http_code}\n' http://127.0.0.1/api/health
curl -s -o /dev/null -w 'frontend: %{http_code}\n' http://127.0.0.1:3000/
echo "Done. Open: http://$(curl -s http://checkip.amazonaws.com)"
