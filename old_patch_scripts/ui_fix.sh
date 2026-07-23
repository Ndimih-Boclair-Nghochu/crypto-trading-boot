#!/usr/bin/env bash
set -euo pipefail
cd ~/crypto-trading-boot
say(){ echo -e "\n=== $* ==="; }

say "1/3 Open-positions columns + Stop button"
python3 - <<'PYEOF'
from pathlib import Path
pg = Path("frontend/app/page.tsx"); p = pg.read_text(encoding="utf-8")

hdr_old = """                      <th>Symbol</th>
                      <th>Dir</th>
                      <th>Entry</th>
                      <th>SL</th>
                      <th>TP1</th>
                      <th>Qty</th>
                      <th>Opened</th>"""
hdr_new = """                      <th>Symbol</th>
                      <th>Dir</th>
                      <th>Entry</th>
                      <th>Now</th>
                      <th>SL</th>
                      <th>TP1</th>
                      <th>Qty</th>
                      <th>Unreal. P&amp;L</th>
                      <th>Opened</th>
                      <th>Action</th>"""

row_old = """                      <tr key={p.trade_id ?? `${p.symbol}-${p.entry_time}`}>
                        <td>{p.symbol}</td>
                        <td className={p.direction?.toLowerCase() === "long" ? "up" : "down"}>{p.direction}</td>
                        <td>{fmt(p.entry_price, 4)}</td>
                        <td>{fmt(p.sl_price, 4)}</td>
                        <td>{fmt(p.tp1_price, 4)}</td>
                        <td>{fmt(p.quantity, 4)}</td>
                        <td>{fmtTime(p.entry_time)}</td>
                      </tr>"""
row_new = """                      <tr key={p.trade_id ?? `${p.symbol}-${p.entry_time}`}>
                        <td>{p.symbol}</td>
                        <td className={p.direction?.toLowerCase() === "long" ? "up" : "down"}>{p.direction}</td>
                        <td>{fmt(p.entry_price, 4)}</td>
                        <td>{p.current_price != null ? fmt(p.current_price, 4) : "—"}</td>
                        <td>{fmt(p.sl_price, 4)}</td>
                        <td>{fmt(p.tp1_price, 4)}</td>
                        <td>{fmt(p.quantity, 4)}</td>
                        <td className={p.unrealized_pnl != null ? (num(p.unrealized_pnl) >= 0 ? "up" : "down") : ""}>
                          {p.unrealized_pnl != null ? fmtSigned(p.unrealized_pnl) : "—"}
                        </td>
                        <td>{fmtTime(p.entry_time)}</td>
                        <td>
                          <button
                            className="btn btn--danger"
                            disabled={!!stopping[p.symbol]}
                            onClick={() => handleStop(p.symbol)}
                          >
                            {stopping[p.symbol] ? "Stopping…" : "Stop"}
                          </button>
                        </td>
                      </tr>"""

if "p.current_price" not in p:
    assert hdr_old in p, "open-positions header anchor not found"
    assert row_old in p, "open-positions row anchor not found"
    p = p.replace(hdr_old, hdr_new, 1).replace(row_old, row_new, 1)
    pg.write_text(p, encoding="utf-8")
    print("  open-positions patched")
else:
    print("  already patched")
PYEOF

say "2/3 Mobile-responsive CSS"
python3 - <<'PYEOF'
from pathlib import Path
css = Path("frontend/app/globals.css"); c = css.read_text(encoding="utf-8")
BLOCK = """

/* ============ Mobile responsiveness ============ */
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.table-wrap table { min-width: max-content; }
.btn { min-height: 38px; }
.btn--danger { white-space: nowrap; }

@media (max-width: 720px) {
  .ticker { flex-wrap: wrap; gap: 6px 10px; font-size: 12px; padding: 8px 10px; }
  .ticker__brand { width: 100%; margin-bottom: 2px; }
  .ticker__spacer { display: none; }
  .shell { padding: 12px; }
  .grid { grid-template-columns: repeat(2, 1fr) !important; gap: 10px !important; }
  .grid--two { grid-template-columns: 1fr !important; gap: 10px !important; }
  .metric__value { font-size: 20px; }
  h1 { font-size: 20px; }
  h2 { font-size: 16px; }
  table { font-size: 12px; }
  th, td { padding: 6px 8px; white-space: nowrap; }
  .btn { padding: 8px 12px; }
}

@media (max-width: 430px) {
  .grid { grid-template-columns: 1fr !important; }
  .ticker { font-size: 11px; }
  .metric__value { font-size: 18px; }
}
"""
if "Mobile responsiveness" not in c:
    css.write_text(c + BLOCK, encoding="utf-8")
    print("  mobile CSS appended")
else:
    print("  already present")
PYEOF

say "3/3 Rebuild frontend"
cd ~/crypto-trading-boot/frontend
npm run build
sudo systemctl restart crypto-frontend
sleep 6
curl -s -o /dev/null -w 'frontend: %{http_code}\n' http://127.0.0.1:3000/ || true
echo "Done. Hard-refresh the page (Ctrl+Shift+R). Open: http://$(curl -s http://checkip.amazonaws.com)"
