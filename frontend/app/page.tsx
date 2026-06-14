"use client";

import { useCallback, useEffect, useState } from "react";
import {
  api,
  Overview,
  RiskSettings,
  TradingState,
} from "../lib/api";
import { fmt, fmtSigned, fmtTime, num } from "../lib/format";
import { EquitySparkline } from "../components/EquitySparkline";

const POLL_MS = 10_000;

function periodPnl(trades: Overview["trades"], period: "day" | "week"): number {
  const now = new Date();
  return trades.reduce((sum, t) => {
    if (!t.exit_time || t.pnl_usd === null || t.pnl_usd === undefined) return sum;
    const exit = new Date(t.exit_time);
    if (Number.isNaN(exit.getTime())) return sum;
    if (period === "day") {
      if (exit.toDateString() !== now.toDateString()) return sum;
    } else {
      const startOfWeek = new Date(now);
      const dayIdx = (now.getUTCDay() + 6) % 7; // Monday = 0
      startOfWeek.setUTCDate(now.getUTCDate() - dayIdx);
      startOfWeek.setUTCHours(0, 0, 0, 0);
      if (exit < startOfWeek) return sum;
    }
    return sum + num(t.pnl_usd);
  }, 0);
}

export default function Page() {
  const [health, setHealth] = useState<TradingState | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [risk, setRisk] = useState<RiskSettings | null>(null);
  const [riskDraft, setRiskDraft] = useState<RiskSettings | null>(null);
  const [now, setNow] = useState(new Date());
  const [error, setError] = useState<string | null>(null);
  const [toggling, setToggling] = useState(false);
  const [savingRisk, setSavingRisk] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [h, o] = await Promise.all([api.health(), api.overview()]);
      setHealth(h);
      setOverview(o);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to reach the trading API");
    }
  }, []);

  useEffect(() => {
    refresh();
    api
      .riskSettings()
      .then((r) => {
        setRisk(r);
        setRiskDraft(r);
      })
      .catch(() => undefined);
    const dataTimer = setInterval(refresh, POLL_MS);
    const clockTimer = setInterval(() => setNow(new Date()), 1000);
    return () => {
      clearInterval(dataTimer);
      clearInterval(clockTimer);
    };
  }, [refresh]);

  const handleToggle = async () => {
    if (!health) return;
    setToggling(true);
    try {
      const updated = await api.toggleTrading(!health.trading_enabled);
      setHealth((prev) => (prev ? { ...prev, ...updated } : updated));
    } catch {
      setError("Could not update trading state");
    } finally {
      setToggling(false);
    }
  };

  const handleClose = async (symbol: string) => {
    try {
      await api.closePosition(symbol);
      refresh();
    } catch {
      setError(`Could not request close for ${symbol}`);
    }
  };

  const handleSaveRisk = async () => {
    if (!riskDraft) return;
    setSavingRisk(true);
    try {
      const saved = await api.saveRiskSettings(riskDraft);
      setRisk(saved);
    } catch {
      setError("Could not save risk settings");
    } finally {
      setSavingRisk(false);
    }
  };

  const trades = overview?.trades ?? [];
  const openPositions = overview?.open_positions ?? [];
  const equity = overview?.equity ?? [];
  const performance = overview?.performance ?? [];
  const events = overview?.events ?? [];
  const noTrade = overview?.no_trade ?? [];
  const symbols = overview?.symbols ?? [];

  const latestEquity = equity[0];
  const totalEquity = num(latestEquity?.total_equity);
  const peakEquity = num(latestEquity?.peak_equity);
  const drawdown = num(latestEquity?.drawdown_pct);
  const todayPnl = periodPnl(trades, "day");
  const weekPnl = periodPnl(trades, "week");
  const allPnl = trades.reduce((sum, t) => sum + num(t.pnl_usd), 0);

  // equity array is ordered newest-first; chart wants oldest-first.
  const sparkValues = [...equity].reverse().map((e) => num(e.total_equity));
  const equityTrendPositive = sparkValues.length > 1 ? sparkValues[sparkValues.length - 1] >= sparkValues[0] : true;

  const status = health?.status ?? "—";
  const connected = !error;
  const dotClass = !connected
    ? "ticker__dot--down"
    : health?.trading_enabled
    ? "ticker__dot--live"
    : "ticker__dot--paused";

  const watchlist: Array<Partial<Overview["no_trade"][number]> & { symbol: string }> =
    noTrade.length > 0
      ? Object.values(
          noTrade.reduce<Record<string, Overview["no_trade"][number]>>((acc, row) => {
            if (!acc[row.symbol]) acc[row.symbol] = row;
            return acc;
          }, {})
        ).slice(0, 6)
      : symbols.slice(0, 6).map((s) => ({ symbol: s }));

  return (
    <>
      <div className="ticker">
        <span className="ticker__brand">CRYPTO TRADING DESK</span>
        <span className="ticker__item">
          <span className={`ticker__dot ${dotClass}`} />
          {connected ? status : "API UNREACHABLE"}
        </span>
        <span className="ticker__item">
          UTC <strong>{now.toISOString().slice(11, 19)}</strong>
        </span>
        <span className="ticker__item">
          EQUITY <strong>{fmt(totalEquity)}</strong>
        </span>
        <EquitySparkline values={sparkValues} positive={equityTrendPositive} />
        <span className="ticker__item">
          {health?.testnet ? <span className="badge badge--amber">TESTNET</span> : <span className="badge badge--green">LIVE</span>}
        </span>
        <span className="ticker__spacer" />
        <button
          type="button"
          className="switch ticker__toggle"
          onClick={handleToggle}
          disabled={!health || toggling}
          aria-disabled={!health || toggling}
          aria-pressed={!!health?.trading_enabled}
        >
          <span className={`switch__track ${health?.trading_enabled ? "switch__track--on" : ""}`}>
            <span className="switch__thumb" />
          </span>
          {health?.trading_enabled ? "TRADING ON" : "TRADING OFF"}
        </button>
      </div>

      <div className="shell">
        <header className="page-head">
          <h1>Account overview</h1>
          <p>Live status, positions, and risk controls for the autonomous trading bot.</p>
          {error && <p className="footer-note down">{error} — retrying every {POLL_MS / 1000}s.</p>}
        </header>

        <section className="grid">
          <div className="card">
            <div className="metric__label">Total equity (USDT)</div>
            <div className="metric__value">{fmt(totalEquity)}</div>
            <div className="metric__sub">Peak {fmt(peakEquity)}</div>
          </div>
          <div className="card">
            <div className="metric__label">Today&rsquo;s P&amp;L</div>
            <div className={`metric__value ${todayPnl >= 0 ? "up" : "down"}`}>{fmtSigned(todayPnl)}</div>
          </div>
          <div className="card">
            <div className="metric__label">Week&rsquo;s P&amp;L</div>
            <div className={`metric__value ${weekPnl >= 0 ? "up" : "down"}`}>{fmtSigned(weekPnl)}</div>
          </div>
          <div className="card">
            <div className="metric__label">All-time P&amp;L</div>
            <div className={`metric__value ${allPnl >= 0 ? "up" : "down"}`}>{fmtSigned(allPnl)}</div>
          </div>
          <div className="card">
            <div className="metric__label">Drawdown</div>
            <div className="metric__value">{fmt(drawdown)}%</div>
            <div className="metric__sub">vs peak {fmt(peakEquity)}</div>
          </div>
        </section>

        <section className="grid--two">
          <div className="card">
            <h2>Open positions</h2>
            {openPositions.length === 0 ? (
              <p className="empty">No open positions.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Dir</th>
                      <th>Entry</th>
                      <th>SL</th>
                      <th>TP1</th>
                      <th>Qty</th>
                      <th>Opened</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {openPositions.map((p) => (
                      <tr key={p.trade_id ?? `${p.symbol}-${p.entry_time}`}>
                        <td>{p.symbol}</td>
                        <td className={p.direction?.toLowerCase() === "long" ? "up" : "down"}>{p.direction}</td>
                        <td>{fmt(p.entry_price, 4)}</td>
                        <td>{fmt(p.sl_price, 4)}</td>
                        <td>{fmt(p.tp1_price, 4)}</td>
                        <td>{fmt(p.quantity, 4)}</td>
                        <td>{fmtTime(p.entry_time)}</td>
                        <td>
                          <button type="button" className="btn btn--danger" onClick={() => handleClose(p.symbol)}>
                            Close
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="card">
            <h2>Market watchlist</h2>
            {watchlist.length === 0 ? (
              <p className="empty">No symbols configured.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Regime</th>
                      <th>LSTM conf.</th>
                      <th>Confluence</th>
                      <th>Gate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {watchlist.map((row) => (
                      <tr key={row.symbol}>
                        <td>{row.symbol}</td>
                        <td>{row.regime ?? "—"}</td>
                        <td>{row.lstm_confidence !== undefined ? fmt(row.lstm_confidence, 2) : "—"}</td>
                        <td>{row.confluence_score !== undefined ? fmt(row.confluence_score, 2) : "—"}</td>
                        <td>{row.gate_failed ?? "WAITING"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        <section className="grid--two">
          <div className="card">
            <h2>Trade history</h2>
            {trades.length === 0 ? (
              <p className="empty">No trades recorded yet.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Symbol</th>
                      <th>Dir</th>
                      <th>Entry</th>
                      <th>Exit</th>
                      <th>P&amp;L</th>
                      <th>R</th>
                      <th>Outcome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.slice(0, 12).map((t) => (
                      <tr key={t.trade_id ?? `${t.symbol}-${t.entry_time}`}>
                        <td>{fmtTime(t.entry_time)}</td>
                        <td>{t.symbol}</td>
                        <td className={t.direction?.toLowerCase() === "long" ? "up" : "down"}>{t.direction}</td>
                        <td>{fmt(t.entry_price, 4)}</td>
                        <td>{t.exit_price !== null && t.exit_price !== undefined ? fmt(t.exit_price, 4) : "—"}</td>
                        <td className={num(t.pnl_usd) >= 0 ? "up" : "down"}>{fmtSigned(t.pnl_usd)}</td>
                        <td>{t.r_multiple !== null && t.r_multiple !== undefined ? fmt(t.r_multiple, 2) : "—"}</td>
                        <td>{t.outcome}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="card">
            <h2>Strategy performance</h2>
            {performance.length === 0 ? (
              <p className="empty">No strategy performance yet.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th>Regime</th>
                      <th>Win rate</th>
                      <th>Trades</th>
                      <th>PF</th>
                      <th>Avg R</th>
                    </tr>
                  </thead>
                  <tbody>
                    {performance.map((p) => (
                      <tr key={`${p.strategy_name}-${p.regime ?? ""}`}>
                        <td>{p.strategy_name}</td>
                        <td>{p.regime ?? "—"}</td>
                        <td>
                          <div className="row" style={{ gap: 8 }}>
                            <span>{fmt(p.win_rate, 1)}%</span>
                            <span
                              style={{
                                flex: 1,
                                height: 4,
                                borderRadius: 2,
                                background: "var(--border)",
                                position: "relative",
                                minWidth: 40,
                              }}
                            >
                              <span
                                style={{
                                  position: "absolute",
                                  inset: 0,
                                  width: `${Math.min(100, Math.max(0, num(p.win_rate)))}%`,
                                  background: num(p.win_rate) >= 50 ? "var(--accent-green)" : "var(--accent-red)",
                                  borderRadius: 2,
                                }}
                              />
                            </span>
                          </div>
                        </td>
                        <td>{p.total_trades}</td>
                        <td>{p.profit_factor !== undefined ? fmt(p.profit_factor, 2) : "—"}</td>
                        <td>{p.avg_r_multiple !== undefined ? fmt(p.avg_r_multiple, 2) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        <section className="grid--two">
          <div className="card">
            <h2>Risk settings</h2>
            {riskDraft ? (
              <>
                <div className="field-grid">
                  <div className="field">
                    <label htmlFor="risk-pct">Risk per trade %</label>
                    <input
                      id="risk-pct"
                      type="number"
                      step="0.1"
                      min={0.1}
                      max={5}
                      value={riskDraft.max_risk_per_trade_pct}
                      onChange={(e) => setRiskDraft({ ...riskDraft, max_risk_per_trade_pct: parseFloat(e.target.value) })}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="daily-pct">Daily loss limit %</label>
                    <input
                      id="daily-pct"
                      type="number"
                      step="0.5"
                      min={1}
                      max={20}
                      value={riskDraft.max_daily_loss_pct}
                      onChange={(e) => setRiskDraft({ ...riskDraft, max_daily_loss_pct: parseFloat(e.target.value) })}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="weekly-pct">Weekly loss limit %</label>
                    <input
                      id="weekly-pct"
                      type="number"
                      step="0.5"
                      min={1}
                      max={30}
                      value={riskDraft.max_weekly_loss_pct}
                      onChange={(e) => setRiskDraft({ ...riskDraft, max_weekly_loss_pct: parseFloat(e.target.value) })}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="max-trades">Max concurrent trades</label>
                    <input
                      id="max-trades"
                      type="number"
                      step="1"
                      min={1}
                      max={10}
                      value={riskDraft.max_concurrent_trades}
                      onChange={(e) => setRiskDraft({ ...riskDraft, max_concurrent_trades: parseInt(e.target.value, 10) })}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="confidence">Confidence threshold</label>
                    <input
                      id="confidence"
                      type="number"
                      step="0.01"
                      min={0.5}
                      max={0.99}
                      value={riskDraft.confidence_threshold}
                      onChange={(e) => setRiskDraft({ ...riskDraft, confidence_threshold: parseFloat(e.target.value) })}
                    />
                  </div>
                </div>
                <div className="row">
                  <span className="footer-note">Takes effect on the next trade cycle.</span>
                  <button type="button" className="btn btn--primary" onClick={handleSaveRisk} disabled={savingRisk}>
                    {savingRisk ? "Saving…" : "Save risk settings"}
                  </button>
                </div>
              </>
            ) : (
              <p className="empty">Loading risk settings…</p>
            )}
          </div>

          <div className="card">
            <h2>System events</h2>
            {events.length === 0 ? (
              <p className="empty">No system events.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Severity</th>
                      <th>Type</th>
                      <th>Message</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.slice(0, 10).map((e, i) => (
                      <tr key={`${e.occurred_at}-${i}`}>
                        <td>{fmtTime(e.occurred_at)}</td>
                        <td>
                          <span
                            className={`badge ${
                              e.severity === "ERROR" ? "badge--red" : e.severity === "WARNING" ? "badge--amber" : "badge--green"
                            }`}
                          >
                            {e.severity}
                          </span>
                        </td>
                        <td>{e.event_type}</td>
                        <td>{e.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>
      </div>
    </>
  );
}
