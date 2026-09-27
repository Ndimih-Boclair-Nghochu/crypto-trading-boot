"""Exit-strategy sweep backtest for the NBN trading system.

Phase 1 (slow): walk historical public candles through the REAL models + gate +
risk engine (exactly like _evaluate_symbol) to find every trade the system would
enter, caching each entry plus its forward price path.

Phase 2 (instant): replay those identical entries through several EXIT configs
(current peak-guard vs. letting winners run to 2R/3R) and report profit factor,
win rate and expectancy for each — so we can see which exit rule actually makes
the edge positive, measured on the same trades.

Runs on Binance public klines (no key, no testnet, no live gate).
Usage:  docker compose exec -T app python -u backtest.py [bars_per_symbol]
"""
from __future__ import annotations

import bisect
import sys
from decimal import Decimal

from binance.client import Client

from config import settings
from data.market_data import Candle, MarketMeta
from analysis.technical_analysis import TechnicalAnalysisEngine
from models.lstm_model import LSTMModelService
from models.rl_agent import RLAgentService
from models.confidence_gate import ConfidenceGate
from trading.strategy_engine import StrategyEngine
from trading.risk_manager import RiskManager, TradeCandidate

FWD = 96                       # bars of forward price to cache per entry (~4 days at 1h)
_TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

# Exit configs to compare. arm_r / giveback define the peak-guard; target_R sets
# the profit target as a multiple of risk (None = use the plan's own target).
# time_stop_bars only closes a trade that is still ~flat (mirrors live behaviour).
CONFIGS = [
    # be_arm: once peak_r reaches this, the stop jumps to be_floor (in R) so a
    # trade that proved itself can't fall back to a full -1R. This is the fix
    # aimed at the green-then-red losers that currently bleed a full R.
    ("CURRENT (arm0.5/gb5%, no BE)", dict(arm_r=0.5, giveback=0.05, target_R=None)),
    ("+BE floor .1@.4", dict(arm_r=0.5, giveback=0.05, target_R=None, be_arm=0.4, be_floor=0.1)),
    ("+BE floor .1@.4, run arm1.2/gb35% 2R", dict(arm_r=1.2, giveback=0.35, target_R=2.0, be_arm=0.4, be_floor=0.1)),
    ("+BE floor .0@.3, run arm1.0/gb30%", dict(arm_r=1.0, giveback=0.30, target_R=None, be_arm=0.3, be_floor=0.0)),
    ("+BE floor .15@.5, run arm1.5/gb40% 3R", dict(arm_r=1.5, giveback=0.40, target_R=3.0, be_arm=0.5, be_floor=0.15)),
    ("+BE floor .1@.4, target 2R only", dict(arm_r=999, giveback=0.99, target_R=2.0, be_arm=0.4, be_floor=0.1)),
]


def fetch(pub: Client, symbol: str, tf: str, want: int) -> list[Candle]:
    import time as _t
    out: list[list] = []
    end = None
    while len(out) < want:
        batch = None
        for attempt in range(4):  # survive transient network stalls
            try:
                batch = pub.get_klines(symbol=symbol, interval=tf,
                                       limit=min(1000, want - len(out)),
                                       **({"endTime": end} if end else {}))
                break
            except Exception as exc:
                if attempt == 3:
                    print(f"  fetch {symbol} {tf} failed after retries: {exc}", flush=True)
                    raise
                _t.sleep(2 * (attempt + 1))
        if not batch:
            break
        out = batch + out
        end = batch[0][0] - 1
        if len(batch) < 1000:
            break
    return [Candle(symbol=symbol, timeframe=tf, open_time=int(k[0]), open=Decimal(k[1]),
                   high=Decimal(k[2]), low=Decimal(k[3]), close=Decimal(k[4]),
                   volume=Decimal(k[5]), close_time=int(k[6])) for k in out]


async def find_entries(bars: int):
    pub = Client(requests_params={"timeout": 25})  # never hang forever on a bad connection
    symbols = list(settings.symbols)
    tfs = list({tf.lower() for tf in settings.timeframes} | {settings.primary_timeframe})
    primary = settings.primary_timeframe

    ta, lstm, rl = TechnicalAnalysisEngine(), LSTMModelService(), RLAgentService()
    gate, strat, risk = ConfidenceGate(), StrategyEngine(), RiskManager()

    async def _no_news(*a, **k):
        return False
    gate._major_news_window = _no_news
    meta = MarketMeta(fear_greed=None)

    entries = []
    approved = blocked = 0

    for symbol in symbols:
        print(f"  walking {symbol} ...", flush=True)
        data = {tf: fetch(pub, symbol, tf, bars if tf == primary else max(600, bars))
                for tf in tfs}
        prim = data[primary]
        if len(prim) < 320:
            print(f"{symbol}: only {len(prim)} {primary} candles — skipping")
            continue
        # precompute open_time index arrays for fast alignment
        times = {tf: [c.open_time for c in data[tf]] for tf in tfs}

        i = 300
        while i < len(prim) - 2:
            t = prim[i].open_time
            cbytf = {}
            for tf in tfs:
                hi = bisect.bisect_right(times[tf], t)
                cbytf[tf] = data[tf][max(0, hi - 500):hi]
            analysis = ta.compute_all(cbytf, meta=meta)
            if not analysis.get("timeframes"):
                i += 1
                continue
            pframe = strat.primary_frame(analysis)
            lstm_sig = lstm.predict(symbol, pframe)
            rl_dec = rl.decide(strat.build_rl_features(analysis))
            analysis["regime"] = str(strat.classify_regime(analysis, meta.fear_greed).value)
            g = await gate.passes(lstm_sig, rl_dec, analysis, symbol, cooldown_reason=None)
            if not g.approved:
                blocked += 1
                i += 1
                continue
            approved += 1
            sig = strat.build_trade_signal(symbol, analysis, lstm_sig, rl_dec, g, meta.fear_greed)
            latest = sig.indicator_state.get("latest", {})
            cand = TradeCandidate(signal=sig,
                                  entry_price=Decimal(str(latest.get("close", "0") or "0")),
                                  atr=Decimal(str(latest.get("atr_14", "0") or "0")),
                                  account_balance=Decimal("10000"),
                                  available_balance=Decimal("10000"),
                                  conviction=g.conviction)
            plan = await risk.calculate(cand)
            if not plan.approved:
                i += 1
                continue
            risk_u = plan.initial_risk_per_unit or abs(plan.entry_price - plan.sl_price)
            if risk_u <= 0 or plan.entry_price <= 0:
                i += 1
                continue
            path = [(prim[j].high, prim[j].low, prim[j].close)
                    for j in range(i + 1, min(i + 1 + FWD, len(prim)))]
            entries.append({
                "symbol": symbol,
                "is_long": (plan.direction or "LONG") == "LONG",
                "regime": sig.regime_at_entry,
                "strategy": sig.strategy_used,
                "entry": plan.entry_price, "sl": plan.sl_price, "risk": risk_u,
                "plan_target": plan.final_target_price or plan.tp1_price, "path": path,
            })
            # occupancy: skip to the baseline exit so the next entry doesn't overlap
            i += _baseline_exit_bars(entries[-1]) + 1
    return entries, approved, blocked


def _rmult(px, entry, risk, is_long):
    return ((px - entry) if is_long else (entry - px)) / risk


def _baseline_exit_bars(e) -> int:
    """How many bars the trade occupies under the live (baseline) exit — used only
    to space out entries so they don't overlap."""
    r = simulate_exit(e, arm_r=Decimal("0.5"), giveback=Decimal("0.05"), target_R=None)
    return r["bars"]


def simulate_exit(e, arm_r, giveback, target_R, be_arm=None, be_floor=Decimal("0")):
    entry, sl, risk, is_long = e["entry"], e["sl"], e["risk"], e["is_long"]
    be_arm = None if be_arm is None else Decimal(str(be_arm))
    be_floor = Decimal(str(be_floor))
    target = e["plan_target"] if target_R is None else (
        entry + target_R * risk if is_long else entry - target_R * risk)
    time_stop_bars = int(float(settings.time_stop_hours) * 60 / _TF_MIN[settings.primary_timeframe])
    peak_r = Decimal("0")
    floor_r = Decimal("-1")  # effective stop in R; rises to be_floor once armed
    for j, (hi, lo, close) in enumerate(e["path"]):
        hi_r = _rmult(hi if is_long else lo, entry, risk, is_long)
        lo_r = _rmult(lo if is_long else hi, entry, risk, is_long)
        peak_r = max(peak_r, hi_r)
        # Breakeven-floor: once the trade has been be_arm in front, its stop can
        # never sit below be_floor again -- so a proven trade can't lose a full R.
        if be_arm is not None and peak_r >= be_arm and be_floor > floor_r:
            floor_r = be_floor
        # Effective stop (original SL at -1R, or the raised breakeven floor).
        if lo_r <= floor_r:
            reason = "BE_FLOOR" if floor_r > Decimal("-1") else "SL"
            return {"r": floor_r, "reason": reason, "bars": j + 1}
        if peak_r >= arm_r and lo_r < peak_r * (Decimal("1") - giveback):
            return {"r": peak_r * (Decimal("1") - giveback), "reason": "PEAK_GUARD", "bars": j + 1}
        if (is_long and hi >= target) or ((not is_long) and lo <= target):
            return {"r": _rmult(target, entry, risk, is_long), "reason": "TARGET", "bars": j + 1}
        cr = _rmult(close, entry, risk, is_long)
        if (j + 1) >= time_stop_bars and Decimal("-0.5") < cr < Decimal("0.5"):
            return {"r": cr, "reason": "TIME_STOP", "bars": j + 1}
    last = e["path"][-1][2] if e["path"] else entry
    return {"r": _rmult(last, entry, risk, is_long), "reason": "END", "bars": len(e["path"])}


def report(name, results):
    fee_r = Decimal("2") * Decimal(str(settings.taker_fee_rate))  # ~ round-trip fee as fraction; small
    rs = [x["r"] for x in results]
    n = len(rs)
    if n == 0:
        print(f"{name:32s}  no trades")
        return
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gw = sum(wins) or Decimal("0")
    gl = -sum(losses) or Decimal("0")
    pf = float(gw / gl) if gl > 0 else float("inf")
    exp = float(sum(rs) / n)
    wr = len(wins) / n * 100
    print(f"{name:32s}  n={n:3d}  win={wr:4.1f}%  avgR={exp:+.3f}  PF={pf:4.2f}  sumR={float(sum(rs)):+6.2f}")


async def main():
    bars = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    print(f"Finding entries over {bars} bars/symbol on {list(settings.symbols)} ...", flush=True)
    entries, approved, blocked = await find_entries(bars)
    print(f"\ngate approved={approved}  blocked={blocked}  cached entries={len(entries)}\n")
    print("================ EXIT-STRATEGY SWEEP (same entries) ================")
    for name, cfg in CONFIGS:
        results = [simulate_exit(e,
                                 arm_r=Decimal(str(cfg["arm_r"])),
                                 giveback=Decimal(str(cfg["giveback"])),
                                 target_R=(None if cfg["target_R"] is None else Decimal(str(cfg["target_R"]))),
                                 be_arm=cfg.get("be_arm"),
                                 be_floor=cfg.get("be_floor", 0))
                   for e in entries]
        report(name, results)
    print("===================================================================")

    # Cross-regime / both-direction breakdown under the WINNING exit config:
    # tight peak-guard arm 0.5 / giveback 5%, NO profit-floor (the floor was
    # proven to destroy the edge). Answers "does the edge hold across regimes,
    # and do longs AND shorts each pay their way?"
    def _live(e):
        return simulate_exit(e, arm_r=Decimal("0.5"), giveback=Decimal("0.05"),
                             target_R=None, be_arm=None, be_floor=0)

    def group(entries, key_fn, label):
        buckets = {}
        for e in entries:
            buckets.setdefault(key_fn(e), []).append(_live(e))
        print(f"---- by {label} ----")
        for k in sorted(buckets, key=str):
            report(str(k), buckets[k])

    print("\n============ CROSS-REGIME BREAKDOWN (live exit config) ============")
    report("ALL", [_live(e) for e in entries])
    group(entries, lambda e: "LONG " if e["is_long"] else "SHORT", "direction")
    group(entries, lambda e: e.get("strategy", "?"), "strategy")
    group(entries, lambda e: e.get("regime", "?"), "regime")
    print("===================================================================")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
