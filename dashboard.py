from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import nest_asyncio
import pandas as pd
import streamlit as st

from config import settings
from db.connection import Database


STATE_PATH = settings.trading_state_path
RISK_OVERRIDE_PATH = settings.runtime_dir / "risk_overrides.json"


def read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"trading_enabled": False, "status": "PAUSED"}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"trading_enabled": False, "status": "ERROR"}


def write_state(enabled: bool) -> None:
    state = read_state()
    state.update(
        {
            "trading_enabled": enabled,
            "status": "ANALYZING" if enabled else "PAUSED",
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    write_state_dict(state)


def write_state_dict(state_dict: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    state_dict["updated_at"] = datetime.now(UTC).isoformat()
    STATE_PATH.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")


def read_risk_overrides() -> dict[str, Any]:
    if not RISK_OVERRIDE_PATH.exists():
        return {}
    try:
        return json.loads(RISK_OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


async def fetch_dashboard_data() -> dict[str, Any]:
    db = Database()
    try:
        await db.initialize()
        return {
            "trades": await db.fetch_all("SELECT * FROM trades ORDER BY entry_time DESC LIMIT 50"),
            "open_positions": await db.fetch_all("SELECT * FROM trades WHERE outcome = 'OPEN' ORDER BY entry_time DESC"),
            "equity": await db.fetch_all("SELECT * FROM equity_snapshots ORDER BY captured_at DESC LIMIT 300"),
            "events": await db.fetch_all("SELECT * FROM system_events ORDER BY occurred_at DESC LIMIT 100"),
            "performance": await db.fetch_all(
                """
                SELECT strategy_name, regime, win_rate, total_trades, profit_factor, avg_r_multiple
                FROM strategy_performance
                ORDER BY updated_at DESC
                LIMIT 50
                """
            ),
            "no_trade": await db.fetch_all("SELECT * FROM no_trade_log ORDER BY logged_at DESC LIMIT 50"),
        }
    except Exception:
        return {"trades": [], "open_positions": [], "equity": [], "events": [], "performance": [], "no_trade": []}
    finally:
        await db.close()


def main() -> None:
    st.set_page_config(page_title="Crypto AI Trading Desk", layout="wide")
    state = read_state()
    enabled = bool(state.get("trading_enabled", False))

    left, mid, right = st.columns([1.2, 1.6, 1.2])
    with left:
        new_enabled = st.toggle("TRADING ON", value=enabled)
        if new_enabled != enabled:
            write_state(new_enabled)
            st.rerun()
    with mid:
        status = "TRADING PAUSED" if not enabled else state.get("status", "ANALYZING")
        st.markdown(
            f"<h1 style='margin:0;color:{'#c1121f' if not enabled else '#1b7f3a'}'>{status}</h1>",
            unsafe_allow_html=True,
        )
    with right:
        st.metric("UTC Time", datetime.now(UTC).strftime("%H:%M:%S"))
        st.metric("Binance", "TESTNET" if settings.use_testnet else "LIVE")

    nest_asyncio.apply()
    data = asyncio.get_event_loop().run_until_complete(fetch_dashboard_data())
    trades = pd.DataFrame(data["trades"])
    open_positions = pd.DataFrame(data["open_positions"])
    equity = pd.DataFrame(data["equity"])
    performance = pd.DataFrame(data["performance"])
    events = pd.DataFrame(data["events"])
    no_trade = pd.DataFrame(data["no_trade"])

    balance = float(equity["total_equity"].iloc[0]) if not equity.empty else 0.0
    peak = float(equity["peak_equity"].iloc[0]) if not equity.empty else 0.0
    drawdown = float(equity["drawdown_pct"].iloc[0]) if not equity.empty else 0.0
    today_pnl = _period_pnl(trades, "day")
    week_pnl = _period_pnl(trades, "week")
    all_pnl = float(pd.to_numeric(trades.get("pnl_usd", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not trades.empty else 0.0

    st.subheader("Account Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Equity USDT", f"{balance:,.2f}")
    c2.metric("Today's P&L", f"{today_pnl:,.2f}")
    c3.metric("Week's P&L", f"{week_pnl:,.2f}")
    c4.metric("All-time P&L", f"{all_pnl:,.2f}")
    c5.metric("Drawdown", f"{drawdown:.2f}%", delta=f"Peak {peak:,.2f}")

    st.subheader("Open Positions")
    if open_positions.empty:
        st.info("No open positions.")
    else:
        cols_to_show = ["symbol", "direction", "entry_price", "sl_price", "tp1_price", "quantity", "entry_time"]
        for _, row in open_positions.iterrows():
            col_data, col_btn = st.columns([5, 1])
            with col_data:
                st.write({c: row.get(c, "") for c in cols_to_show if c in row})
            with col_btn:
                symbol = str(row.get("symbol", ""))
                if st.button(f"Close {symbol}", key=f"close_{row.get('trade_id', '')}"):
                    state = read_state()
                    closes = state.get("close_requests", [])
                    closes.append(symbol)
                    state["close_requests"] = closes
                    write_state_dict(state)
                    st.warning(f"Close request sent for {symbol}")
                    st.rerun()

    st.subheader("Market Analysis")
    if no_trade.empty:
        watched = pd.DataFrame({"symbol": settings.symbols, "status": ["WAITING"] * len(settings.symbols)})
    else:
        watched = no_trade.groupby("symbol").head(1)[
            ["symbol", "regime", "lstm_confidence", "confluence_score", "gate_failed", "logged_at"]
        ]
    st.dataframe(watched.head(5), use_container_width=True, hide_index=True)

    st.subheader("Trade History")
    if trades.empty:
        st.info("No trades recorded yet.")
    else:
        cols = ["entry_time", "symbol", "direction", "entry_price", "exit_price", "pnl_usd", "r_multiple", "outcome"]
        st.dataframe(trades[[c for c in cols if c in trades]], use_container_width=True, hide_index=True)

    st.subheader("Performance Stats")
    p1, p2 = st.columns(2)
    with p1:
        if not performance.empty:
            chart = performance.copy()
            chart["win_rate"] = pd.to_numeric(chart["win_rate"], errors="coerce").fillna(0)
            st.bar_chart(chart, x="strategy_name", y="win_rate")
        else:
            st.info("No strategy performance yet.")
    with p2:
        if not equity.empty:
            curve = equity.sort_values("captured_at")
            curve["total_equity"] = pd.to_numeric(curve["total_equity"], errors="coerce")
            st.line_chart(curve, x="captured_at", y="total_equity")
        else:
            st.info("No equity curve yet.")

    st.subheader("Risk Settings (editable - takes effect on next trade cycle)")
    overrides = read_risk_overrides()
    r1, r2, r3, r4, r5 = st.columns(5)
    risk_pct = r1.number_input(
        "Risk %",
        value=float(overrides.get("max_risk_per_trade_pct", settings.max_risk_per_trade_pct)),
        min_value=0.1,
        max_value=5.0,
        step=0.1,
    )
    daily_pct = r2.number_input(
        "Daily Loss %",
        value=float(overrides.get("max_daily_loss_pct", settings.max_daily_loss_pct)),
        min_value=1.0,
        max_value=20.0,
    )
    weekly_pct = r3.number_input(
        "Weekly Loss %",
        value=float(overrides.get("max_weekly_loss_pct", settings.max_weekly_loss_pct)),
        min_value=1.0,
        max_value=30.0,
    )
    max_trades = r4.number_input(
        "Max Trades",
        value=int(overrides.get("max_concurrent_trades", settings.max_concurrent_trades)),
        min_value=1,
        max_value=10,
        step=1,
    )
    confidence = r5.number_input(
        "Confidence Threshold",
        value=float(overrides.get("confidence_threshold", settings.confidence_threshold)),
        min_value=0.5,
        max_value=0.99,
        step=0.01,
    )
    if st.button("Save Risk Settings"):
        risk_overrides = {
            "max_risk_per_trade_pct": risk_pct,
            "max_daily_loss_pct": daily_pct,
            "max_weekly_loss_pct": weekly_pct,
            "max_concurrent_trades": int(max_trades),
            "confidence_threshold": confidence,
        }
        RISK_OVERRIDE_PATH.parent.mkdir(exist_ok=True)
        RISK_OVERRIDE_PATH.write_text(json.dumps(risk_overrides, indent=2), encoding="utf-8")
        st.success("Risk settings saved. Active on next trade cycle.")

    st.subheader("System Events")
    if events.empty:
        st.info("No system events.")
    else:
        cols = ["occurred_at", "severity", "event_type", "message"]
        st.dataframe(events[[c for c in cols if c in events]], use_container_width=True, hide_index=True)

    time.sleep(5)
    st.rerun()


def _period_pnl(trades: pd.DataFrame, period: str) -> float:
    if trades.empty or "exit_time" not in trades or "pnl_usd" not in trades:
        return 0.0
    df = trades.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0)
    now = datetime.now(UTC)
    if period == "day":
        df = df[df["exit_time"].dt.date == now.date()]
    elif period == "week":
        iso = now.isocalendar()
        df = df[df["exit_time"].apply(lambda ts: False if pd.isna(ts) else ts.isocalendar()[:2] == iso[:2])]
    return float(df["pnl_usd"].sum())


if __name__ == "__main__":
    main()
