# Crypto Trading Desk — Frontend

A Next.js dashboard for the trading bot's backend API (`/api` in the repo root,
deployed on Render). Shows live status (including Binance and database
connectivity), open positions, equity, trade history, strategy performance,
risk settings, and system events. The bot trades continuously and has no
manual on/off switch in this dashboard.

## Local development

```bash
cd frontend
npm install
cp .env.example .env.local   # point NEXT_PUBLIC_API_URL at your backend
npm run dev
```

## Deploying to Vercel

This frontend lives in a subdirectory of the `crypto-trading-boot` repo
alongside the Python backend. **Vercel must be told to build only this
folder**, otherwise it will try to build the whole repo as a Python project
(and fail — the backend's dependencies like `torch` are far too large for a
serverless function).

1. In the Vercel dashboard, create a new project from the
   `crypto-trading-boot` GitHub repo.
2. In **Project Settings → General → Root Directory**, set it to `frontend`
   and save. (If you already created the project, do this before the next
   deploy — Vercel will then auto-detect Next.js correctly.)
3. In **Project Settings → Environment Variables**, add:
   - `NEXT_PUBLIC_API_URL` = the URL of your Render backend, e.g.
     `https://crypto-trading-boot.onrender.com`
4. Redeploy.

## Backend CORS

The FastAPI backend (`api/server.py`) allows all origins (`*`) unconditionally.
This API is read-mostly (status, trades, equity, risk settings) with no
authentication or destructive actions, so there's no security reason to
restrict the origin — and making it depend on an env var matching the exact
Vercel URL was a real failure mode in practice (a missing or mismatched
`FRONTEND_ORIGIN` silently blocked every request with no clear error). No
CORS-related environment variable needs to be set on Render.
