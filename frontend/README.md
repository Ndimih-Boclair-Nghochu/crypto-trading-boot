# Crypto Trading Desk — Frontend

A Next.js dashboard for the trading bot's backend API (`/api` in the repo root,
deployed on Render). Shows live status, open positions, equity, trade
history, strategy performance, risk settings, and system events, and lets you
toggle trading on/off.

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

The FastAPI backend (`api/server.py`) reads `FRONTEND_ORIGIN` to set its
allowed CORS origins. On Render, set:

- `FRONTEND_ORIGIN` = `https://<your-vercel-app>.vercel.app`

(comma-separate multiple origins if you also test from `http://localhost:3000`).
