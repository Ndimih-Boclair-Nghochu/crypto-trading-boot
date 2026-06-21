# Deploying to Fly.io

This is an alternative to Render for the backend (bot + API), useful if
Render's instance memory is too small for PyTorch model training. Fly.io
deploys straight from the existing `Dockerfile` -- no manual server setup,
no firewall configuration, HTTPS is automatic.

The frontend (Vercel) is unaffected. The database can stay on Render
Postgres (Fly can reach it over the public internet using Render's
*external* connection string) or move to Fly's own Postgres -- both are
covered below.

## 0. Important: trial accounts only run machines for 5 minutes

Without a payment method on file, Fly runs your app in **trial mode**:
machines are force-stopped after 5 minutes (`Trial machine stopping. To
run for longer than 5m0s, add a credit card...` in the logs), then
restarted, repeatedly. This looks like a crash but isn't one -- it's a
deliberate trial limitation, separate from any application bug.

For a continuously-running trading bot, this needs to be resolved before
the app can stay up: add a payment method at https://fly.io/dashboard ->
your org -> Billing. Fly's free allowance still applies after adding a
card -- this isn't necessarily a charge, it just lifts the 5-minute
trial-mode cap.

## 1. Install the CLI and sign up

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

This opens a browser to sign up / log in (GitHub, Google, or email). No
separate signup page needed -- `fly auth login` handles account creation.

## 2. Launch the app

From the repo root (where `Dockerfile` and `fly.toml` already live):

```bash
fly launch --no-deploy
```

`--no-deploy` lets you review/adjust settings (region, app name) before the
first deploy. It will detect the existing `fly.toml` and `Dockerfile` and
ask to confirm or adjust the app name (must be globally unique on Fly) and
region. Accept or edit as prompted -- `fly.toml` in this repo already sets
sensible defaults (1GB RAM, persistent/non-scaling-to-zero, health check on
`/api/health`).

## 3. Set environment variables (secrets)

Fly env vars/secrets are set via the CLI, not committed to `fly.toml`. Set
everything from `.env.example`:

```bash
fly secrets set \
  DATABASE_URL="postgresql+asyncpg://..." \
  BINANCE_API_KEY="..." \
  BINANCE_SECRET="..." \
  USE_TESTNET=true \
  SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ADAUSDT"
  # ...and any others you want to override from their defaults in config.py
```

Setting secrets automatically triggers a redeploy with them applied.

### Using Render's existing Postgres

If you want to keep your current Render Postgres database (recommended --
no data migration needed), use its **external** connection string (not the
internal one, which only works from inside Render's network) as
`DATABASE_URL` above. Find it in the Render dashboard under your Postgres
instance -> Connections -> External Database URL.

### Or: create a new Postgres on Fly instead

```bash
fly postgres create --name crypto-trading-boot-db
fly postgres attach crypto-trading-boot-db --app crypto-trading-boot
```

`fly postgres attach` automatically sets `DATABASE_URL` as a secret on your
app -- skip setting it manually above if you do this.

## 4. Deploy

```bash
fly deploy
```

This builds the Dockerfile (remotely on Fly's builders by default, no
local Docker needed) and deploys it. Watch the first boot:

```bash
fly logs
```

The first boot runs the same training bootstrap as on Render -- this is
exactly the step that was getting OOM-killed there. Watch for the
`[memory]` log lines added for diagnosing this; with 1GB (vs. whatever
smaller amount Render's instance had) it may already be enough. If it
still OOMs, bump memory:

```bash
fly scale memory 2048   # 2GB
```

and redeploy. Fly's paid tiers are pay-as-you-go by the second, so this is
cheap to test at a few different sizes if needed.

## 5. Point the frontend at the new backend

Your app's URL is `https://<app-name>.fly.dev` (HTTPS automatic, no setup
needed). In Vercel, update the `NEXT_PUBLIC_API_URL` environment variable
to this URL and redeploy.

## Updating the deployed code later

```bash
git pull   # if needed
fly deploy
```

Unlike Render, this isn't automatic on every git push by default -- you
(or a GitHub Actions workflow you set up separately, which Fly's docs also
support) run `fly deploy` when you want to ship a new version.

## Useful commands

```bash
fly status        # is it running, how much memory/CPU is it using
fly logs           # tail live logs
fly secrets list   # see which secrets are set (not their values)
fly scale memory 2048   # change the VM's memory allocation
fly ssh console     # shell into the running machine, if needed
```
