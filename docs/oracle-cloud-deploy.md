# Self-hosting on Oracle Cloud (Always Free ARM)

This is an alternative to Render for the backend (bot + API), useful if
Render's instance memory is too small for PyTorch model training. Oracle's
Always Free Ampere A1 (ARM/aarch64) shape gives up to 4 OCPUs / 24GB RAM at
no cost (subject to Oracle's current Always Free limits, which have been
reduced over time -- check the console at signup).

The frontend (Vercel) and database (can stay on Render Postgres, or move to
this same server via the `postgres` service in `docker-compose.yml`) are
unaffected by this guide.

## 1. Create the account and instance

1. Sign up at https://signup.oraclecloud.com/ (requires a real credit/debit
   card for identity verification; Always Free resources are not charged).
2. In the OCI Console, create a **Compute Instance**:
   - Shape: **VM.Standard.A1.Flex** (the Ampere ARM Always Free shape)
   - OCPUs / Memory: e.g. 2 OCPU / 12GB (well within the free allowance)
   - Image: **Ubuntu 22.04** (aarch64)
   - Generate an SSH key pair when prompted, or upload your own public key
     -- you'll need the private key to log in.
3. Note the instance's **public IP address** once it's running.

## 2. Open the firewall

Oracle blocks all inbound traffic except SSH by default, at two layers:

- **Security List** (OCI Console -> your VCN -> Security Lists -> Default
  Security List): add an Ingress Rule for TCP port **10000** (or whatever
  `PORT` you choose), source `0.0.0.0/0`.
- **The instance's own OS firewall** (Ubuntu uses iptables/netfilter via
  Oracle's cloud-init, separate from the Security List above):
  ```bash
  sudo iptables -I INPUT -p tcp --dport 10000 -j ACCEPT
  sudo netfilter-persistent save   # if installed; otherwise add to /etc/iptables/rules.v4
  ```

## 3. Install Docker

```bash
ssh -i /path/to/your-key.pem ubuntu@<instance-public-ip>

sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
# log out and back in for the group change to take effect
```

## 4. Clone the repo and configure environment

```bash
git clone https://github.com/Ndimih-Boclair-Nghochu/crypto-trading-boot.git
cd crypto-trading-boot
cp .env.example .env
nano .env   # fill in BINANCE_API_KEY, BINANCE_SECRET, etc. -- see .env.example
            # for the full list and README for which are required vs optional
```

If you want Postgres to also run on this server (instead of keeping Render's
managed Postgres), `docker-compose.yml` already includes a `postgres`
service and points `DATABASE_URL` at it automatically. If you'd rather keep
using Render's Postgres, remove the `DATABASE_URL` override line under the
`app` service in `docker-compose.yml` and put your existing Render Postgres
connection string in `.env` instead -- Render Postgres instances are
reachable from outside Render as long as you use their *external* (not
internal) connection string.

## 5. Build and run

```bash
docker compose up -d --build
docker compose logs -f app   # watch the bot/API start up and (on first run) train
```

The first boot will run the same training bootstrap as on Render -- this is
exactly the step that was getting OOM-killed there; with 12-24GB available
here it should complete without issue. Watch for
`MODEL_TRAINING: Initial model training completed` and the `[memory]` log
lines added for diagnosing this -- they'll now show genuinely comfortable
headroom instead of climbing toward a limit.

## 6. Point the frontend at the new backend

In Vercel, update the `NEXT_PUBLIC_API_URL` environment variable to:

```
http://<instance-public-ip>:10000
```

and redeploy. (For a production setup later, consider adding a domain name
+ HTTPS via Caddy or nginx + Let's Encrypt in front of this, and switching
to `https://...` -- plain HTTP is fine to get things running and verify the
memory fix first.)

## Updating the deployed code later

```bash
ssh -i /path/to/your-key.pem ubuntu@<instance-public-ip>
cd crypto-trading-boot
git pull
docker compose up -d --build
```

There's no automatic deploy-on-push here (unlike Render) -- you (or a CI
step you set up separately) need to SSH in and re-run the above after
pushing changes to GitHub.
