export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type TradingState = {
  status: string;
  reason?: string | null;
  trading_enabled: boolean;
  paused?: boolean;
  testnet: boolean;
  binance_connected?: boolean;
  updated_at?: string | null;
  db_connected?: boolean;
};

export type Trade = {
  trade_id?: string;
  symbol: string;
  direction: string;
  entry_price: number | string;
  exit_price?: number | string | null;
  sl_price?: number | string;
  tp1_price?: number | string;
  quantity?: number | string;
  pnl_usd?: number | string | null;
  r_multiple?: number | string | null;
  outcome: string;
  entry_time: string;
  exit_time?: string | null;
  current_price?: number | string | null;
  market_value?: number | string | null;
  unrealized_pnl?: number | string | null;
  expected_tp_usd?: number | string | null;
  expected_sl_usd?: number | string | null;
  unrealized_pct?: number | string | null;
};

export type EquityPoint = {
  total_equity: number | string;
  peak_equity: number | string;
  drawdown_pct: number | string;
  captured_at: string;
};

export type PerformanceRow = {
  strategy_name: string;
  regime?: string;
  win_rate: number | string;
  total_trades: number | string;
  profit_factor?: number | string;
  avg_r_multiple?: number | string;
};

export type SystemEvent = {
  occurred_at: string;
  severity: string;
  event_type: string;
  message: string;
};

export type NoTradeRow = {
  symbol: string;
  regime?: string;
  lstm_confidence?: number | string;
  confluence_score?: number | string;
  gate_failed?: string;
  gate_reasons?: string[];
  analysis_notes?: string;
  logged_at?: string;
};

export type Overview = {
  trades: Trade[];
  open_positions: Trade[];
  equity: EquityPoint[];
  events: SystemEvent[];
  performance: PerformanceRow[];
  no_trade: NoTradeRow[];
  symbols: string[];
  mode?: "demo" | "live" | string;
  db_connected?: boolean;
};

export type ModeInfo = {
  mode: "demo" | "live" | string;
  is_live: boolean;
  market_type: string;
  shorting_available: boolean;
  futures_keys_configured: boolean;
  live_trading_allowed: boolean;
  live_trading_reviewed: boolean;
  completed_trade_count: number;
  min_live_trades: number;
};

export type RiskSettings = {
  max_risk_per_trade_pct: number;
  max_daily_loss_pct: number;
  max_weekly_loss_pct: number;
  max_concurrent_trades: number;
  confidence_threshold: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<TradingState>("/api/health"),
  systemStop: () => request<{ ok: boolean; paused: boolean }>("/api/system/stop", { method: "POST" }),
  systemResume: () => request<{ ok: boolean; paused: boolean }>("/api/system/resume", { method: "POST" }),
  closePosition: (symbol: string) =>
    request<{ ok: boolean; queued: string }>(`/api/positions/${symbol}/close`, { method: "POST" }),
  overview: () => request<Overview>("/api/overview"),
  mode: () => request<ModeInfo>("/api/mode"),
  goLive: (confirm: string) =>
    request<{ ok: boolean; error?: string; note?: string }>("/api/system/go-live", {
      method: "POST",
      body: JSON.stringify({ confirm }),
    }),
  goDemo: () =>
    request<{ ok: boolean; note?: string }>("/api/system/go-demo", { method: "POST" }),
  riskSettings: () => request<RiskSettings>("/api/risk-settings"),
  saveRiskSettings: (settings: RiskSettings) =>
    request<RiskSettings>("/api/risk-settings", {
      method: "POST",
      body: JSON.stringify(settings),
    }),
};
