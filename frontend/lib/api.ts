export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type TradingState = {
  status: string;
  trading_enabled: boolean;
  testnet: boolean;
  updated_at?: string | null;
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
  overview: () => request<Overview>("/api/overview"),
  riskSettings: () => request<RiskSettings>("/api/risk-settings"),
  toggleTrading: (enabled: boolean) =>
    request<TradingState>("/api/state/toggle", {
      method: "POST",
      body: JSON.stringify({ enabled }),
    }),
  closePosition: (symbol: string) =>
    request<{ close_requests: string[] }>("/api/state/close-position", {
      method: "POST",
      body: JSON.stringify({ symbol }),
    }),
  saveRiskSettings: (settings: RiskSettings) =>
    request<RiskSettings>("/api/risk-settings", {
      method: "POST",
      body: JSON.stringify(settings),
    }),
};
