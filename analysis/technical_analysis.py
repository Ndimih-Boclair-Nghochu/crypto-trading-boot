from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from analysis.pattern_recognition import analyze_patterns, pattern_bias
from data.market_data import Candle, MarketMeta


class TechnicalAnalysisEngine:
    def compute_all(self, candles_by_timeframe: dict[str, list[Candle]], meta: MarketMeta | None = None) -> dict[str, Any]:
        frames: dict[str, dict[str, Any]] = {}
        for timeframe, candles in candles_by_timeframe.items():
            if len(candles) >= 30:
                frames[timeframe.lower()] = self.compute_timeframe(candles, fear_greed=meta.fear_greed if meta else None)
        confluence = self.multi_timeframe_confluence(frames)
        return {"timeframes": frames, "confluence": confluence}

    def compute_timeframe(self, candles: list[Candle], fear_greed: float | None = None) -> dict[str, Any]:
        df = candles_to_frame(candles)
        df = enrich_indicators(df, fear_greed=fear_greed)
        patterns = analyze_patterns(df)
        latest = df.iloc[-1].replace({np.nan: None}).to_dict()
        latest["patterns"] = {
            "candlestick": patterns.candlestick,
            "chart": patterns.chart,
            "support_levels": patterns.support_levels,
            "resistance_levels": patterns.resistance_levels,
            "bias": pattern_bias(patterns),
        }
        latest["bias"] = directional_bias(latest)
        return {
            "latest": latest,
            "series_tail": df.tail(120).replace({np.nan: None}).to_dict(orient="records"),
            "timeframe": candles[-1].timeframe,
        }

    def multi_timeframe_confluence(self, frames: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if not frames:
            return {"score": 0.0, "direction": "NEUTRAL", "timeframes": {}}
        biases = {tf: payload["latest"].get("bias", "NEUTRAL") for tf, payload in frames.items()}
        directional = [bias for bias in biases.values() if bias in {"LONG", "SHORT"}]
        if not directional:
            return {"score": 0.0, "direction": "NEUTRAL", "timeframes": biases}
        counts = Counter(directional)
        direction, count = counts.most_common(1)[0]
        strength_values = []
        for payload in frames.values():
            latest = payload["latest"]
            if latest.get("bias") == direction:
                strength_values.append(_timeframe_strength(latest, direction))
        base = 65.0 * min(count / 3, 1.0)
        agreement = 20.0 * (count / max(len(frames), 1))
        strength = 15.0 * (float(np.mean(strength_values)) if strength_values else 0.0)
        score = min(100.0, base + agreement + strength)
        if count < 3:
            score = min(score, 64.0)
        return {"score": round(score, 2), "direction": direction, "timeframes": biases}


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    rows = [c.to_dict() for c in candles if c.is_valid()]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid candles supplied")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def enrich_indicators(df: pd.DataFrame, fear_greed: float | None = None) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    for period in (9, 21, 50, 100, 200):
        out[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd_line"] = ema12 - ema26
    out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]

    out["atr_14"] = atr(out, 14)
    out["adx"], out["di_plus"], out["di_minus"] = adx(out, 14)

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    out["ichimoku_tenkan"] = tenkan
    out["ichimoku_kijun"] = kijun
    out["ichimoku_senkou_a"] = ((tenkan + kijun) / 2).shift(26)
    out["ichimoku_senkou_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    out["ichimoku_chikou"] = close.shift(-26)

    out["rsi_14"] = rsi(close, 14)
    out["rsi_bullish_divergence"], out["rsi_bearish_divergence"] = rsi_divergence(out)
    stoch = (out["rsi_14"] - out["rsi_14"].rolling(14).min()) / (
        out["rsi_14"].rolling(14).max() - out["rsi_14"].rolling(14).min()
    )
    out["stoch_rsi_k"] = stoch.rolling(3).mean() * 100
    out["stoch_rsi_d"] = out["stoch_rsi_k"].rolling(3).mean()
    out["williams_r"] = -100 * ((high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min()))
    typical = (high + low + close) / 3
    out["cci_20"] = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).apply(_mean_abs_dev, raw=True))

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_mid
    out["bb_percent_b"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["bb_width_percentile"] = out["bb_width"].rolling(100, min_periods=20).rank(pct=True) * 100
    out["bb_squeeze"] = out["bb_width_percentile"] < 30

    out["keltner_mid"] = close.ewm(span=20, adjust=False).mean()
    out["keltner_upper"] = out["keltner_mid"] + 2 * out["atr_14"]
    out["keltner_lower"] = out["keltner_mid"] - 2 * out["atr_14"]
    out["historical_volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(365)
    out["volatility_percentile"] = out["historical_volatility_20"].rolling(100, min_periods=20).rank(pct=True) * 100

    out["obv"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    out["vwap"] = daily_vwap(out)
    out["cmf_20"] = chaikin_money_flow(out, 20)
    out["volume_profile_nodes"] = [volume_profile_nodes(out)] * len(out)
    # Neutral midpoint (0-100 scale) when no live Fear & Greed Index reading
    # is available, e.g. during historical-data training. Leaving this as
    # None/NaN poisons every row for any downstream consumer (like the LSTM
    # model) that requires a complete feature set -- dropna() would then
    # drop the entire dataframe rather than just the rows that genuinely
    # lack data.
    out["fear_greed"] = fear_greed if fear_greed is not None else 50.0
    out["atr_spike"] = out["atr_14"] > 3 * out["atr_14"].rolling(20).mean()
    return out


def _mean_abs_dev(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values - np.mean(values))))


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_values = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_values
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_values
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean(), plus_di, minus_di


def rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> tuple[pd.Series, pd.Series]:
    bullish = pd.Series(False, index=df.index)
    bearish = pd.Series(False, index=df.index)
    if "rsi_14" not in df or len(df) < lookback:
        return bullish, bearish
    for idx in range(lookback, len(df)):
        window = df.iloc[idx - lookback : idx + 1]
        price_low_now = window["low"].iloc[-1] <= window["low"].quantile(0.2)
        rsi_low_now = window["rsi_14"].iloc[-1] > window["rsi_14"].quantile(0.2)
        price_high_now = window["high"].iloc[-1] >= window["high"].quantile(0.8)
        rsi_high_now = window["rsi_14"].iloc[-1] < window["rsi_14"].quantile(0.8)
        bullish.iloc[idx] = bool(price_low_now and rsi_low_now)
        bearish.iloc[idx] = bool(price_high_now and rsi_high_now)
    return bullish, bearish


def daily_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    grouped = df.index.date
    return pv.groupby(grouped).cumsum() / df["volume"].groupby(grouped).cumsum().replace(0, np.nan)


def volume_profile_nodes(df: pd.DataFrame, bins: int = 24) -> list[float]:
    recent = df.tail(120)
    if recent.empty:
        return []
    prices = ((recent["high"] + recent["low"] + recent["close"]) / 3).to_numpy()
    volumes = recent["volume"].to_numpy()
    hist, edges = np.histogram(prices, bins=bins, weights=volumes)
    top_idx = np.argsort(hist)[-3:]
    nodes = [(edges[i] + edges[i + 1]) / 2 for i in top_idx]
    return sorted(float(node) for node in nodes)


def chaikin_money_flow(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low
    mfv = mfm.fillna(0) * df["volume"]
    return mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def directional_bias(latest: dict[str, Any]) -> str:
    bullish = 0
    bearish = 0
    close = _num(latest.get("close"))
    ema21 = _num(latest.get("ema_21"))
    ema50 = _num(latest.get("ema_50"))
    ema200 = _num(latest.get("ema_200"))
    macd_hist = _num(latest.get("macd_hist"))
    rsi_value = _num(latest.get("rsi_14"))
    adx_value = _num(latest.get("adx"))
    di_plus = _num(latest.get("di_plus"))
    di_minus = _num(latest.get("di_minus"))
    bb_percent_b = _num(latest.get("bb_percent_b"))

    bullish += int(close > ema21 > ema50)
    bearish += int(close < ema21 < ema50)
    bullish += int(ema50 > ema200)
    bearish += int(ema50 < ema200)
    bullish += int(macd_hist > 0)
    bearish += int(macd_hist < 0)
    bullish += int(rsi_value > 52)
    bearish += int(rsi_value < 48)
    bullish += int(adx_value > 20 and di_plus > di_minus)
    bearish += int(adx_value > 20 and di_minus > di_plus)
    bullish += int(bb_percent_b > 0.55)
    bearish += int(bb_percent_b < 0.45)

    pattern_direction = latest.get("patterns", {}).get("bias", {}).get("direction")
    bullish += int(pattern_direction == "LONG")
    bearish += int(pattern_direction == "SHORT")

    if bullish >= bearish + 2:
        return "LONG"
    if bearish >= bullish + 2:
        return "SHORT"
    return "NEUTRAL"


def _timeframe_strength(latest: dict[str, Any], direction: str) -> float:
    adx_value = min(_num(latest.get("adx")) / 50, 1.0)
    macd = abs(_num(latest.get("macd_hist"))) / max(abs(_num(latest.get("close"))) * 0.001, 1e-12)
    rsi_value = _num(latest.get("rsi_14"))
    rsi_strength = abs(rsi_value - 50) / 50
    return float(np.clip(np.mean([adx_value, min(macd, 1.0), rsi_strength]), 0, 1))


def _num(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0
