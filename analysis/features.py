from __future__ import annotations

"""Stationary model inputs, first-touch labels, and the single RL observation layout.

Everything the LSTM and the RL agent consume is defined here, once. Two reasons:

1. The previous feature set fed raw price levels (open/high/low/close/ema_21/
   ema_50/obv) into the network and z-scored them with one global mean/std
   computed over the whole history. On a trending asset that puts every recent
   bar permanently at the same end of the distribution, so the model cannot
   generalise from the past to the present. Every column below is a ratio, a
   return, or a bounded oscillator -- scale-free and comparable across symbols
   and across time.

2. The RL observation vector used to be built in two places with two different
   orderings, so three of sixteen features silently arrived in the wrong slots
   at inference time. There is now exactly one builder.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


# Order matters and is persisted alongside trained weights. Appending is safe;
# reordering or removing invalidates every saved model, which is why the name
# list is written into the model metadata and checked on load.
MODEL_FEATURES: tuple[str, ...] = (
    "ret_1",
    "ret_3",
    "ret_8",
    "ret_24",
    "close_vs_ema21",
    "ema21_vs_ema50",
    "ema50_vs_ema200",
    "close_vs_vwap",
    "rsi_centered",
    "stoch_rsi_centered",
    "macd_hist_atr",
    "adx_norm",
    "di_spread",
    "atr_pct",
    "bb_percent_centered",
    "bb_width_pct",
    "volume_ratio",
    "obv_slope",
    "cmf_20",
    "vol_percentile",
    "fear_greed_centered",
)

# Appended to MODEL_FEATURES to form the RL observation. These describe the
# agent's own position, which the price model knows nothing about.
RL_POSITION_FEATURES: tuple[str, ...] = (
    "position",
    "unrealized_r",
    "bars_held",
    "drawdown",
)

RL_OBSERVATION_SIZE = len(MODEL_FEATURES) + len(RL_POSITION_FEATURES)

INDEX_TO_SIGNAL = {0: "LONG", 1: "SHORT", 2: "NO_TRADE"}
SIGNAL_TO_INDEX = {value: key for key, value in INDEX_TO_SIGNAL.items()}
LABEL_INVALID = -1


def _safe_div(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    denom = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index)
    denom = denom.replace(0, np.nan)
    return numerator / denom


def build_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn an enrich_indicators() frame into the stationary feature matrix.

    Missing indicator columns degrade to neutral values rather than raising, so
    a short or partially-warmed frame still produces a usable row instead of
    taking the whole symbol out of service.
    """
    out = pd.DataFrame(index=frame.index)
    close = frame["close"].astype(float)

    log_close = np.log(close.replace(0, np.nan))
    out["ret_1"] = log_close.diff(1)
    out["ret_3"] = log_close.diff(3)
    out["ret_8"] = log_close.diff(8)
    out["ret_24"] = log_close.diff(24)

    ema21 = _column(frame, "ema_21", close)
    ema50 = _column(frame, "ema_50", close)
    ema200 = _column(frame, "ema_200", close)
    out["close_vs_ema21"] = _safe_div(close, ema21) - 1.0
    out["ema21_vs_ema50"] = _safe_div(ema21, ema50) - 1.0
    out["ema50_vs_ema200"] = _safe_div(ema50, ema200) - 1.0
    out["close_vs_vwap"] = _safe_div(close, _column(frame, "vwap", close)) - 1.0

    out["rsi_centered"] = (_column(frame, "rsi_14", 50.0) - 50.0) / 50.0
    out["stoch_rsi_centered"] = (_column(frame, "stoch_rsi_k", 50.0) - 50.0) / 50.0

    atr = _column(frame, "atr_14", np.nan).replace(0, np.nan)
    out["macd_hist_atr"] = (_column(frame, "macd_hist", 0.0) / atr).clip(-5, 5)
    out["adx_norm"] = (_column(frame, "adx", 0.0) / 50.0).clip(0, 2)
    out["di_spread"] = ((_column(frame, "di_plus", 0.0) - _column(frame, "di_minus", 0.0)) / 50.0).clip(-2, 2)
    out["atr_pct"] = _safe_div(atr, close).clip(0, 0.5)

    out["bb_percent_centered"] = (_column(frame, "bb_percent_b", 0.5) - 0.5).clip(-2, 2)
    out["bb_width_pct"] = (_column(frame, "bb_width_percentile", 50.0) / 100.0).clip(0, 1)
    out["vol_percentile"] = (_column(frame, "volatility_percentile", 50.0) / 100.0).clip(0, 1)

    volume = _column(frame, "volume", 0.0).astype(float)
    volume_mean = volume.rolling(20, min_periods=5).mean()
    out["volume_ratio"] = (_safe_div(volume, volume_mean) - 1.0).clip(-1, 5)

    # OBV is an unbounded cumulative sum, so its level is meaningless across
    # time. Its *rate of change*, scaled by its own recent volatility, is not.
    obv = _column(frame, "obv", 0.0).astype(float)
    obv_change = obv.diff()
    obv_scale = obv_change.rolling(50, min_periods=10).std().replace(0, np.nan)
    out["obv_slope"] = (obv_change / obv_scale).clip(-5, 5)

    out["cmf_20"] = _column(frame, "cmf_20", 0.0).clip(-1, 1)
    out["fear_greed_centered"] = (_column(frame, "fear_greed", 50.0) - 50.0) / 50.0

    out = out[list(MODEL_FEATURES)]
    return out.replace([np.inf, -np.inf], np.nan)


def _column(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    if isinstance(default, pd.Series):
        return default
    return pd.Series(default, index=frame.index, dtype="float64")


def triple_barrier_labels(
    frame: pd.DataFrame,
    horizon: int = 12,
    atr_multiple: float = 1.2,
) -> pd.Series:
    """Label each bar by which barrier price touches *first*.

    The previous labeller asked "did price rise 1.5xATR at any point in the next
    8 bars?" and "did it fall 1.5xATR?" independently, marking NO_TRADE when
    both were true. That ignores ordering, so a bar whose price dropped to the
    stop and only later recovered was labelled the same as one that ran straight
    to target. Walking forward bar by bar and stopping at the first touch makes
    the label mean what the trade would actually have done.

    Bars whose forward window runs past the end of the data are marked
    LABEL_INVALID and must be dropped -- they are unlabelable, not NO_TRADE.
    """
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    atr = pd.to_numeric(frame.get("atr_14", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(
        dtype=np.float64
    )

    total = len(close)
    labels = np.full(total, SIGNAL_TO_INDEX["NO_TRADE"], dtype=np.int64)
    long_idx = SIGNAL_TO_INDEX["LONG"]
    short_idx = SIGNAL_TO_INDEX["SHORT"]
    no_trade_idx = SIGNAL_TO_INDEX["NO_TRADE"]

    for i in range(total):
        width = atr[i]
        if not np.isfinite(width) or width <= 0 or not np.isfinite(close[i]):
            labels[i] = LABEL_INVALID
            continue
        upper = close[i] + atr_multiple * width
        lower = close[i] - atr_multiple * width
        end = min(i + horizon, total - 1)
        outcome = no_trade_idx
        for j in range(i + 1, end + 1):
            hit_up = np.isfinite(high[j]) and high[j] >= upper
            hit_down = np.isfinite(low[j]) and low[j] <= lower
            if hit_up and hit_down:
                # Both barriers inside one bar: the order is unknowable from
                # OHLC alone, so this is not a trade we can claim to predict.
                outcome = no_trade_idx
                break
            if hit_up:
                outcome = long_idx
                break
            if hit_down:
                outcome = short_idx
                break
        labels[i] = outcome

    if horizon > 0:
        labels[max(0, total - horizon) :] = LABEL_INVALID
    return pd.Series(labels, index=frame.index, name="label")


def build_rl_observation(
    features: Mapping[str, Any] | np.ndarray,
    position: float = 0.0,
    unrealized_r: float = 0.0,
    bars_held: float = 0.0,
    drawdown: float = 0.0,
) -> np.ndarray:
    """The one and only RL observation layout, used by training and inference.

    `features` is either a mapping keyed by MODEL_FEATURES names or an already
    ordered array of len(MODEL_FEATURES).
    """
    if isinstance(features, np.ndarray):
        base = features.astype(np.float32, copy=False)
        if base.shape[-1] != len(MODEL_FEATURES):
            raise ValueError(f"expected {len(MODEL_FEATURES)} features, got {base.shape[-1]}")
        values = list(base)
    else:
        values = [_finite(features.get(name, 0.0)) for name in MODEL_FEATURES]

    values.extend(
        [
            _finite(position),
            _finite(unrealized_r),
            _finite(bars_held) / 50.0,
            _finite(drawdown),
        ]
    )
    return np.array(values, dtype=np.float32)


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return result


def label_distribution(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels[labels >= 0], minlength=3)
    return {INDEX_TO_SIGNAL[i]: int(counts[i]) for i in range(3)}
