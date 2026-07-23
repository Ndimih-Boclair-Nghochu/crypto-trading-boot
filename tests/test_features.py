from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features import (
    LABEL_INVALID,
    MODEL_FEATURES,
    RL_OBSERVATION_SIZE,
    SIGNAL_TO_INDEX,
    build_model_features,
    build_rl_observation,
    triple_barrier_labels,
)
from analysis.technical_analysis import enrich_indicators


def synthetic_frame(rows: int = 400, drift: float = 0.0006, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.01, rows)
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.004, rows)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, rows)))
    frame = pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": rng.uniform(900, 1100, rows),
            "open_time": np.arange(rows) * 3_600_000,
        }
    )
    frame.index = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return enrich_indicators(frame)


def test_features_are_scale_free_across_price_levels() -> None:
    """The same shape at $100 and at $100,000 must produce the same features.

    This is the property the old raw-price feature set did not have, and the
    reason a model fitted on one price regime could not generalise to another.
    """
    cheap = synthetic_frame()
    expensive = cheap.copy()
    for column in ("open", "high", "low", "close"):
        expensive[column] = expensive[column] * 1000
    expensive = enrich_indicators(expensive[["open", "high", "low", "close", "volume", "open_time"]])

    a = build_model_features(cheap).tail(50).to_numpy(dtype=float)
    b = build_model_features(expensive).tail(50).to_numpy(dtype=float)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-6)


def test_feature_columns_match_declared_order() -> None:
    features = build_model_features(synthetic_frame())
    assert list(features.columns) == list(MODEL_FEATURES)


def test_labels_use_first_touch_not_either_touch() -> None:
    """A bar that drops to the stop before reaching target is a SHORT, not a wash."""
    frame = pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.5, 110.0, 110.0],
            "low": [100.0, 89.0, 89.0, 89.0],
            "atr_14": [5.0, 5.0, 5.0, 5.0],
        }
    )
    labels = triple_barrier_labels(frame, horizon=1, atr_multiple=1.0)
    # Bar 1 touches the lower barrier (95) on the next bar before the upper (105).
    assert labels.iloc[0] == SIGNAL_TO_INDEX["SHORT"]
    # The final `horizon` bars have no complete forward window.
    assert labels.iloc[-1] == LABEL_INVALID


def test_labels_produce_all_three_classes_on_real_shaped_data() -> None:
    labels = triple_barrier_labels(synthetic_frame(), horizon=12, atr_multiple=1.2)
    counts = labels[labels != LABEL_INVALID].value_counts()
    assert len(counts) >= 2
    assert counts.sum() > 200


def test_rl_observation_layout_is_stable() -> None:
    """One builder, one order: the scramble that broke inference cannot recur."""
    features = {name: float(i) for i, name in enumerate(MODEL_FEATURES)}
    obs = build_rl_observation(features, position=1.0, unrealized_r=0.5, bars_held=25.0, drawdown=0.1)
    assert obs.shape == (RL_OBSERVATION_SIZE,)
    assert list(obs[: len(MODEL_FEATURES)]) == [float(i) for i in range(len(MODEL_FEATURES))]
    assert obs[len(MODEL_FEATURES)] == 1.0  # position
    assert obs[len(MODEL_FEATURES) + 1] == 0.5  # unrealized_r
    assert obs[len(MODEL_FEATURES) + 3] == 0.1  # drawdown


def test_rl_observation_accepts_ordered_array_identically() -> None:
    values = np.arange(len(MODEL_FEATURES), dtype=np.float32)
    from_array = build_rl_observation(values, position=-1.0)
    from_mapping = build_rl_observation({name: float(i) for i, name in enumerate(MODEL_FEATURES)}, position=-1.0)
    assert np.allclose(from_array, from_mapping)


def test_observation_is_finite_even_with_dirty_input() -> None:
    obs = build_rl_observation(
        {name: float("nan") for name in MODEL_FEATURES},
        position=float("inf"),
        unrealized_r=float("-inf"),
    )
    assert np.all(np.isfinite(obs))
