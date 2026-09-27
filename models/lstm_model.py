from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.features import (
    INDEX_TO_SIGNAL,
    LABEL_INVALID,
    MODEL_FEATURES,
    SIGNAL_TO_INDEX,
    build_model_features,
    label_distribution,
    triple_barrier_labels,
)
from config import BASE_DIR, settings
from utils.logger import logger

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Small shared CPU container: PyTorch otherwise spawns one intra-op thread
    # per visible core, each with its own allocation overhead, for no benefit on
    # a model this size.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
except Exception as _torch_import_error:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    logger.error(
        f"PyTorch failed to import; the LSTM model will be unable to train or predict: {_torch_import_error!r}"
    )


# Bumped whenever the feature set or label definition changes. Weights whose
# metadata carries a different version are refused rather than silently
# producing predictions from a mismatched feature vector.
MODEL_VERSION = 2

# Kept as module-level names because tests and older call sites import them.
FEATURES = list(MODEL_FEATURES)


def _log_memory_checkpoint(label: str) -> None:
    try:
        import resource

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info(f"[memory] {label}: {rss_mb:.0f} MB RSS (peak)")
    except Exception:
        pass


@dataclass(frozen=True)
class LSTMSignal:
    direction: str
    confidence: float
    probabilities: dict[str, float]
    reason: str | None = None
    # Signed separation between the two directional classes, in the direction's
    # favour. A model that is 40/38/22 across LONG/SHORT/NO_TRADE has a high
    # argmax probability but essentially no directional opinion; margin catches
    # that where a raw confidence check does not.
    margin: float = 0.0
    # Out-of-sample directional precision recorded when this symbol's weights
    # were trained. Used to refuse trading on models that never learned.
    model_quality: float = 0.0


if torch and nn:

    class LSTMPriceModel(nn.Module):
        def __init__(self, input_size: int = len(MODEL_FEATURES), hidden_size: int = 96, num_layers: int = 2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(hidden_size, 48)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(48, 3)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            output, _ = self.lstm(x)
            last = self.norm(output[:, -1, :])
            x = self.dropout(last)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

else:
    LSTMPriceModel = None  # type: ignore[assignment]


class LSTMModelService:
    def __init__(self, weights_dir: Path | None = None, sequence_length: int = 60) -> None:
        self.weights_dir = weights_dir or (BASE_DIR / "models" / "weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self._models: dict[str, Any] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ predict

    def predict(self, symbol: str, indicators: dict[str, Any]) -> LSTMSignal:
        if torch is None or LSTMPriceModel is None:
            return self._no_trade("PyTorch unavailable")
        try:
            model = self._load_model(symbol)
            if model is None:
                return self._no_trade("weights missing")
            rows = indicators.get("series_tail", [])
            frame = pd.DataFrame(rows)
            if len(frame) < self.sequence_length:
                return self._no_trade("insufficient sequence")

            features = self._prepare_features(symbol, frame)
            if features is None:
                return self._no_trade("feature/scaler mismatch")
            tensor = torch.tensor(features[None, -self.sequence_length :, :], dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

            probabilities = {INDEX_TO_SIGNAL[i]: float(probs[i]) for i in range(3)}
            long_p = probabilities["LONG"]
            short_p = probabilities["SHORT"]
            quality = float(self._meta.get(symbol, {}).get("directional_precision", 0.0) or 0.0)
            idx = int(np.argmax(probs))
            direction = INDEX_TO_SIGNAL[idx]

            if direction == "NO_TRADE":
                # Report the best *directional* probability so the dashboard
                # number stays comparable row to row, but keep the direction as
                # NO_TRADE so nothing downstream can act on it. The old
                # LAB_MODE path renormalised over LONG/SHORT here and returned a
                # tradeable signal, which turned a model saying "84% do not
                # trade" into a displayed "0.80 confidence" SHORT.
                return LSTMSignal(
                    "NO_TRADE",
                    max(long_p, short_p),
                    probabilities,
                    f"model favours no-trade (P={probabilities['NO_TRADE']:.2f})",
                    margin=abs(long_p - short_p),
                    model_quality=quality,
                )

            margin = (long_p - short_p) if direction == "LONG" else (short_p - long_p)
            return LSTMSignal(direction, float(probs[idx]), probabilities, margin=margin, model_quality=quality)
        except Exception as exc:
            logger.exception(f"LSTM prediction failed for {symbol}: {exc}")
            return self._no_trade(str(exc))

    def _no_trade(self, reason: str) -> LSTMSignal:
        return LSTMSignal("NO_TRADE", 0.0, {"LONG": 0.0, "SHORT": 0.0, "NO_TRADE": 1.0}, reason)

    # -------------------------------------------------------------------- train

    def train(
        self,
        symbol: str,
        frame: pd.DataFrame,
        epochs: int = 40,
        batch_size: int = 64,
        horizon: int | None = None,
        atr_multiple: float | None = None,
    ) -> dict[str, float]:
        # Barriers default to the live stop distance, so a training label means
        # precisely "would this trade have reached +1R before -1R". Labelling
        # against some unrelated threshold teaches the model to predict moves the
        # exit logic would never actually have captured.
        horizon = int(settings.label_horizon if horizon is None else horizon)
        atr_multiple = float(settings.stop_atr_multiple if atr_multiple is None else atr_multiple)
        if torch is None or LSTMPriceModel is None or DataLoader is None or TensorDataset is None:
            raise RuntimeError(
                "PyTorch is unavailable in this environment, so the LSTM model cannot be trained. "
                "Check that 'torch' installed successfully in the build and that the instance has "
                "enough memory to import it."
            )
        if len(frame) < self.sequence_length + horizon + 100:
            raise ValueError(
                f"Not enough rows to train LSTM for {symbol}: {len(frame)} "
                f"(need at least {self.sequence_length + horizon + 100})"
            )

        source = frame.copy().replace([np.inf, -np.inf], np.nan)
        features = build_model_features(source)
        labels = triple_barrier_labels(source, horizon=horizon, atr_multiple=atr_multiple)

        dataset = features.copy()
        dataset["label"] = labels
        # Forward-fill only: a backward fill would pull future values into past
        # rows, which is leakage in a time series.
        dataset[list(MODEL_FEATURES)] = dataset[list(MODEL_FEATURES)].ffill()
        dataset = dataset[dataset["label"] != LABEL_INVALID].dropna()
        if len(dataset) < self.sequence_length + 100:
            raise ValueError(f"Not enough labelled rows to train LSTM for {symbol}: {len(dataset)}")

        values = dataset[list(MODEL_FEATURES)].astype("float32")
        y = dataset["label"].astype(int).to_numpy(dtype=np.int64)

        # Chronological split, and the scaler is fitted on the training portion
        # only. Fitting it over the whole set (as before) leaks validation-period
        # statistics into training and flatters the reported accuracy.
        row_split = max(int(len(values) * 0.8), self.sequence_length + 1)
        row_split = min(row_split, len(values) - 1)
        train_values = values.iloc[:row_split]
        means = train_values.mean()
        stds = train_values.std().replace(0, 1.0).fillna(1.0)
        scaled = ((values - means) / stds).to_numpy(dtype=np.float32)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        _log_memory_checkpoint(f"{symbol}: before sequence build ({len(scaled)} rows)")
        x_arr, y_arr, seq_end = _build_sequences(scaled, y, self.sequence_length)
        if len(x_arr) < 200:
            raise ValueError(f"Only {len(x_arr)} sequences available for {symbol}; refusing to train")

        split = int(np.searchsorted(seq_end, row_split))
        split = max(min(split, len(x_arr) - 1), 1)
        train_ds = TensorDataset(torch.from_numpy(x_arr[:split]), torch.from_numpy(y_arr[:split]))
        val_x = torch.from_numpy(x_arr[split:])
        val_y = torch.from_numpy(y_arr[split:])
        _log_memory_checkpoint(f"{symbol}: after tensor creation")

        model = LSTMPriceModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # Barrier labels are dominated by NO_TRADE. With an unweighted loss the
        # cheapest way to minimise error is to predict NO_TRADE unconditionally
        # -- high accuracy, zero trading value. Inverse-frequency weights make
        # the directional classes worth learning.
        counts = np.bincount(y_arr[:split], minlength=3).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (3.0 * counts)
        logger.info(
            f"{symbol}: train label counts LONG/SHORT/NO_TRADE={counts.astype(int).tolist()} "
            f"class_weights={[round(w, 3) for w in weights.tolist()]}"
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        best_score = -1.0
        best_state = None
        best_metrics: dict[str, float] = {}
        patience = 8
        stale = 0

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            metrics = self._evaluate(model, val_x, val_y, criterion)
            if epoch == 0:
                _log_memory_checkpoint(f"{symbol}: after epoch 1")

            # Selected on directional precision, not loss. A checkpoint with the
            # lowest loss is often the one that learned to abstain; what this
            # system needs is the checkpoint whose LONG/SHORT calls are most
            # often right.
            score = metrics["directional_precision"]
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = metrics
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        if best_state:
            model.load_state_dict(best_state)
        if not best_metrics:
            best_metrics = self._evaluate(model, val_x, val_y, criterion)

        self._save_model(
            symbol,
            model,
            means.to_dict(),
            stds.to_dict(),
            metrics=best_metrics,
            label_counts=label_distribution(y_arr),
            rows_trained=int(len(x_arr)),
            horizon=horizon,
            atr_multiple=atr_multiple,
        )
        logger.info(f"{symbol}: LSTM metrics {best_metrics}")
        return best_metrics

    def _evaluate(self, model: Any, x: Any, y: Any, criterion: Any) -> dict[str, float]:
        if len(x) == 0:
            return {"val_accuracy": 0.0, "val_loss": float("inf"), "directional_precision": 0.0, "directional_calls": 0.0}
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss = float(criterion(logits, y).item())
            preds = torch.argmax(logits, dim=1)
            accuracy = float((preds == y).float().mean().item())

            directional = preds != SIGNAL_TO_INDEX["NO_TRADE"]
            calls = int(directional.sum().item())
            if calls:
                correct = int(((preds == y) & directional).sum().item())
                precision = correct / calls
            else:
                precision = 0.0
        return {
            "val_accuracy": accuracy,
            "val_loss": loss,
            "directional_precision": float(precision),
            "directional_calls": float(calls),
        }

    # ----------------------------------------------------------------- plumbing

    def _prepare_features(self, symbol: str, frame: pd.DataFrame) -> np.ndarray | None:
        meta = self._meta.get(symbol)
        if not meta:
            self._load_meta(symbol)
            meta = self._meta.get(symbol)
        if not meta:
            return None
        if list(meta.get("features", [])) != list(MODEL_FEATURES):
            logger.warning(
                f"{symbol}: saved model was trained on a different feature set; refusing to predict. "
                f"Retrain with --mode=train_models."
            )
            return None

        built = build_model_features(frame).ffill().fillna(0.0)
        means = pd.Series(meta["mean"]).reindex(list(MODEL_FEATURES)).fillna(0.0)
        stds = pd.Series(meta["std"]).reindex(list(MODEL_FEATURES)).replace(0, 1.0).fillna(1.0)
        scaled = (built - means) / stds
        return np.nan_to_num(scaled.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _weight_path(self, symbol: str) -> Path:
        return self.weights_dir / f"lstm_{symbol}.pt"

    def _meta_path(self, symbol: str) -> Path:
        return self.weights_dir / f"lstm_{symbol}.json"

    def _load_model(self, symbol: str) -> Any | None:
        if symbol in self._models:
            return self._models[symbol]
        path = self._weight_path(symbol)
        if not path.exists() or LSTMPriceModel is None:
            return None
        self._load_meta(symbol)
        meta = self._meta.get(symbol, {})
        if int(meta.get("model_version", 0)) != MODEL_VERSION:
            logger.warning(
                f"{symbol}: weights are model_version={meta.get('model_version')} but this build expects "
                f"{MODEL_VERSION}; refusing to load. Retrain with --mode=train_models."
            )
            return None
        model = LSTMPriceModel()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        self._models[symbol] = model
        return model

    def _load_meta(self, symbol: str) -> None:
        path = self._meta_path(symbol)
        if path.exists():
            try:
                self._meta[symbol] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"{symbol}: could not read model metadata: {exc}")

    def model_quality(self, symbol: str) -> float:
        if symbol not in self._meta:
            self._load_meta(symbol)
        return float(self._meta.get(symbol, {}).get("directional_precision", 0.0) or 0.0)

    def _save_model(
        self,
        symbol: str,
        model: Any,
        means: dict[str, float],
        stds: dict[str, float],
        *,
        metrics: dict[str, float],
        label_counts: dict[str, int],
        rows_trained: int,
        horizon: int,
        atr_multiple: float,
    ) -> None:
        torch.save(model.state_dict(), self._weight_path(symbol))
        meta = {
            "model_version": MODEL_VERSION,
            "features": list(MODEL_FEATURES),
            "sequence_length": self.sequence_length,
            "mean": means,
            "std": stds,
            "val_accuracy": float(metrics.get("val_accuracy", 0.0)),
            "val_loss": float(metrics.get("val_loss", 0.0)),
            "directional_precision": float(metrics.get("directional_precision", 0.0)),
            "directional_calls": float(metrics.get("directional_calls", 0.0)),
            "label_counts": label_counts,
            "rows_trained": rows_trained,
            "horizon": horizon,
            "atr_multiple": atr_multiple,
            "trained_at": datetime.now(UTC).isoformat(),
        }
        self._meta_path(symbol).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        self._models[symbol] = model
        self._meta[symbol] = meta


def _build_sequences(scaled: np.ndarray, y: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stride-tricks sequence view, materialised once.

    Building this with a Python list of 17k slices peaked at several hundred MB
    of intermediate objects; the strided view plus one copy does not.
    """
    total = len(scaled) - sequence_length
    if total <= 0:
        return np.empty((0, sequence_length, scaled.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(scaled, (sequence_length, scaled.shape[1]))
    x_arr = np.ascontiguousarray(windows[:total, 0], dtype=np.float32)
    y_arr = y[sequence_length : sequence_length + total].astype(np.int64)
    seq_end = np.arange(sequence_length, sequence_length + total, dtype=np.int64)
    return x_arr, y_arr, seq_end


def generate_labels(frame: pd.DataFrame, horizon: int = 12, atr_multiple: float = 1.2) -> pd.Series:
    """Backwards-compatible alias for the first-touch labeller."""
    return triple_barrier_labels(frame, horizon=horizon, atr_multiple=atr_multiple)
