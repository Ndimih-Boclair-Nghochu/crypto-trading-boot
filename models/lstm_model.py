from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import BASE_DIR
from utils.logger import logger

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi_14",
    "macd_hist",
    "atr_14",
    "bb_percent_b",
    "ema_21",
    "ema_50",
    "adx",
    "obv",
    "fear_greed",
]
INDEX_TO_SIGNAL = {0: "LONG", 1: "SHORT", 2: "NO_TRADE"}
SIGNAL_TO_INDEX = {value: key for key, value in INDEX_TO_SIGNAL.items()}


@dataclass(frozen=True)
class LSTMSignal:
    direction: str
    confidence: float
    probabilities: dict[str, float]
    reason: str | None = None


if torch and nn:

    class LSTMPriceModel(nn.Module):
        def __init__(self, input_size: int = len(FEATURES), hidden_size: int = 256, num_layers: int = 2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(hidden_size, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 3)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            output, _ = self.lstm(x)
            last = output[:, -1, :]
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
        self._scalers: dict[str, dict[str, list[float]]] = {}

    def predict(self, symbol: str, indicators: dict[str, Any]) -> LSTMSignal:
        if torch is None or LSTMPriceModel is None:
            return LSTMSignal("NO_TRADE", 0.0, {"LONG": 0.0, "SHORT": 0.0, "NO_TRADE": 1.0}, "PyTorch unavailable")
        try:
            model = self._load_model(symbol)
            if model is None:
                return LSTMSignal("NO_TRADE", 0.0, {"LONG": 0.0, "SHORT": 0.0, "NO_TRADE": 1.0}, "weights missing")
            rows = indicators.get("series_tail", [])
            frame = pd.DataFrame(rows)
            if len(frame) < self.sequence_length:
                return LSTMSignal("NO_TRADE", 0.0, {"LONG": 0.0, "SHORT": 0.0, "NO_TRADE": 1.0}, "insufficient sequence")
            features = self._prepare_features(symbol, frame.tail(self.sequence_length))
            tensor = torch.tensor(features[None, :, :], dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probabilities = {INDEX_TO_SIGNAL[i]: float(probs[i]) for i in range(3)}
            idx = int(np.argmax(probs))
            return LSTMSignal(INDEX_TO_SIGNAL[idx], float(probs[idx]), probabilities)
        except Exception as exc:
            logger.exception(f"LSTM prediction failed for {symbol}: {exc}")
            return LSTMSignal("NO_TRADE", 0.0, {"LONG": 0.0, "SHORT": 0.0, "NO_TRADE": 1.0}, str(exc))

    def train(self, symbol: str, frame: pd.DataFrame, epochs: int = 30, batch_size: int = 128) -> dict[str, float]:
        if torch is None or LSTMPriceModel is None or DataLoader is None or TensorDataset is None:
            logger.warning("PyTorch unavailable; skipping LSTM training.")
            return {"val_accuracy": 0.0, "val_loss": float("inf")}
        if len(frame) < self.sequence_length + 20:
            raise ValueError("Not enough rows to train LSTM")

        dataset = frame.copy().replace([np.inf, -np.inf], np.nan).ffill().bfill()
        labels = generate_labels(dataset)
        dataset["label"] = labels
        dataset = dataset.dropna(subset=FEATURES + ["label"])

        values = dataset[FEATURES].astype(float)
        means = values.mean()
        stds = values.std().replace(0, 1)
        scaled = ((values - means) / stds).to_numpy(dtype=np.float32)
        y = dataset["label"].astype(int).to_numpy(dtype=np.int64)

        sequences: list[np.ndarray] = []
        targets: list[int] = []
        for idx in range(self.sequence_length, len(scaled)):
            sequences.append(scaled[idx - self.sequence_length : idx])
            targets.append(int(y[idx]))
        x_arr = np.stack(sequences)
        y_arr = np.array(targets)
        split = max(int(len(x_arr) * 0.8), 1)
        train_ds = TensorDataset(torch.tensor(x_arr[:split]), torch.tensor(y_arr[:split]))
        val_x = torch.tensor(x_arr[split:])
        val_y = torch.tensor(y_arr[split:])

        model = LSTMPriceModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        best_loss = float("inf")
        patience = 5
        stale = 0
        best_state = None

        for _epoch in range(epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            val_loss, val_accuracy = self._evaluate(model, val_x, val_y, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        if best_state:
            model.load_state_dict(best_state)
        self._save_model(symbol, model, means.to_dict(), stds.to_dict())
        return {"val_accuracy": float(val_accuracy), "val_loss": float(best_loss)}

    def _evaluate(self, model: Any, x: Any, y: Any, criterion: Any) -> tuple[float, float]:
        if len(x) == 0:
            return float("inf"), 0.0
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y).item()
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y).float().mean().item()
        return float(loss), float(accuracy)

    def _prepare_features(self, symbol: str, frame: pd.DataFrame) -> np.ndarray:
        data = frame.copy()
        for feature in FEATURES:
            if feature not in data:
                data[feature] = 0.0
        data = data[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        scaler = self._scalers.get(symbol)
        if scaler is None:
            self._load_scaler(symbol)
            scaler = self._scalers.get(symbol)
        if scaler:
            means = pd.Series(scaler["mean"])
            stds = pd.Series(scaler["std"]).replace(0, 1)
            data = (data - means) / stds
        return data.to_numpy(dtype=np.float32)

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
        model = LSTMPriceModel()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        self._models[symbol] = model
        self._load_scaler(symbol)
        return model

    def _load_scaler(self, symbol: str) -> None:
        path = self._meta_path(symbol)
        if path.exists():
            self._scalers[symbol] = json.loads(path.read_text(encoding="utf-8"))

    def _save_model(self, symbol: str, model: Any, means: dict[str, float], stds: dict[str, float]) -> None:
        torch.save(model.state_dict(), self._weight_path(symbol))
        self._meta_path(symbol).write_text(
            json.dumps({"mean": means, "std": stds}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._models[symbol] = model
        self._scalers[symbol] = {"mean": means, "std": stds}


def generate_labels(frame: pd.DataFrame, horizon: int = 8, atr_multiple: float = 1.5) -> pd.Series:
    close = frame["close"].astype(float)
    atr_values = frame["atr_14"].astype(float).replace(0, np.nan)
    future_max = close.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    future_min = close.shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    up = future_max - close
    down = close - future_min
    labels = pd.Series(SIGNAL_TO_INDEX["NO_TRADE"], index=frame.index)
    labels[up >= atr_multiple * atr_values] = SIGNAL_TO_INDEX["LONG"]
    labels[down >= atr_multiple * atr_values] = SIGNAL_TO_INDEX["SHORT"]
    labels[(up >= atr_multiple * atr_values) & (down >= atr_multiple * atr_values)] = SIGNAL_TO_INDEX["NO_TRADE"]
    return labels
