from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis.features import (
    MODEL_FEATURES,
    RL_OBSERVATION_SIZE,
    build_rl_observation,
)
from config import BASE_DIR, settings
from utils.logger import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as _gym_import_error:  # pragma: no cover
    gym = None
    spaces = None
    logger.error(f"gymnasium failed to import; the RL agent will be unable to train: {_gym_import_error!r}")

try:
    from stable_baselines3 import PPO
except Exception as _sb3_import_error:  # pragma: no cover
    PPO = None
    logger.error(f"stable-baselines3 failed to import; the RL agent will be unable to train: {_sb3_import_error!r}")


ACTIONS = {0: "BUY", 1: "SELL", 2: "HOLD", 3: "CLOSE_LONG", 4: "CLOSE_SHORT"}
ACTION_INDEX = {value: key for key, value in ACTIONS.items()}

AGENT_VERSION = 2
SEGMENT_KEY = "__segment"


@dataclass(frozen=True)
class RLDecision:
    action: str
    confidence: float
    reason: str | None = None
    probabilities: dict[str, float] | None = None


def _feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.array(
        [[_finite(row.get(name, 0.0)) for name in MODEL_FEATURES] for row in rows],
        dtype=np.float32,
    )


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def fit_scaler(matrix: np.ndarray) -> dict[str, list[float]]:
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def apply_scaler(matrix: np.ndarray, scaler: dict[str, list[float]] | None) -> np.ndarray:
    if not scaler:
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(scaler.get("mean", []), dtype=np.float32)
    std = np.asarray(scaler.get("std", []), dtype=np.float32)
    if mean.shape[-1] != matrix.shape[-1] or std.shape[-1] != matrix.shape[-1]:
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.where(std == 0, 1.0, std)
    return np.nan_to_num((matrix - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


if gym and spaces:

    class TradingEnv(gym.Env):
        """Episodic trading environment over one or more contiguous symbol segments.

        Two things the previous version got wrong and this one does not:

        * Rows from every symbol were concatenated into one continuous tape, so
          the step from the last BTC bar to the first ETH bar produced a
          price "return" of several thousand percent and a reward to match.
          Rows now carry a segment id and the episode resets at each boundary.

        * Reward only arrived on an explicit CLOSE action, so the agent had to
          discover a two-step credit chain before it learned anything. It is now
          marked to market every bar, with trading costs charged on entry and
          exit, which is both a denser signal and a truthful one.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            rows: Sequence[Mapping[str, Any]],
            scaler: dict[str, list[float]] | None = None,
            fee_rate: float = 0.001,
            max_drawdown: float = 0.15,
            stop_atr_multiple: float = 1.5,
        ) -> None:
            super().__init__()
            if not rows:
                raise ValueError("TradingEnv requires at least one row")
            self.rows = list(rows)
            self.scaler = scaler
            self.fee_rate = float(fee_rate)
            self.max_drawdown = float(max_drawdown)
            self.stop_atr_multiple = float(stop_atr_multiple)

            self.features = apply_scaler(_feature_matrix(self.rows), scaler)
            self.close = np.array([_finite(row.get("close", 0.0)) for row in self.rows], dtype=np.float64)
            self.atr_pct = np.array([abs(_finite(row.get("atr_pct", 0.0))) for row in self.rows], dtype=np.float64)
            self.segments = np.array([int(row.get(SEGMENT_KEY, 0)) for row in self.rows], dtype=np.int64)

            self.action_space = spaces.Discrete(len(ACTIONS))
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(RL_OBSERVATION_SIZE,), dtype=np.float32
            )
            self.index = 0
            self.position = 0
            self.entry_price = 0.0
            self.bars_held = 0
            self.equity = 1.0
            self.peak = 1.0

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            self.index = 0
            self._flatten()
            self.equity = 1.0
            self.peak = 1.0
            return self._obs(), {}

        def _flatten(self) -> None:
            self.position = 0
            self.entry_price = 0.0
            self.bars_held = 0

        def step(self, action: int):
            price = float(self.close[self.index])
            reward = 0.0
            fee = self.fee_rate * 100.0

            if action == ACTION_INDEX["BUY"] and self.position == 0:
                self.position = 1
                self.entry_price = price
                self.bars_held = 0
                reward -= fee
            elif action == ACTION_INDEX["SELL"] and self.position == 0:
                self.position = -1
                self.entry_price = price
                self.bars_held = 0
                reward -= fee
            elif action == ACTION_INDEX["CLOSE_LONG"] and self.position == 1:
                self._flatten()
                reward -= fee
            elif action == ACTION_INDEX["CLOSE_SHORT"] and self.position == -1:
                self._flatten()
                reward -= fee
            elif action == ACTION_INDEX["HOLD"] and self.position == 0:
                reward += self._flat_bonus(self.index)

            next_index = self.index + 1
            terminated = next_index >= len(self.rows) - 1
            crossed_segment = (not terminated) and self.segments[next_index] != self.segments[self.index]

            if not terminated and not crossed_segment and price > 0:
                next_price = float(self.close[next_index])
                step_return = (next_price - price) / price
                if self.position:
                    reward += self.position * step_return * 100.0
                    self.bars_held += 1
                    self.equity *= 1.0 + self.position * step_return
                    self.peak = max(self.peak, self.equity)
                    drawdown = (self.peak - self.equity) / max(self.peak, 1e-12)
                    if drawdown > self.max_drawdown:
                        reward -= 10.0 * drawdown

            if crossed_segment:
                # New symbol: bank nothing across the seam, just go flat.
                self._flatten()
                self.equity = 1.0
                self.peak = 1.0

            self.index = next_index
            return self._obs(), float(reward), bool(terminated), False, {}

        def _flat_bonus(self, index: int) -> float:
            """Small credit for standing aside in directionless, quiet tape."""
            row = self.rows[index]
            adx_norm = abs(_finite(row.get("adx_norm", 0.0)))
            bb_width = abs(_finite(row.get("bb_width_pct", 0.5)))
            return 0.05 if adx_norm < 0.4 and bb_width < 0.35 else 0.0

        def _unrealized_r(self, index: int) -> float:
            if not self.position or self.entry_price <= 0:
                return 0.0
            price = float(self.close[index])
            atr = self.atr_pct[index] * price
            risk = self.stop_atr_multiple * atr
            if risk <= 0:
                return 0.0
            return float(self.position * (price - self.entry_price) / risk)

        def _obs(self) -> np.ndarray:
            index = min(self.index, len(self.rows) - 1)
            drawdown = (self.peak - self.equity) / max(self.peak, 1e-12)
            return build_rl_observation(
                self.features[index],
                position=float(self.position),
                unrealized_r=self._unrealized_r(index),
                bars_held=float(self.bars_held),
                drawdown=float(drawdown),
            )

else:
    TradingEnv = None  # type: ignore[assignment]


class RLAgentService:
    def __init__(self, weights_dir: Path | None = None) -> None:
        self.weights_dir = weights_dir or (BASE_DIR / "models" / "weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._agent: Any | None = None
        self._scaler: dict[str, list[float]] | None = None
        self._meta: dict[str, Any] | None = None

    # ------------------------------------------------------------------ decide

    def decide(
        self,
        features: Mapping[str, Any],
        position: float = 0.0,
        unrealized_r: float = 0.0,
        bars_held: float = 0.0,
        drawdown: float = 0.0,
    ) -> RLDecision:
        if PPO is None or TradingEnv is None:
            return RLDecision("HOLD", 0.0, "stable-baselines3 unavailable")
        try:
            agent = self._load_agent()
            if agent is None:
                return RLDecision("HOLD", 0.0, "PPO weights missing")
            raw = np.array([[_finite(features.get(name, 0.0)) for name in MODEL_FEATURES]], dtype=np.float32)
            scaled = apply_scaler(raw, self._scaler)[0]
            obs = build_rl_observation(
                scaled,
                position=position,
                unrealized_r=unrealized_r,
                bars_held=bars_held,
                drawdown=drawdown,
            )
            action, _ = agent.predict(obs, deterministic=True)
            action_name = ACTIONS[int(action)]
            probabilities = self._action_probabilities(agent, obs)
            confidence = probabilities.get(action_name, 0.0) if probabilities else 0.5
            return RLDecision(action_name, float(confidence), probabilities=probabilities)
        except Exception as exc:
            logger.exception(f"RL decision failed: {exc}")
            return RLDecision("HOLD", 0.0, str(exc))

    def _action_probabilities(self, agent: Any, obs: np.ndarray) -> dict[str, float] | None:
        """Real policy probabilities instead of a hardcoded constant.

        The old code reported 0.75 for every decision, which made the gate's
        'exceptional confidence' branch (rl_confidence >= 0.90) unreachable and
        left the dashboard's RL column meaningless.
        """
        try:
            tensor, _ = agent.policy.obs_to_tensor(obs)
            distribution = agent.policy.get_distribution(tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
            return {ACTIONS[i]: float(probs[i]) for i in range(min(len(probs), len(ACTIONS)))}
        except Exception:
            return None

    def is_ready(self) -> bool:
        return self._load_agent() is not None

    # ------------------------------------------------------------------- train

    def train(
        self,
        rows: Sequence[Mapping[str, Any]],
        timesteps: int = 60_000,
        fee_rate: float | None = None,
    ) -> dict[str, Any]:
        if PPO is None or TradingEnv is None:
            raise RuntimeError(
                "stable-baselines3 is unavailable in this environment, so the RL agent cannot be "
                "trained. Check that it installed successfully in the build."
            )
        if len(rows) < 500:
            raise ValueError(f"RL training needs at least 500 rows, got {len(rows)}")

        scaler = fit_scaler(_feature_matrix(rows))
        env = TradingEnv(rows, scaler=scaler, fee_rate=fee_rate if fee_rate is not None else settings.taker_fee_rate)
        agent = PPO("MlpPolicy", env, verbose=0, n_steps=1024, batch_size=64, ent_coef=0.01)
        agent.learn(total_timesteps=timesteps)
        self._persist(agent, scaler, {"rows": len(rows), "timesteps": timesteps, "mode": "full"})
        return {"rows": len(rows), "timesteps": timesteps}

    def online_update(
        self,
        rows: Sequence[Mapping[str, Any]],
        timesteps: int = 4_000,
        min_rows: int = 500,
    ) -> dict[str, Any] | None:
        """Incremental refresh, with a floor on how little data may reshape the policy.

        This used to run on ~90 rows every 60 seconds, which is not learning --
        it is overwriting a policy trained on tens of thousands of bars with
        whatever the last hour looked like. It now refuses to run on thin data
        and keeps the previous weights recoverable.
        """
        if PPO is None or TradingEnv is None:
            return None
        if len(rows) < min_rows:
            logger.info(f"RL online update skipped: {len(rows)} rows is below the {min_rows}-row floor")
            return None
        try:
            agent = self._load_agent()
            scaler = self._scaler or fit_scaler(_feature_matrix(rows))
            env = TradingEnv(rows, scaler=scaler, fee_rate=settings.taker_fee_rate)
            if agent is None:
                agent = PPO("MlpPolicy", env, verbose=0, n_steps=1024, batch_size=64, ent_coef=0.01)
                reset_timesteps = True
            else:
                agent.set_env(env)
                reset_timesteps = False
            self._backup_weights()
            agent.learn(total_timesteps=timesteps, reset_num_timesteps=reset_timesteps)
            self._persist(agent, scaler, {"rows": len(rows), "timesteps": timesteps, "mode": "online"})
            return {"rows": len(rows), "timesteps": timesteps}
        except Exception as exc:
            logger.warning(f"RL online update failed, keeping existing policy: {exc}")
            return None

    # ---------------------------------------------------------------- plumbing

    def _agent_path(self) -> Path:
        return self.weights_dir / "ppo_trading_agent.zip"

    def _scaler_path(self) -> Path:
        return self.weights_dir / "ppo_trading_agent.json"

    def _backup_path(self) -> Path:
        return self.weights_dir / "ppo_trading_agent.previous.zip"

    def _backup_weights(self) -> None:
        source = self._agent_path()
        if source.exists():
            try:
                self._backup_path().write_bytes(source.read_bytes())
            except Exception as exc:
                logger.warning(f"Could not back up RL weights: {exc}")

    def _persist(self, agent: Any, scaler: dict[str, list[float]], info: dict[str, Any]) -> None:
        agent.save(self._agent_path().with_suffix(""))
        meta = {
            "agent_version": AGENT_VERSION,
            "features": list(MODEL_FEATURES),
            "observation_size": RL_OBSERVATION_SIZE,
            "scaler": scaler,
            **info,
        }
        self._scaler_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._agent = agent
        self._scaler = scaler
        self._meta = meta

    def _load_agent(self) -> Any | None:
        if self._agent is not None:
            return self._agent
        path = self._agent_path()
        if not path.exists() or PPO is None:
            return None
        meta_path = self._scaler_path()
        if not meta_path.exists():
            logger.warning(
                "RL weights found without metadata; refusing to load. The observation layout changed, "
                "so weights from an earlier build would be read with features in the wrong slots. "
                "Retrain with --mode=train_models."
            )
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not read RL metadata: {exc}")
            return None
        if int(meta.get("agent_version", 0)) != AGENT_VERSION or list(meta.get("features", [])) != list(MODEL_FEATURES):
            logger.warning(
                f"RL weights are agent_version={meta.get('agent_version')} with a different feature set; "
                f"refusing to load. Retrain with --mode=train_models."
            )
            return None
        try:
            self._agent = PPO.load(path)
        except Exception as exc:
            logger.warning(f"Could not load RL weights: {exc}")
            return None
        self._meta = meta
        self._scaler = meta.get("scaler")
        return self._agent
