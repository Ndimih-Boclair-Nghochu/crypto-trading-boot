from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import BASE_DIR
from utils.logger import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover
    PPO = None


ACTIONS = {0: "BUY", 1: "SELL", 2: "HOLD", 3: "CLOSE_LONG", 4: "CLOSE_SHORT"}
ACTION_INDEX = {value: key for key, value in ACTIONS.items()}


@dataclass(frozen=True)
class RLDecision:
    action: str
    confidence: float
    reason: str | None = None


if gym and spaces:

    class TradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, rows: list[dict[str, float]], min_rr: float = 2.0, max_drawdown: float = 0.10) -> None:
            super().__init__()
            self.rows = rows
            self.min_rr = min_rr
            self.max_drawdown = max_drawdown
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
            self.index = 0
            self.position = 0
            self.entry_price = 0.0
            self.equity = 1.0
            self.peak = 1.0

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            self.index = 0
            self.position = 0
            self.entry_price = 0.0
            self.equity = 1.0
            self.peak = 1.0
            return self._obs(), {}

        def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            row = self.rows[self.index]
            next_row = self.rows[min(self.index + 1, len(self.rows) - 1)]
            price = float(row.get("close", 0))
            next_price = float(next_row.get("close", price))
            reward = 0.0

            if action == ACTION_INDEX["BUY"]:
                if self.position == 0:
                    self.position = 1
                    self.entry_price = price
                    reward -= self._rr_penalty(row)
            elif action == ACTION_INDEX["SELL"]:
                if self.position == 0:
                    self.position = -1
                    self.entry_price = price
                    reward -= self._rr_penalty(row)
            elif action == ACTION_INDEX["CLOSE_LONG"] and self.position == 1:
                reward += self._close_reward(price)
            elif action == ACTION_INDEX["CLOSE_SHORT"] and self.position == -1:
                reward += self._close_reward(price)
            elif action == ACTION_INDEX["HOLD"] and self.position == 0:
                reward += self._no_trade_bonus(row)

            if self.position:
                pnl = self.position * (next_price - self.entry_price) / max(self.entry_price, 1e-12)
                self.equity = 1.0 + pnl
                self.peak = max(self.peak, self.equity)
                drawdown = (self.peak - self.equity) / max(self.peak, 1e-12)
                if drawdown > self.max_drawdown:
                    reward -= 10 * drawdown

            self.index += 1
            terminated = self.index >= len(self.rows) - 2
            return self._obs(), float(reward), terminated, False, {}

        def _close_reward(self, price: float) -> float:
            pnl = self.position * (price - self.entry_price) / max(self.entry_price, 1e-12)
            reward = pnl * 100
            if pnl > 0:
                reward *= 1.5
            else:
                reward *= 2.0
            self.position = 0
            self.entry_price = 0.0
            return reward

        def _rr_penalty(self, row: dict[str, float]) -> float:
            rr = float(row.get("reward_risk", 2.0) or 0)
            return 1.0 if rr < self.min_rr else 0.0

        def _no_trade_bonus(self, row: dict[str, float]) -> float:
            adx = float(row.get("adx", 0) or 0)
            bb_width_percentile = float(row.get("bb_width_percentile", 50) or 50)
            return 0.1 if adx < 20 and bb_width_percentile < 35 else 0.0

        def _obs(self) -> np.ndarray:
            row = self.rows[self.index]
            keys = [
                "close",
                "rsi_14",
                "macd_hist",
                "atr_14",
                "bb_percent_b",
                "ema_21",
                "ema_50",
                "adx",
                "obv",
                "fear_greed",
                "lstm_confidence",
                "confluence_score",
            ]
            values = [float(row.get(key, 0) or 0) for key in keys]
            drawdown = (self.peak - self.equity) / max(self.peak, 1e-12)
            values.extend([float(self.position), float(self.equity - 1.0), float(drawdown), float(self.entry_price)])
            return np.array(values, dtype=np.float32)

else:
    TradingEnv = None  # type: ignore[assignment]


class RLAgentService:
    def __init__(self, weights_dir: Path | None = None) -> None:
        self.weights_dir = weights_dir or (BASE_DIR / "models" / "weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._agent: Any | None = None

    def decide(self, state: dict[str, Any]) -> RLDecision:
        if PPO is None or TradingEnv is None:
            return RLDecision("HOLD", 1.0, "stable-baselines3 unavailable")
        try:
            agent = self._load_agent()
            if agent is None:
                return RLDecision("HOLD", 1.0, "PPO weights missing")
            obs = self._state_to_obs(state)
            action, _ = agent.predict(obs, deterministic=True)
            action_name = ACTIONS[int(action)]
            return RLDecision(action_name, 0.75)
        except Exception as exc:
            logger.exception(f"RL decision failed: {exc}")
            return RLDecision("HOLD", 1.0, str(exc))

    def train(self, rows: list[dict[str, float]], timesteps: int = 25_000) -> None:
        if PPO is None or TradingEnv is None:
            logger.warning("stable-baselines3 unavailable; skipping PPO training.")
            return
        env = TradingEnv(rows)
        agent = PPO("MlpPolicy", env, verbose=0)
        agent.learn(total_timesteps=timesteps)
        agent.save(self.weights_dir / "ppo_trading_agent")
        self._agent = agent

    def online_update(self, rows: list[dict[str, float]], timesteps: int = 2_000) -> None:
        if PPO is None or TradingEnv is None:
            return
        env = TradingEnv(rows)
        agent = self._load_agent()
        if agent is None:
            agent = PPO("MlpPolicy", env, verbose=0)
        else:
            agent.set_env(env)
        agent.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        agent.save(self.weights_dir / "ppo_trading_agent")
        self._agent = agent

    def _load_agent(self) -> Any | None:
        if self._agent is not None:
            return self._agent
        path = self.weights_dir / "ppo_trading_agent.zip"
        if not path.exists() or PPO is None:
            return None
        self._agent = PPO.load(path)
        return self._agent

    def _state_to_obs(self, state: dict[str, Any]) -> np.ndarray:
        keys = [
            "close",
            "rsi_14",
            "macd_hist",
            "atr_14",
            "bb_percent_b",
            "ema_21",
            "ema_50",
            "adx",
            "obv",
            "fear_greed",
            "lstm_confidence",
            "confluence_score",
            "open_pnl",
            "drawdown",
            "position",
            "entry_price",
        ]
        return np.array([float(state.get(key, 0) or 0) for key in keys], dtype=np.float32)
