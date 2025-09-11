# mmgym_lite/envs/trading.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any

from rewards.Base import RewardFunction
from rewards.RewardFunctions import PnLReward  # default

ASSET_PRICE, INVENTORY, TIMEIDX, CASH = 0, 1, 2, 3

class TradingEnv(gym.Env):
    """
    Vectorized, modular market-making env with pluggable rewards.
    Internal batch size = N = dynamics.num_traj (many paths at once).
    For RL, set return_vectorized=False (single obs/action/reward view).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dynamics,                      # LimitOrderDynamics or LimitAndMarketDynamics
        T: float,
        M: int,
        reward_fn: RewardFunction | None = None,
        init_inventory: int = 0,
        init_cash: float = 0.0,
        max_inventory: int = 10_000,
        seed: int | None = None,
        return_vectorized: bool = False,
    ):
        super().__init__()
        self.dyn = dynamics
        self.T, self.M = float(T), int(M)
        self.dt = self.T / self.M
        self.N = self.dyn.num_traj
        self.max_inventory = int(max_inventory)
        self.return_vectorized = bool(return_vectorized)

        # state vectors (N,)
        self.price = np.copy(self.dyn.mid.state[:, 0])
        self.q     = np.full(self.N, init_inventory, dtype=int)
        self.cash  = np.full(self.N, init_cash, dtype=float)
        self.t_idx = 0

        self.rng = np.random.default_rng(seed)
        self.action_space = self.dyn.get_action_space()
        self.observation_space = spaces.Box(
            low = np.array([0.0, -np.inf, 0.0, -np.inf], np.float32),
            high= np.array([np.inf,  np.inf, float(self.M),  np.inf], np.float32),
            dtype=np.float32
        )

        self.reward_fn = reward_fn or PnLReward()
        self.reward_fn.reset()

    # -------- helpers --------
    def _obs_first(self):
        return np.array([self.price[0], self.q[0], self.t_idx, self.cash[0]], dtype=np.float32)

    def _obs_batch(self):
        return np.stack([self.price, self.q, np.full(self.N, self.t_idx), self.cash], axis=1).astype(np.float32)

    def _pack_state(self) -> np.ndarray:
        # full batch (N,4)
        return np.stack([self.price, self.q, np.full(self.N, self.t_idx), self.cash], axis=1).astype(np.float32)

    # -------- Gym API --------
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.dyn.mid.reset()
        self.dyn.arr.reset()
        self.price = np.copy(self.dyn.mid.state[:, 0])
        self.q[:] = 0
        self.cash[:] = 0.0
        self.t_idx = 0
        self.reward_fn.reset()
        obs = self._obs_batch() if self.return_vectorized else self._obs_first()
        return obs, {}

    def step(self, action: np.ndarray):
        # Broadcast action to (N, A)
        action = np.asarray(action, float)
        if action.ndim == 1:
            action = np.repeat(action.reshape(1, -1), self.N, axis=0)
        elif action.shape[0] != self.N:
            raise ValueError(f"action batch size {action.shape[0]} != num_traj {self.N}")

        # Snapshot current batch state BEFORE transition
        current_state = self._pack_state()

        # 1) midprice step
        mid_out = self.dyn.mid.step()
        self.price = mid_out["price"]

        # 2) split half-spreads (first two cols always half-spreads)
        half_spreads = action[:, :2]

        # 3) arrivals + fills
        arrivals, fills = self.dyn.arrivals_and_fills(half_spreads, self.rng)

        # 4) market orders (if available)
        dq_mo = np.zeros(self.N, int)
        dcash_mo = np.zeros(self.N, float)
        if action.shape[1] >= 4:
            dq_mo, dcash_mo = self.dyn.market_order_deltas(action)

        # 5) limit-order deltas
        dq_lim, dcash_lim = self.dyn.cash_inventory_delta(half_spreads, arrivals, fills)

        # 6) apply updates within bounds
        dq = dq_mo + dq_lim
        dcash = dcash_mo + dcash_lim

        can_buy  = (self.q + dq) <=  self.max_inventory
        can_sell = (self.q + dq) >= -self.max_inventory
        ok = can_buy & can_sell

        self.q[ok]    += dq[ok]
        self.cash[ok] += dcash[ok]

        # 7) advance time index
        self.t_idx += 1
        terminated = self.t_idx >= self.M
        done_vec = np.full(self.N, terminated, dtype=bool)

        # Next state AFTER transition
        next_state = self._pack_state()

        # 8) info payload for rewards/analytics
        info_batch = {
            "arrivals": arrivals,              # (N,2) bool
            "fills": fills,                    # (N,2) bool
            "half_spreads": half_spreads,      # (N,2) float
            "dq": dq, "dcash": dcash           # (N,) arrays
        }

        # 9) reward via pluggable interface (vectorized)
        rewards = self.reward_fn.calculate(current_state, action, next_state, done_vec, info_batch)

        # 10) Output
        if self.return_vectorized:
            obs = next_state
            info = info_batch
            return obs, rewards.astype(np.float32), done_vec, False, info
        else:
            obs = next_state[0]
            info = {k: (v[0] if isinstance(v, np.ndarray) and v.shape[0] == self.N else v) for k, v in info_batch.items()}
            return obs.astype(np.float32), float(rewards[0]), bool(terminated), False, info
