# mmgym_lite/envs/trading.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any
from agents.Agents import as_infinite_half_spreads
from agents.Agents import as_half_spreads
from typing import List, Any, Type, Optional, Union, Sequence, Dict, Tuple
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

from rewards.Base import RewardFunction
from rewards.RewardFunctions import PnLReward  # default

ASSET_PRICE, INVENTORY_INDEX, TIME_INDEX, CASH = 0, 1, 2, 3

# class TradingEnv(gym.Env):
#     """
#     Vectorized, modular market-making env with pluggable rewards.
#     Internal batch size = N = dynamics.num_traj (many paths at once).
#     For RL, set return_vectorized=False (single obs/action/reward view).
#     """
#     metadata = {"render_modes": ["human"]}

#     def __init__(
#         self,
#         dynamics,
#         T: float,
#         M: int,
#         reward_fn: RewardFunction | None = None,
#         init_inventory: int = 0,
#         init_cash: float = 0.0,
#         max_inventory: int = 10_000,
#         seed: int | None = None,
#         return_vectorized: bool = False,
#         static_gamma: float | None = None,
#         static_skew: float | None = None,
#     ):
#         super().__init__()
#         self.dyn = dynamics
#         self.T, self.M = float(T), int(M)
#         self.dt = self.T / self.M
#         self.N = self.dyn.num_traj
#         self.max_inventory = int(max_inventory)
#         self.return_vectorized = bool(return_vectorized)
        
#         # ADD: Static strategy parameters
#         self.static_gamma = static_gamma
#         self.static_skew = static_skew
#         self._use_static_strategy = static_gamma is not None

#         # state vectors (N,)
#         self.price = np.copy(self.dyn.mid.state[:, 0])
#         self.q     = np.full(self.N, init_inventory, dtype=int)
#         self.cash  = np.full(self.N, init_cash, dtype=float)
#         self.t_idx = 0

#         self.rng = np.random.default_rng(seed)
#         self.action_space = self.dyn.get_action_space()
#         self.observation_space = spaces.Box(
#             low = np.array([0.0, -np.inf, 0.0, -np.inf], np.float32),
#             high= np.array([np.inf,  np.inf, float(self.M),  np.inf], np.float32),
#             dtype=np.float32
#         )

#         self.reward_fn = reward_fn or PnLReward()
#         self.reward_fn.reset()

#     # -------- helpers --------
#     def _obs_first(self):
#         return np.array([self.price[0], self.q[0], self.t_idx, self.cash[0]], dtype=np.float32)

#     def _obs_batch(self):
#         return np.stack([self.price, self.q, np.full(self.N, self.t_idx), self.cash], axis=1).astype(np.float32)

#     def _pack_state(self) -> np.ndarray:
#         # full batch (N,4)
#         return np.stack([self.price, self.q, np.full(self.N, self.t_idx), self.cash], axis=1).astype(np.float32)

#     # -------- Gym API --------
#     def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
#         super().reset(seed=seed)
#         self.dyn.mid.reset()
#         self.dyn.arr.reset()
#         self.price = np.copy(self.dyn.mid.state[:, 0])
#         self.q[:] = 0
#         self.cash[:] = 0.0
#         self.t_idx = 0
#         self.reward_fn.reset()
#         obs = self._obs_batch() if self.return_vectorized else self._obs_first()
#         return obs, {}

#     def step(self, action: np.ndarray):
#         # Broadcast action to (N, A)
#         action = np.asarray(action, float)
#         if action.ndim == 1:
#             action = np.repeat(action.reshape(1, -1), self.N, axis=0)
#         elif action.shape[0] != self.N:
#             raise ValueError(f"action batch size {action.shape[0]} != num_traj {self.N}")

#         # Snapshot current batch state BEFORE transition
#         current_state = self._pack_state()

#         # 1) midprice step
#         mid_out = self.dyn.mid.step()
#         self.price = mid_out["price"]

#         # 2) split half-spreads (first two cols always half-spreads)
#         half_spreads = action[:, :2]

#         # 3) arrivals + fills
#         arrivals, fills = self.dyn.arrivals_and_fills(half_spreads, self.rng)

#         # 4) market orders (if available)
#         dq_mo = np.zeros(self.N, int)
#         dcash_mo = np.zeros(self.N, float)
#         if action.shape[1] >= 4:
#             dq_mo, dcash_mo = self.dyn.market_order_deltas(action)

#         # 5) limit-order deltas
#         dq_lim, dcash_lim = self.dyn.cash_inventory_delta(half_spreads, arrivals, fills)

#         # 6) apply updates within bounds
#         dq = dq_mo + dq_lim
#         dcash = dcash_mo + dcash_lim

#         can_buy  = (self.q + dq) <=  self.max_inventory
#         can_sell = (self.q + dq) >= -self.max_inventory
#         ok = can_buy & can_sell

#         self.q[ok]    += dq[ok]
#         self.cash[ok] += dcash[ok]

#         # 7) advance time index
#         self.t_idx += 1
#         terminated = self.t_idx >= self.M
#         done_vec = np.full(self.N, terminated, dtype=bool)

#         # Next state AFTER transition
#         next_state = self._pack_state()

#         # 8) info payload for rewards/analytics
#         info_batch = {
#             "arrivals": arrivals,              # (N,2) bool
#             "fills": fills,                    # (N,2) bool
#             "half_spreads": half_spreads,      # (N,2) float
#             "dq": dq, "dcash": dcash           # (N,) arrays
#         }

#         # 9) reward via pluggable interface (vectorized)
#         rewards = self.reward_fn.calculate(current_state, action, next_state, done_vec, info_batch)

#         # 10) Output
#         if self.return_vectorized:
#             obs = next_state
#             info = info_batch
#             return obs, rewards.astype(np.float32), done_vec, False, info
#         else:
#             obs = next_state[0]
#             info = {k: (v[0] if isinstance(v, np.ndarray) and v.shape[0] == self.N else v) for k, v in info_batch.items()}
#             return obs.astype(np.float32), float(rewards[0]), bool(terminated), False, info

class TradingEnv(gym.Env):
    """
    Vectorized, modular market-making env with pluggable rewards.
    Now supports both static AS operation and RL-controlled parameter adjustment.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dynamics,
        T: float,
        M: int,
        reward_fn: RewardFunction | None = None,
        init_inventory: int = 0,
        init_cash: float = 0.0,
        max_inventory: int = 10_000,
        seed: int | None = None,
        return_vectorized: bool = False,
        # NEW: RL control parameters
        enable_rl_control: bool = False,
        initial_gamma: float = 0.1,
        initial_skew: float = 0.0,
        gamma_choices: list = None,
        skew_choices: list = None,
        as_mode: str = "finite",
        k_fill: float | None = None,
        omega: float | None = None
    ):
        super().__init__()
        self.dyn = dynamics
        self.T, self.M = float(T), int(M)
        self.dt = self.T / self.M
        self.N = self.dyn.num_traj
        self.max_inventory = int(max_inventory)
        self.return_vectorized = bool(return_vectorized)

        # NEW: RL control parameters with defaults
        self.enable_rl_control = enable_rl_control
        self.current_gamma = initial_gamma
        self.current_skew = initial_skew
        self.gamma_choices = gamma_choices or [0.01, 0.1, 0.2, 0.9]
        self.skew_choices = skew_choices or [-0.1, -0.05, 0, 0.05, 0.1]
        self.as_mode = as_mode
        self.k_fill = k_fill if k_fill is not None else getattr(dynamics, "fill_k", 1.0)
        self.omega = omega
        
        # Store AS parameters per trajectory for vectorized mode
        if self.return_vectorized and self.enable_rl_control:
            self.current_gamma_vec = np.full(self.N, initial_gamma, dtype=float)
            self.current_skew_vec = np.full(self.N, initial_skew, dtype=float)
        else:
            self.current_gamma_vec = None
            self.current_skew_vec = None

        # state vectors (N,)
        self.price = np.copy(self.dyn.mid.state[:, 0])
        self.q = np.full(self.N, init_inventory, dtype=int)
        self.cash = np.full(self.N, init_cash, dtype=float)
        self.t_idx = 0

        self.rng = np.random.default_rng(seed)
        
        # MODIFIED: Dynamic action space based on mode
        if enable_rl_control:
            # Discrete action space: choose from gamma_choices × skew_choices
            self.action_space = spaces.Discrete(len(self.gamma_choices) * len(self.skew_choices))
        else:
            # Keep original continuous action space for half-spreads
            self.action_space = self.dyn.get_action_space()

        # MODIFIED: Expand observation space to include current parameters
        obs_low = np.array([0.0, -np.inf, 0.0, -np.inf, min(self.gamma_choices), min(self.skew_choices)], np.float32)
        obs_high = np.array([np.inf, np.inf, float(self.M), np.inf, max(self.gamma_choices), max(self.skew_choices)], np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.reward_fn = reward_fn or PnLReward()
        self.reward_fn.reset()

    # -------- AS Spread Computation --------
    def _compute_as_spreads(self, trajectory_idx: int | slice = 0) -> np.ndarray:
        """Compute AS spreads using current parameters"""
        if self.return_vectorized and self.enable_rl_control:
            # Vectorized mode: compute for all trajectories
            if isinstance(trajectory_idx, slice):
                s = self.price
                q = self.q
                tI = np.full(self.N, self.t_idx)
                gamma_vec = self.current_gamma_vec
                skew_vec = self.current_skew_vec
            else:
                # Single trajectory in vectorized mode
                s = np.array([self.price[trajectory_idx]])
                q = np.array([self.q[trajectory_idx]])
                tI = np.array([self.t_idx])
                gamma_vec = np.array([self.current_gamma_vec[trajectory_idx]])
                skew_vec = np.array([self.current_skew_vec[trajectory_idx]])
        else:
            # Single trajectory mode
            if self.return_vectorized:
                s = np.array([self.price[trajectory_idx]])
                q = np.array([self.q[trajectory_idx]])
            else:
                s = np.array([self.price[0]])
                q = np.array([self.q[0]])
            tI = np.array([self.t_idx])
            gamma_vec = np.array([self.current_gamma])
            skew_vec = np.array([self.current_skew])

        # Import here to avoid circular imports
        if self.as_mode == "infinite":
            halves = as_infinite_half_spreads(s, q, gamma_vec, self.dyn.mid.sigma, self.omega)
        else:
            halves = as_half_spreads(
                inventory=q,
                t_idx=tI,
                T=self.T,
                sigma=self.dyn.mid.sigma,
                gamma=gamma_vec,
                k=self.k_fill,
                dt=self.dt,
            )
        
        # Apply skew adjustment
        bid_half = halves[:, 0] * (1 + skew_vec)
        ask_half = halves[:, 1] * (1 + skew_vec)
        
        # Ensure non-negative spreads
        bid_half = np.maximum(bid_half, 0.0)
        ask_half = np.maximum(ask_half, 0.0)
        
        return np.stack([bid_half, ask_half], axis=1)

    # -------- State Management --------
    def _obs_first(self):
        base_obs = np.array([self.price[0], self.q[0], self.t_idx, self.cash[0]], dtype=np.float32)
        # Append current parameters
        extended_obs = np.concatenate([base_obs, [self.current_gamma, self.current_skew]])
        return extended_obs

    def _obs_batch(self):
        base_obs = np.stack([self.price, self.q, np.full(self.N, self.t_idx), self.cash], axis=1)
        # Append current parameters
        if self.enable_rl_control and self.return_vectorized:
            param_obs = np.stack([self.current_gamma_vec, self.current_skew_vec], axis=1)
        else:
            param_obs = np.tile([self.current_gamma, self.current_skew], (self.N, 1))
        extended_obs = np.concatenate([base_obs, param_obs], axis=1)
        return extended_obs.astype(np.float32)

    def _pack_state(self) -> np.ndarray:
        """Extended state with current parameters"""
        if self.return_vectorized:
            return self._obs_batch()
        else:
            return self._obs_first()

    # -------- Parameter Control --------
    def set_rl_parameters(self, gamma: float, skew: float, trajectory_idx: int | None = None):
        """Update parameters when in RL control mode"""
        if self.enable_rl_control:
            if self.return_vectorized and trajectory_idx is not None:
                self.current_gamma_vec[trajectory_idx] = gamma
                self.current_skew_vec[trajectory_idx] = skew
            elif self.return_vectorized and trajectory_idx is None:
                # Apply to all trajectories
                self.current_gamma_vec[:] = gamma
                self.current_skew_vec[:] = skew
            else:
                self.current_gamma = gamma
                self.current_skew = skew

    def decode_rl_action(self, action: int) -> tuple[float, float]:
        """Convert discrete action to (gamma, skew)"""
        gamma_idx = action // len(self.skew_choices)
        skew_idx = action % len(self.skew_choices)
        return self.gamma_choices[gamma_idx], self.skew_choices[skew_idx]

    def set_mode(self, enable_rl_control: bool):
        """Switch between static and RL-controlled modes"""
        self.enable_rl_control = enable_rl_control
        # Update action space accordingly
        if enable_rl_control:
            self.action_space = spaces.Discrete(len(self.gamma_choices) * len(self.skew_choices))
        else:
            self.action_space = self.dyn.get_action_space()

    # -------- Gym API --------
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            super().reset(seed=seed)
            self.rng = np.random.default_rng(seed)
            
        self.dyn.mid.reset()
        self.dyn.arr.reset()
        self.price = np.copy(self.dyn.mid.state[:, 0])
        self.q[:] = 0
        self.cash[:] = 0.0
        self.t_idx = 0
        
        # Reset parameters to initial values
        self.current_gamma = self.current_gamma  # Keep the initial value
        self.current_skew = self.current_skew    # Keep the initial value
        if self.return_vectorized and self.enable_rl_control:
            self.current_gamma_vec[:] = self.current_gamma
            self.current_skew_vec[:] = self.current_skew
        
        self.reward_fn.reset()
        obs = self._pack_state()
        info = {}  # Always return an info dict
        return obs, info  # FIX: Return tuple (obs, info)

    def step(self, action: np.ndarray):
        # Handle RL control mode
        if self.enable_rl_control:
            if not self.return_vectorized:
                # Single trajectory: decode action and update parameters
                gamma, skew = self.decode_rl_action(int(action))
                self.set_rl_parameters(gamma, skew)
                half_spreads = self._compute_as_spreads()
            else:
                # Vectorized: handle multiple actions
                half_spreads_list = []
                for i in range(self.N):
                    gamma, skew = self.decode_rl_action(int(action[i]))
                    self.set_rl_parameters(gamma, skew, trajectory_idx=i)
                    trajectory_spreads = self._compute_as_spreads(i)
                    half_spreads_list.append(trajectory_spreads[0])
                half_spreads = np.array(half_spreads_list)
        else:
            # Original logic: action contains half-spreads directly
            action = np.asarray(action, float)
            if action.ndim == 1:
                action = np.repeat(action.reshape(1, -1), self.N, axis=0)
            elif action.shape[0] != self.N:
                raise ValueError(f"action batch size {action.shape[0]} != num_traj {self.N}")
            half_spreads = action[:, :2]

        # Snapshot current batch state BEFORE transition
        current_state = self._pack_state()

        # 1) midprice step
        mid_out = self.dyn.mid.step()
        self.price = mid_out["price"]

        # 2) arrivals + fills
        arrivals, fills = self.dyn.arrivals_and_fills(half_spreads, self.rng)

        # 3) market orders (if available)
        dq_mo = np.zeros(self.N, int)
        dcash_mo = np.zeros(self.N, float)
        if hasattr(self.dyn, 'market_order_deltas') and half_spreads.shape[1] >= 4:
            dq_mo, dcash_mo = self.dyn.market_order_deltas(half_spreads)

        # 4) limit-order deltas
        dq_lim, dcash_lim = self.dyn.cash_inventory_delta(half_spreads, arrivals, fills)

        # 5) apply updates within bounds
        dq = dq_mo + dq_lim
        dcash = dcash_mo + dcash_lim

        can_buy  = (self.q + dq) <=  self.max_inventory
        can_sell = (self.q + dq) >= -self.max_inventory
        ok = can_buy & can_sell

        self.q[ok]    += dq[ok]
        self.cash[ok] += dcash[ok]

        # 6) advance time index
        self.t_idx += 1
        terminated = self.t_idx >= self.M
        done_vec = np.full(self.N, terminated, dtype=bool)

        # Next state AFTER transition
        next_state = self._pack_state()

        # 7) info payload for rewards/analytics
        info_batch = {
            "arrivals": arrivals,
            "fills": fills,
            "half_spreads": half_spreads,
            "dq": dq, 
            "dcash": dcash,
            "current_gamma": self.current_gamma if not self.return_vectorized else self.current_gamma_vec,
            "current_skew": self.current_skew if not self.return_vectorized else self.current_skew_vec
        }

        # 8) reward via pluggable interface (vectorized)
        # Ensure states are properly shaped for the reward function
        if self.return_vectorized:
            rewards = self.reward_fn.calculate(current_state, half_spreads, next_state, done_vec, info_batch)
        else:
            # For single trajectory, reshape to (1, 6) for compatibility
            current_state_2d = current_state.reshape(1, -1)
            next_state_2d = next_state.reshape(1, -1)
            half_spreads_2d = half_spreads.reshape(1, -1)
            done_vec_2d = np.array([done_vec[0]])
            
            rewards = self.reward_fn.calculate(current_state_2d, half_spreads_2d, next_state_2d, done_vec_2d, info_batch)

        # 9) Output
        if self.return_vectorized:
            obs = next_state
            info = info_batch
            return obs, rewards.astype(np.float32), done_vec, False, info
        else:
            obs = next_state
            # For single trajectory, extract scalar reward
            reward_scalar = float(rewards[0]) if hasattr(rewards, '__len__') else float(rewards)
            info = {k: (v[0] if hasattr(v, 'shape') and v.shape[0] == self.N else v) for k, v in info_batch.items()}
            return obs.astype(np.float32), reward_scalar, bool(terminated), False, info

    def get_current_parameters(self):
        """Return current parameters"""
        if self.return_vectorized and self.enable_rl_control:
            return self.current_gamma_vec, self.current_skew_vec
        else:
            return self.current_gamma, self.current_skew