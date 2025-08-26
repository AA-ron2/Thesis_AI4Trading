# =========================
# mbt_gym/gym/index_names.py
# =========================
ASSET_PRICE_INDEX = 0
INVENTORY_INDEX   = 1
TIME_INDEX        = 2
CASH_INDEX        = 3

# (Optionally expose BID/ASK indices via info dict rather than the state)


# ===========================================
# mbt_gym/stochastic_processes/base_models.py
# ===========================================
import numpy as np

class StochasticProcessModel:
    def __init__(self, step_size: float, terminal_time: float, seed: int = None):
        self.step_size = float(step_size)
        self.terminal_time = float(terminal_time)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass


# ==============================================
# mbt_gym/stochastic_processes/midprice_models.py
# ==============================================
import numpy as np

class BrownianMotionMidpriceModel(StochasticProcessModel):
    def __init__(self, initial_price: float, volatility: float, step_size: float, terminal_time: float, seed: int = None):
        super().__init__(step_size, terminal_time, seed)
        self.initial_price = float(initial_price)
        self.volatility = float(volatility)
        self.current_price = float(initial_price)

    def reset(self):
        self.current_price = float(self.initial_price)

    def step(self) -> float:
        z = self.rng.standard_normal()
        self.current_price += self.volatility * np.sqrt(self.step_size) * z
        return self.current_price


# =============================================
# mbt_gym/stochastic_processes/arrival_models.py
# =============================================
import numpy as np

class PoissonArrivalModel(StochasticProcessModel):
    """
    Baseline exogenous arrivals with constant intensity A (per side).
    In mbt-gym style, this produces binary arrival flags per side each step.
    """
    def __init__(self, intensity: float, step_size: float, terminal_time: float, seed: int = None):
        super().__init__(step_size, terminal_time, seed)
        self.intensity = float(intensity)  # A

    def reset(self):
        pass

    def get_arrivals(self) -> np.ndarray:
        # Bernoulli per side with p = A * dt (clip for safety)
        p = np.clip(self.intensity * self.step_size, 0.0, 1.0)
        unif = self.rng.uniform(size=(2,))
        return (unif < p).astype(bool)  # [sell-arrival-on-bid, buy-arrival-on-ask]


# ===========================================================
# mbt_gym/stochastic_processes/fill_probability_models.py
# ===========================================================
import numpy as np

class ExponentialFillFunction(StochasticProcessModel):
    """
    P(fill | arrival, half-spread delta) = exp(-k * delta).
    """
    def __init__(self, fill_exponent: float, step_size: float, terminal_time: float, seed: int = None):
        super().__init__(step_size, terminal_time, seed)
        self.fill_exponent = float(fill_exponent)  # k

    def reset(self):
        pass

    def prob_fill(self, delta: float) -> float:
        return float(np.exp(-self.fill_exponent * max(0.0, float(delta))))


# ===================================
# mbt_gym/gym/ModelDynamics.py
# ===================================
from dataclasses import dataclass

@dataclass
class LimitOrderModelDynamics:
    midprice_model: BrownianMotionMidpriceModel
    arrival_model: PoissonArrivalModel
    fill_probability_model: ExponentialFillFunction

    def reset(self):
        self.midprice_model.reset()
        self.arrival_model.reset()
        self.fill_probability_model.reset()

    @property
    def dt(self) -> float:
        return self.midprice_model.step_size


# ===================================
# mbt_gym/rewards/RewardFunctions.py
# ===================================
def pnl_increment(prev_price: float, prev_q: int, prev_cash: float,
                  new_price: float, new_q: int, new_cash: float) -> float:
    prev_val = prev_q * prev_price + prev_cash
    new_val  = new_q * new_price  + new_cash
    return float(new_val - prev_val)


# =======================================
# mbt_gym/agents/BaselineAgents.py (AS)
# =======================================
import numpy as np
import warnings
from math import log

class AvellanedaStoikovAgent:
    """
    Deterministic AS quoting agent producing [bid_half_spread, ask_half_spread].
    Matches your finite-horizon (limit_horizon=True) and infinite-horizon variants.
    """
    def __init__(
        self,
        env,                       # TradingEnvironment (for T, sigma, etc.)
        risk_aversion: float = 0.1,
        limit_horizon: bool = True,
        q_max: int = 100
    ):
        self.env = env
        self.risk_aversion = float(risk_aversion)
        self.limit_horizon = bool(limit_horizon)
        self.q_max = int(q_max)

        # Pull needed params from env/model
        self.T = env.terminal_time
        self.sigma = env.model_dynamics.midprice_model.volatility
        self.k = env.model_dynamics.fill_probability_model.fill_exponent

    def _price_adjustment(self, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        # inventory * γ * σ^2 * (T - t)
        return q * self.risk_aversion * (self.sigma ** 2) * (self.T - t)

    def _spread_finite(self, t: np.ndarray) -> np.ndarray:
        γ = self.risk_aversion
        if γ == 0:
            return np.full_like(t, 2.0 / self.k)
        vol_term = γ * (self.sigma ** 2) * (self.T - t)
        fill_term = (2.0 / γ) * np.log(1.0 + γ / self.k)
        return vol_term + fill_term

    def _spread_infinite(self, q: np.ndarray) -> np.ndarray:
        # Uses your omega definition; the actual quoting happens per side in env’s state, but for coherence:
        γ = self.risk_aversion
        σ2 = self.sigma ** 2
        ω  = 0.5 * (γ ** 2) * σ2 * (self.q_max + 1) ** 2
        coef = (γ ** 2) * σ2 / (2 * ω - (γ ** 2) * (q ** 2) * σ2)
        # We return the half-spreads indirectly via env formula; here we just encode a symmetric proxy:
        # spread ~ log(1+(1-2q)*coef) - log(1+(-1-2q)*coef)
        left  = np.log(1.0 + (1.0 - 2.0 * q) * coef)
        right = np.log(1.0 + (-1.0 - 2.0 * q) * coef)
        return np.maximum(0.0, (left - right))  # safe guard

    def get_action(self, state: np.ndarray) -> np.ndarray:
        # state shape: (num_traj, state_dim). We need inventory and time.
        q   = state[:, INVENTORY_INDEX].astype(float)
        t   = state[:, TIME_INDEX].astype(float)
        adj = self._price_adjustment(q, t)

        if self.limit_horizon:
            spr = self._spread_finite(t)
            bid_half = (adj + 0.5 * spr).reshape(-1, 1)
            ask_half = (-adj + 0.5 * spr).reshape(-1, 1)
        else:
            # Infinite-horizon variant consistent with your env step
            # Use symmetric half-spread from proxy, still skew with adj
            spr = self._spread_infinite(q)
            bid_half = (adj + 0.5 * spr).reshape(-1, 1)
            ask_half = (-adj + 0.5 * spr).reshape(-1, 1)

        action = np.concatenate([bid_half, ask_half], axis=1)
        if np.any(action < 0):
            warnings.warn("AS agent produced a negative half-spread.")
        return action


# =========================================
# mbt_gym/gym/TradingEnvironment.py
# =========================================
import gym
from gym import spaces
import numpy as np
import pandas as pd

from mbt_gym.gym.index_names import (
    ASSET_PRICE_INDEX, INVENTORY_INDEX, TIME_INDEX, CASH_INDEX
)
from mbt_gym.rewards.RewardFunctions import pnl_increment

class TradingEnvironment(gym.Env):
    """
    mbt-gym style wrapper:
    - State = [S, q, t, cash]
    - Action = [bid_half_spread, ask_half_spread] (Box)
    - Dynamics = (midprice step) + (Poisson arrivals) + (exp fill probability)
    - Reward = step PnL increment
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        model_dynamics,           # LimitOrderModelDynamics
        terminal_time: float = 1.0,
        n_steps: int = 200,
        seed: int = None,
        initial_inventory: int = 0,
        initial_price: float = 100.0,
        initial_cash: float = 0.0,
        max_inventory: int = 100,
        num_trajectories: int = 1,              # kept for API symmetry; here we run 1 traj
    ):
        super().__init__()
        assert num_trajectories == 1, "This minimal example runs a single trajectory."

        self.model_dynamics = model_dynamics
        self.terminal_time  = float(terminal_time)
        self.n_steps        = int(n_steps)
        self.dt             = self.terminal_time / self.n_steps
        self.initial_price  = float(initial_price)
        self.initial_inventory = int(initial_inventory)
        self.initial_cash   = float(initial_cash)
        self.max_inventory  = int(max_inventory)

        # RNG
        self.rng = np.random.default_rng(seed)

        # Action: half-spreads (nonnegative)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Observation: [S, q, t, cash]
        low  = np.array([0.0, -np.inf, 0.0, -np.inf], dtype=np.float32)
        high = np.array([np.inf,  np.inf, float(self.n_steps),  np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Render buffer
        self.render_df = pd.DataFrame({
            "Bid": pd.Series(dtype="float64"),
            "Price": pd.Series(dtype="float64"),
            "Ask": pd.Series(dtype="float64"),
            "Spread": pd.Series(dtype="float64"),
            "Cash": pd.Series(dtype="float64"),
            "Inventory": pd.Series(dtype="int64"),
            "PnL": pd.Series(dtype="float64")
        })

        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.model_dynamics.reset()

        self.t_idx = 0
        self.S = self.initial_price
        self.q = int(self.initial_inventory)
        self.cash = float(self.initial_cash)

        self.last_portfolio_value = self.q * self.S + self.cash
        self.cum_pnl = [self.last_portfolio_value]

        obs = self._obs()
        return obs, {}

    def _obs(self):
        # shape (1, state_dim) to match mbt-gym agents expecting batch
        return np.array([[self.S, self.q, self.t_idx, self.cash]], dtype=np.float32)

    def step(self, action):
        # 1) midprice update
        prev_S, prev_q, prev_cash = self.S, self.q, self.cash
        self.S = self.model_dynamics.midprice_model.step()

        # 2) interpret action as half-spreads
        bid_half, ask_half = float(action[0]), float(action[1])
        bid_px = self.S - bid_half
        ask_px = self.S + ask_half

        # 3) arrivals + fills (mbt-gym style: arrivals first, then conditional fills)
        arrivals = self.model_dynamics.arrival_model.get_arrivals()  # [sell-on-bid, buy-on-ask]
        # fill probs conditional on arrival:
        p_bid_fill = self.model_dynamics.fill_probability_model.prob_fill(bid_half)
        p_ask_fill = self.model_dynamics.fill_probability_model.prob_fill(ask_half)

        u = self.rng.uniform(size=2)
        bid_filled = arrivals[0] and (u[0] < p_bid_fill)
        ask_filled = arrivals[1] and (u[1] < p_ask_fill)

        # 4) inventory/cash updates (allow both to happen same step)
        if bid_filled and self.q + 1 <= self.max_inventory:
            self.q += 1
            self.cash -= bid_px
        if ask_filled and self.q - 1 >= -self.max_inventory:
            self.q -= 1
            self.cash += ask_px

        # 5) reward as PnL increment
        reward = pnl_increment(prev_S, prev_q, prev_cash, self.S, self.q, self.cash)

        # 6) bookkeeping
        self.t_idx += 1
        terminated = self.t_idx >= self.n_steps
        truncated = False

        spread = bid_half + ask_half
        self.cum_pnl.append(self.q * self.S + self.cash)
        self.render_df.loc[len(self.render_df)] = {
            "Bid": bid_px, "Price": self.S, "Ask": ask_px, "Spread": spread,
            "Cash": self.cash, "Inventory": self.q, "PnL": self.cum_pnl[-1]
        }

        return self._obs(), float(reward), bool(terminated), bool(truncated), {
            "bid_px": bid_px, "ask_px": ask_px,
            "bid_filled": bool(bid_filled), "ask_filled": bool(ask_filled)
        }
