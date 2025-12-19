import numpy as np
import math
# from envs.tradingenv import TradingEnv
import abc
import warnings
from utils.indx import INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX, CASH_INDEX, BID_INDEX, ASK_INDEX

class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass

    def get_expected_action(self, state: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        return np.array([self.get_action(state) for _ in range(n_samples)]).mean(axis=0)

def as_half_spreads(inventory: np.ndarray, t_idx: np.ndarray, T: float, sigma: float, gamma: float, k: float, dt: float):
    # reservation-price skew term
    t = t_idx * dt
    adj = inventory * gamma * (sigma**2) * (T - t)
    if gamma == 0.0:
        spread = np.full_like(t, 2.0 / k, dtype=float)
    else:
        spread = gamma * (sigma**2) * (T - t) + (2.0 / gamma) * np.log(1.0 + gamma / k)
    bid_half = adj + 0.5 * spread
    ask_half = -adj + 0.5 * spread
    return np.stack([bid_half, ask_half], axis=1)


def as_infinite_half_spreads(s: np.ndarray, q: np.ndarray,
                             gamma: float, sigma: float, omega: float,
                             eps: float = 1e-12) -> np.ndarray:
    """
    Robust infinite-horizon AS:
      r^a = s + (1/gamma) ln( 1 + ((1-2q) gamma^2 σ^2) / (2omega - gamma^2 q^2 σ^2) )
      r^b = s + (1/gamma) ln( 1 + ((-1-2q) gamma^2 σ^2) / (2omega - gamma^2 q^2 σ^2) )
    Returns half-spreads [bid_half, ask_half] >= 0 with NaN-safe clipping.
    """
    s = np.asarray(s, float).reshape(-1)
    q = np.asarray(q, float).reshape(-1)

    gamma = float(gamma)
    sigma_sq = float(sigma) ** 2
    omega  = float(omega)

    # denom must be > 0; clip up to a small epsilon to avoid div-by-zero
    denom = 2.0 * omega - ( (gamma**2) * sigma_sq * (q**2) )
    denom = np.maximum(denom, eps)

    # coef = ( (gamma**2) * sigma_sq ) / denom

    z_ask = 1.0 + ( ( ( 1.0 - 2.0 * q ) * (gamma**2) * sigma_sq) / denom )
    z_bid = 1.0 + ( ( ( -1.0 - 2.0 * q ) * (gamma**2) * sigma_sq) / denom )
    
    z_ask = np.maximum(z_ask, eps)
    z_bid = np.maximum(z_bid, eps)

    ask_px = s + (1.0 / gamma) * np.log(z_ask)
    bid_px = s + (1.0 / gamma) * np.log(z_bid)

    bid_half = np.maximum(0.0, s - bid_px)
    ask_half = np.maximum(0.0, ask_px - s)

    # ensure finite values (replace any inf/nan by 0 to preserve plotting)
    bid_half = np.where(np.isfinite(bid_half), bid_half, 0.0)
    ask_half = np.where(np.isfinite(ask_half), ask_half, 0.0)

    return np.stack([bid_half, ask_half], axis=1)

class AvellanedaStoikovAgent(Agent):
    """
    Deterministic AS agent.
    mode='finite'  -> classic finite-horizon with fill parameter k
    mode='infinite'-> infinite-horizon with inventory penalty omega
    """
    def __init__(self, env, gamma: float = 0.1, mode: str = "finite",
                 k_fill: float | None = None,
                 q_max: int | None = 0,
                 omega: float | None = None):
        self.gamma = gamma
        self.env = env
        self.mode = mode.lower()
        self.sigma = env.dyn.mid.sigma
        self.T = env.T
        self.dt = env.dt

        # finite-horizon params
        self.k = float(k_fill) if k_fill is not None else getattr(env.dyn, "fill_k", None)

        # infinite-horizon params
        if omega is None:
            # omega = 0.5 * gamma^2 σ^2 (q_max+1)^2  (natural choice)
            self.omega = 0.5 * (self.gamma**2) * (self.sigma**2) * ((q_max or 100) + 1)**2
        else:
            self.omega = float(omega)
            
    def get_action(self, state: np.ndarray):
        inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        action = self._get_action(inventory, time)
        if action.min() < 0:
            warnings.warn("Avellaneda-Stoikov agent is quoting a negative spread")
        return action
    
    def get_action(self, state: np.ndarray):
        inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
        action = self._get_action(inventory, time)
        if action.min() < 0:
            warnings.warn("Avellaneda-Stoikov agent is quoting a negative spread")
        return action
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # obs can be (4,) or (N,4) with [price, inventory, time_idx, cash]
        if obs.ndim == 1:
            s  = np.array([obs[0]], float)
            q  = np.array([obs[1]], float)
            tI = np.array([obs[2]], float)  # time index
        else:
            s  = obs[:, 0].astype(float)
            q  = obs[:, 1].astype(float)
            tI = obs[:, 2].astype(float)

        if self.mode == "infinite":
            halves = as_infinite_half_spreads(s, q, self.gamma, self.sigma, self.omega)
        else:
            # finite-horizon via your helper
            halves = as_half_spreads(
                inventory=q,             # (N,)
                t_idx=tI,                # (N,) time index (integer steps)
                T=self.T,
                sigma=self.sigma,
                gamma=self.gamma,
                k=float(self.k),
                dt=self.dt,
            )
            # Ensure valid (non-negative) half-spreads
            halves = np.maximum(halves, 0.0)

        # return shape (2,) or (N,2) to match env
        return halves.astype(np.float32) if obs.ndim > 1 else halves[0].astype(np.float32)