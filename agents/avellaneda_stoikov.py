import numpy as np

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

class AvellanedaStoikovAgent:
    """
    Deterministic AS agent that emits [bid_half, ask_half] for each row in the batch.
    """
    def __init__(self, env, gamma=0.1):
        self.env = env
        self.gamma = float(gamma)
        self.sigma = env.dyn.mid.sigma
        self.k = env.dyn.fill_k
        self.T = env.T
        self.dt = env.dt

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # obs may be single (4,) or batch (N,4) depending on env.return_vectorized
        if obs.ndim == 1:
            inv = np.array([obs[1]], float)
            tidx = np.array([obs[2]], float)
            out = as_half_spreads(inv, tidx, self.T, self.sigma, self.gamma, self.k, self.dt)[0]
            return np.maximum(out, 0.0).astype(np.float32)
        else:
            inv = obs[:, 1].astype(float)
            tidx = obs[:, 2].astype(float)
            out = as_half_spreads(inv, tidx, self.T, self.sigma, self.gamma, self.k, self.dt)
            return np.maximum(out, 0.0).astype(np.float32)
