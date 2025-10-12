import numpy as np
from utils.calibration import glft_half_spreads

class GLFTAgent:
    """
    Deterministic GLFT agent producing [bid_half, ask_half] per observation row.

    Parameters
    ----------
    gamma : risk aversion (same role as in AS)
    A, k  : intensity parameters in λ(δ) = A * exp(-k δ)
    xi    : “trade size / sensitivity” factor used in GLFT (often 1.0 if you
            do not model discrete queue/size effects explicitly)
    tick  : price tick size of your instrument (in price units)
    """
    def __init__(self, env, gamma: float, A: float, k: float, xi: float = 1.0, tick: float = 1.0):
        self.env   = env
        self.gamma = float(gamma)
        self.A     = float(A)
        self.k     = float(k)
        self.xi    = float(xi)
        self.tick  = float(tick)
        # pull sigma from your midprice model
        self.sigma = float(env.dyn.mid.sigma)
        # optional: for your plotting toggle
        self.mode  = "glft"

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # obs: (4,) or (N,4) as [price, inventory, time_idx, cash]
        if obs.ndim == 1:
            q = np.array([obs[1]], float)
            halves = glft_half_spreads(q, self.sigma, self.gamma, self.A, self.k,
                                       xi=self.xi, tick=self.tick)
            return halves[0].astype(np.float32)
        else:
            q = obs[:, 1].astype(float)
            halves = glft_half_spreads(q, self.sigma, self.gamma, self.A, self.k,
                                       xi=self.xi, tick=self.tick)
            return halves.astype(np.float32)