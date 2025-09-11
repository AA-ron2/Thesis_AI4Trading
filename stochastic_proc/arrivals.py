import numpy as np
from .base import ProcessBase

class PoissonArrivals(ProcessBase):
    """
    Constant per-side intensity: state = [lambda_bid, lambda_ask] (kept for uniformity)
    """
    def __init__(self, lam_bid: float, lam_ask: float, num_traj: int, dt: float, T: float, seed=None):
        super().__init__(init_state=np.array([lam_bid, lam_ask]), num_traj=num_traj, dt=dt, T=T, seed=seed)

    def step(self, **kwargs):
        p = np.clip(self.state * self.dt, 0.0, 1.0)               # (N,2)
        u = self.rng.uniform(size=self.state.shape)
        arrivals = (u < p)                                        # bool (N,2): [sell-on-bid, buy-on-ask]
        self.t_idx += 1
        return {"arrivals": arrivals}

class HawkesArrivals(ProcessBase):
    """
    Exponential-kernel Hawkes: state = [lambda_bid, lambda_ask]
    update: λ <- λ + κ(λ0 - λ)dt + jump * arrivals
    """
    def __init__(self, lam0_bid: float, lam0_ask: float, kappa: float, jump: float,
                 num_traj: int, dt: float, T: float, seed=None):
        super().__init__(init_state=np.array([lam0_bid, lam0_ask]), num_traj=num_traj, dt=dt, T=T, seed=seed)
        self.lam0 = np.repeat(self._init, self.num_traj, axis=0)  # baseline per traj
        self.kappa = float(kappa)
        self.jump = float(jump)

    def step(self, **kwargs):
        # sample arrivals from current intensity
        p = np.clip(self.state * self.dt, 0.0, 1.0)
        u = self.rng.uniform(size=self.state.shape)
        arrivals = (u < p)                                       # (N,2)

        # update intensities
        self.state += self.kappa * (self.lam0 - self.state) * self.dt + self.jump * arrivals
        self.t_idx += 1
        return {"arrivals": arrivals, "intensity": self.state}
