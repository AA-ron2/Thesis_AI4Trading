import numpy as np
from .base import ProcessBase

class BrownianMidprice(ProcessBase):
    """
    Single-dimension price process: state[:, 0] = price
    """
    def __init__(self, s0: float, sigma: float, num_traj: int, dt: float, T: float, seed=None):
        super().__init__(init_state=np.array([s0]), num_traj=num_traj, dt=dt, T=T, seed=seed)
        self.sigma = float(sigma)

    @property
    def price(self) -> np.ndarray:
        return self.state[:, 0]

    def step(self, **kwargs):
        z = self.rng.standard_normal(self.num_traj)
        self.state[:, 0] += self.sigma * np.sqrt(self.dt) * z
        self.t_idx += 1
        return {"price": self.state[:, 0]}
    
class RecordedMidprice(ProcessBase):
    """
    Uses L2Feed to supply midprice as a 1-D state per trajectory.
    For BatchFeed, pass a “feed” whose .step() returns per-trajectory mids.
    """
    def __init__(self, feed, num_traj: int, dt: float, T: float):
        self.feed = feed
        # initialize from first snapshot
        s0 = feed.reset(0)
        # if feed is batched, s0["mid"] should be array-like of length num_traj
        init = np.atleast_1d(s0["mid"]).astype(float).reshape(1, -1) if np.ndim(s0["mid"]) > 0 else np.array([s0["mid"]])
        super().__init__(init_state=np.array([np.mean(init)]), num_traj=num_traj, dt=dt, T=T, seed=None)

    def step(self, **kwargs):
        s = self.feed.step()
        mid = np.atleast_1d(s["mid"]).astype(float)
        if mid.shape[0] == 1 and self.num_traj > 1:
            mid = np.repeat(mid, self.num_traj)
        self.state[:, 0] = mid
        self.t_idx += 1
        return {"price": self.state[:, 0]}

class HistoricalMidprice(ProcessBase):
    def __init__(self, mid_prices: np.ndarray, num_traj: int, dt: float, T: float, sigma: float):
        # mid_prices: array of historical mid prices
        self.mid_prices = mid_prices
        self.current_idx = 0
        super().__init__(init_state=np.array([mid_prices[0]]), num_traj=num_traj, dt=dt, T=T)
        
    def reset(self):
      super().reset()
      self.current_idx = 0
      self.state[:, 0] = self.mid_prices[0]

    def step(self, **kwargs):
        self.current_idx += 1
        if self.current_idx < len(self.mid_prices):
            self.state[:, 0] = self.mid_prices[self.current_idx]
        else:
            # repeat the last price if we run out
            self.state[:, 0] = self.mid_prices[-1]
        self.t_idx += 1
        return {"price": self.state[:, 0]}