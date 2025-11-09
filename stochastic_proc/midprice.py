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
    
    # stochastic_proc/midprice.py (updated HistoricalMidprice)
class HistoricalMidprice(ProcessBase):
    """
    Midprice process driven by historical data feed with volatility for AS model
    """
    def __init__(self, feed, num_traj: int, dt: float, T: float, sigma: float = 0.02):
        self.feed = feed
        # Initialize with first data point
        if hasattr(feed, 'snapshot'):
            init_snapshot = feed.snapshot()
        else:
            init_snapshot = feed._get_batch_snapshot()
            
        init_mid = init_snapshot['mid']
        
        if np.isscalar(init_mid):
            init_state = np.array([[init_mid]] * num_traj)
        else:
            init_state = init_mid.reshape(-1, 1)
            
        super().__init__(init_state=init_state, num_traj=num_traj, dt=dt, T=T, seed=None)
        
        # Store volatility for AS model
        self.sigma = float(sigma)
        
    def reset(self):
      super().reset()
      self.current_idx = 0
      self.state[:, 0] = self.mid_prices[0]
    
    def step(self, **kwargs):
        """Advance to next historical data point"""
        if hasattr(self.feed, 'step'):
            snapshot = self.feed.step()
        else:
            snapshot = self.feed._get_batch_snapshot()
            
        mid = snapshot['mid']
        
        if np.isscalar(mid):
            self.state[:, 0] = mid
        else:
            self.state[:, 0] = mid
            
        self.t_idx += 1
        return {"price": self.state[:, 0]}