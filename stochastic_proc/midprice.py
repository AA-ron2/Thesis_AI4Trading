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