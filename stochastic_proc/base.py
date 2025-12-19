from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from numpy.random import default_rng, Generator

class ProcessBase:
    """
    Minimal vectorized process scaffold.
    - state: (N, D) float array for N= num_trajectories
    - RNG, reset, seed provided
    - Subclasses implement .step(...)
    """
    def __init__(
        self,
        init_state: np.ndarray,      # shape (D,) or (1, D)
        num_traj: int,
        dt: float,
        T: float,
        seed: Optional[int] = None,
        min_val: Optional[np.ndarray] = None,
        max_val: Optional[np.ndarray] = None,
    ):
        init = np.asarray(init_state, float).reshape(1, -1)
        self.num_traj = num_traj
        self.dt = float(dt)
        self.T = float(T)
        self._init = init
        self.state = np.repeat(init, self.num_traj, axis=0)
        self.rng: Generator = default_rng(seed)
        self.min_val = None if min_val is None else np.asarray(min_val, float).reshape(1, -1)
        self.max_val = None if max_val is None else np.asarray(max_val, float).reshape(1, -1)
        self.t_idx = 0

    def reset(self) -> None:
        self.state = np.repeat(self._init, self.num_traj, axis=0)
        self.t_idx = 0

    def seed(self, seed: Optional[int]) -> None:
        self.rng = default_rng(seed)

    def clip_(self) -> None:
        if self.min_val is not None and self.max_val is not None:
            np.clip(self.state, self.min_val, self.max_val, out=self.state)

    def step(
        self,
        *,
        arrivals: Optional[np.ndarray] = None,
        fills: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        shared: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError
