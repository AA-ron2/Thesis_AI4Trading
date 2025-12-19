# import numpy as np

# def exp_fill_prob(half_spreads: np.ndarray, k: float) -> np.ndarray:
#     """
#     half_spreads: (N,2) -> probabilities (N,2) with P(fill | arrival) = exp(-k * half_spread)
#     """
#     hs = np.maximum(half_spreads, 0.0)
#     return np.exp(-k * hs)


import abc
from typing import Optional, Tuple

import numpy as np

from .base import ProcessBase


class FillProbabilityModel(ProcessBase):
    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(
            init_state=initial_state,
            num_traj=num_trajectories,
            dt=step_size,
            T=terminal_time,
            seed=seed,
            min_val=min_value,
            max_val=max_value,
        )

    @abc.abstractmethod
    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        pass

    def get_fills(self, depths: np.ndarray) -> np.ndarray:
        assert depths.shape == (self.num_traj, 2), (
            f"Depths must have shape ({self.num_traj}, 2). Got {depths.shape}."
        )
        unif = self.rng.uniform(size=(self.num_traj, 2))
        return unif < self._get_fill_probabilities(depths)

    @property
    @abc.abstractmethod
    def max_depth(self) -> float:
        pass



class ExponentialFillFunction(FillProbabilityModel):
    def __init__(
        self, fill_exponent: float = 1.5, step_size: float = 0.1, num_trajectories: int = 1, seed: Optional[int] = None
    ):
        self.fill_exponent = fill_exponent
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        return np.exp(-self.fill_exponent * depths)

    @property
    def max_depth(self) -> float:
        return -np.log(0.01) / self.fill_exponent

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass