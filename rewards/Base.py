# mmgym_lite/rewards/base.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any

class RewardFunction:
    """
    Vectorized reward interface.
    calculate(...) must return a (N,) float array, where N = num trajectories.
    """
    def reset(self) -> None:
        pass

    def calculate(
        self,
        current_state: np.ndarray,  # shape (N, 4): [price, inv, t_idx, cash]
        action: np.ndarray,         # shape (N, A)
        next_state: np.ndarray,     # shape (N, 4)
        done: np.ndarray,           # shape (N,), bool
        info: Dict[str, Any],       # may include arrivals, fills, etc.
    ) -> np.ndarray:
        raise NotImplementedError