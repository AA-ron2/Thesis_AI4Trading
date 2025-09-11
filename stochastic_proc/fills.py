import numpy as np

def exp_fill_prob(half_spreads: np.ndarray, k: float) -> np.ndarray:
    """
    half_spreads: (N,2) -> probabilities (N,2) with P(fill | arrival) = exp(-k * half_spread)
    """
    hs = np.maximum(half_spreads, 0.0)
    return np.exp(-k * hs)
