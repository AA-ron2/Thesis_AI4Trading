# mmgym_lite/rewards/common.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Iterable, Sequence
from .Base import RewardFunction

class PnLReward(RewardFunction):
    """
    Step PnL increment: (q_new * S_new + cash_new) - (q_old * S_old + cash_old)
    """
    def calculate(self, current_state, action, next_state, done, info):
        S0, q0, _, c0 = current_state[:, 0], current_state[:, 1], current_state[:, 2], current_state[:, 3]
        S1, q1, _, c1 = next_state[:, 0], next_state[:, 1], next_state[:, 2], next_state[:, 3]
        old_val = q0 * S0 + c0
        new_val = q1 * S1 + c1
        return (new_val - old_val).astype(np.float32)

class InventoryQuadraticPenalty(RewardFunction):
    """
    Penalize inventory squared each step:  -λ * q^2 * w
    If you want risk-averse AS-style shaping, pick λ ~ 0.5 * gamma * sigma^2 and w = dt.
    """
    def __init__(self, lam: float, weight: float = 1.0, use_next_q: bool = True):
        self.lam = float(lam)
        self.weight = float(weight)
        self.use_next_q = bool(use_next_q)

    def calculate(self, current_state, action, next_state, done, info):
        q = next_state[:, 1] if self.use_next_q else current_state[:, 1]
        return (-self.lam * (q ** 2) * self.weight).astype(np.float32)

class SpreadRegularizer(RewardFunction):
    """
    Small L1 penalty on quoted half-spreads to discourage extreme widening:
      -α * (bid_half + ask_half)
    Works for both limit-only (2 dims) and limit+market (first two dims are half-spreads).
    """
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def calculate(self, current_state, action, next_state, done, info):
        halves = action[:, :2]
        return (-self.alpha * (halves[:, 0] + halves[:, 1])).astype(np.float32)

class SumReward(RewardFunction):
    """
    Weighted sum of multiple reward terms.
    """
    def __init__(self, terms: Sequence[RewardFunction], weights: Iterable[float] | None = None):
        self.terms = list(terms)
        self.weights = np.ones(len(self.terms), dtype=float) if weights is None else np.asarray(list(weights), float)

    def reset(self) -> None:
        for t in self.terms:
            t.reset()

    def calculate(self, current_state, action, next_state, done, info):
        total = np.zeros(current_state.shape[0], dtype=np.float32)
        for w, term in zip(self.weights, self.terms):
            total += w * term.calculate(current_state, action, next_state, done, info)
        return total