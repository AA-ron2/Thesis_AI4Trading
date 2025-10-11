# utils/sb3_agent.py
import numpy as np

class SB3AgentAdapter:
    """
    Wraps a trained SB3 model so it looks like your heuristic agent:
      get_action(obs) -> np.ndarray shaped (N, act_dim) or (act_dim,)
    """
    def __init__(self, model):
        self.model = model

    def get_action(self, obs: np.ndarray):
        # SB3's predict handles batched (N, obs_dim) or single (obs_dim,)
        action, _ = self.model.predict(obs, deterministic=True)
        return action.astype(np.float32)
