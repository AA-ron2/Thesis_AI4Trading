import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from typing import List, Any, Optional, Union, Sequence, Type
import gymnasium as gym

class StableBaselinesTradingEnv(VecEnv):
    """
    Wrap an internally vectorized env (return_vectorized=True) for SB3.
    """
    def __init__(self, trading_env):
        assert trading_env.return_vectorized, "Env must return vectorized outputs for this adapter."
        self.env = trading_env
        n_envs = self.env.N
        super().__init__(n_envs, self.env.observation_space, self.env.action_space)

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs, rewards, dones, truncated, infos = self.env.step(self._actions)
        if dones.min():
            obs, _ = self.env.reset()
        # SB3 expects (obs, rewards, dones, infos)
        return obs, rewards, dones, infos

    def close(self) -> None: ...
    def get_attr(self, attr_name: str, indices=None) -> List[Any]: ...
    def set_attr(self, attr_name: str, value: Any, indices=None) -> None: ...
    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]: ...
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices=None) -> List[bool]:
        return [False] * self.env.N
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self.env.reset(seed=seed); return [seed]*self.env.N
    def get_images(self) -> Sequence[np.ndarray]: return []
