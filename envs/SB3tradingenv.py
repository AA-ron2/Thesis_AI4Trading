# utils/sb3_adapter.py
from typing import List, Any, Optional, Sequence, Type, Union
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

class SB3TradingVecEnv(VecEnv):
    """
    Adapter: expose your TradingEnv (with return_vectorized=True, num_traj=N)
    as a VecEnv with num_envs = N.
    """
    def __init__(self, trading_env: gym.Env, store_terminal_observation_info: bool = True):
        assert getattr(trading_env, "return_vectorized", False), \
            "TradingEnv must be created with return_vectorized=True."
        self.env = trading_env
        self.store_terminal_observation_info = store_terminal_observation_info
        self._actions = np.zeros_like(self.env.action_space.sample())  # shape (N, act_dim)
        num_envs = self.env.N  # == num_traj
        super().__init__(num_envs, self.env.observation_space, self.env.action_space)

    def reset(self) -> VecEnvObs:
        obs, _ = self.env.reset()
        return obs  # shape (N, obs_dim)

    def step_async(self, actions: np.ndarray) -> None:
        # SB3 gives shape (N, act_dim); your env expects the same
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Gymnasium step: obs, reward, terminated, truncated, info
        obs, rewards, terminated, truncated, infos = self.env.step(self._actions)

        # On episode end, SB3 convention is: provide terminal obs in infos and auto-reset
        # Your env ends all N together; terminated is shape (N,) of bool
        if np.all(terminated) or np.all(truncated):
            if self.store_terminal_observation_info:
                infos = list(infos) if isinstance(infos, tuple) else infos
                for i in range(len(infos)):
                    if isinstance(infos[i], dict):
                        infos[i] = dict(infos[i])
                        infos[i]["terminal_observation"] = obs[i]
            obs, _ = self.env.reset()

        # SB3 expects rewards shape (N,), dones shape (N,), infos list of dict
        dones = np.logical_or(terminated, truncated)
        return obs, rewards.reshape(-1), dones, infos

    # Stubs to satisfy interface (not used)
    def close(self) -> None: pass
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]: return []
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None: pass
    def env_method(self, method_name: str, *args, indices: VecEnvIndices = None, **kwargs) -> List[Any]: return []
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False] * self.env.N
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self.env.reset(seed=seed); return [seed]*self.env.N
    def get_images(self) -> Sequence[np.ndarray]: return []
