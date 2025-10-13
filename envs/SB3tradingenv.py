# utils/sb3_adapter.py
from typing import List, Any, Optional, Sequence, Type, Union
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

class SB3TradingVecEnv(VecEnv):
    """
    Adapter for a *batched* TradingEnv (return_vectorized=True).
    We ALWAYS infer num_envs (N) from a real reset() to avoid 0-length issues.
    """
    def __init__(self, trading_env: gym.Env, store_terminal_observation_info: bool = True):
        assert getattr(trading_env, "return_vectorized", False), (
            "TradingEnv must be created with return_vectorized=True."
        )
        self.env = trading_env
        self.store_terminal_observation_info = store_terminal_observation_info

        # ---- FORCE-INFER N FROM A REAL RESET ----
        try:
            obs, _ = self.env.reset()
        except Exception:
            # legacy reset returning only obs
            obs = self.env.reset()
        obs = np.asarray(obs)

        if obs.ndim == 2:
            N = obs.shape[0]
        elif obs.ndim == 1:
            N = 1
            obs = obs.reshape(1, -1)  # normalize
        else:
            raise RuntimeError(f"Unexpected obs ndim from reset(): {obs.shape}")

        if N <= 0:
            raise RuntimeError(f"Inferred num_envs N={N} from reset(); cannot build VecEnv.")

        # ---- INIT BASE VecEnv ----
        super().__init__(int(N), self.env.observation_space, self.env.action_space)

        # Placeholder actions (N, act_dim)
        sample = self.env.action_space.sample()
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim == 1:
            self._actions = np.repeat(sample.reshape(1, -1), N, axis=0)
        else:
            # assume already (N, act_dim)
            self._actions = np.zeros_like(sample, dtype=np.float32)

        # Cache last obs (VecEnv may call render etc. before first step)
        self._last_obs = obs

    def reset(self) -> VecEnvObs:
        try:
            obs, _ = self.env.reset()
        except Exception:
            obs = self.env.reset()
        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        self._last_obs = obs
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        # Env must return: (obs, rewards, terminated, truncated, infos) in batched shape
        obs, rewards, terminated, truncated, infos = self.env.step(self._actions)

        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        rewards   = np.asarray(rewards).reshape(-1)          # (N,)
        terminated = np.asarray(terminated).reshape(-1)      # (N,)
        truncated  = np.asarray(truncated).reshape(-1)       # (N,)
        dones      = np.logical_or(terminated, truncated)    # (N,)

        # Coerce infos into list of dicts length N
        if isinstance(infos, dict):
            infos = [infos for _ in range(self.num_envs)]
        elif not isinstance(infos, list):
            infos = list(infos)
        if len(infos) != self.num_envs:
            infos = (infos + [{}] * self.num_envs)[: self.num_envs]

        # Auto-reset when all done
        if dones.all():
            if self.store_terminal_observation_info:
                for i in range(self.num_envs):
                    if isinstance(infos[i], dict):
                        infos[i] = dict(infos[i])
                        infos[i]["terminal_observation"] = obs[i].copy()
            try:
                obs, _ = self.env.reset()
            except Exception:
                obs = self.env.reset()
            obs = np.asarray(obs)
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)

        self._last_obs = obs
        return obs, rewards, dones, infos

    # Required stubs
    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return []

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *args, indices: VecEnvIndices = None, **kwargs) -> List[Any]:
        return []

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False] * self.num_envs

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        try:
            self.env.reset(seed=seed)
        except TypeError:
            # older gym signature without seed kw
            pass
        return [seed] * self.num_envs

    def get_images(self) -> Sequence[np.ndarray]:
        return []
