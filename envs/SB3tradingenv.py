# utils/sb3_adapter.py
from typing import List, Any, Optional, Sequence, Type, Union, Tuple, Dict
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices
from envs.tradingenv import TradingEnv

ASSET_PRICE, INVENTORY_INDEX, TIME_INDEX, CASH = 0, 1, 2, 3

# class SB3TradingVecEnv(VecEnv):
#     """
#     Adapter for a *batched* TradingEnv (return_vectorized=True).
#     We ALWAYS infer num_envs (N) from a real reset() to avoid 0-length issues.
#     """
#     def __init__(self, trading_env: gym.Env, store_terminal_observation_info: bool = True):
#         assert getattr(trading_env, "return_vectorized", False), (
#             "TradingEnv must be created with return_vectorized=True."
#         )
#         self.env = trading_env
#         self.store_terminal_observation_info = store_terminal_observation_info

#         # ---- FORCE-INFER N FROM A REAL RESET ----
#         try:
#             obs, _ = self.env.reset()
#         except Exception:
#             # legacy reset returning only obs
#             obs = self.env.reset()
#         obs = np.asarray(obs)

#         if obs.ndim == 2:
#             N = obs.shape[0]
#         elif obs.ndim == 1:
#             N = 1
#             obs = obs.reshape(1, -1)  # normalize
#         else:
#             raise RuntimeError(f"Unexpected obs ndim from reset(): {obs.shape}")

#         if N <= 0:
#             raise RuntimeError(f"Inferred num_envs N={N} from reset(); cannot build VecEnv.")

#         # ---- INIT BASE VecEnv ----
#         super().__init__(int(N), self.env.observation_space, self.env.action_space)

#         # Placeholder actions (N, act_dim)
#         sample = self.env.action_space.sample()
#         sample = np.asarray(sample, dtype=np.float32)
#         if sample.ndim == 1:
#             self._actions = np.repeat(sample.reshape(1, -1), N, axis=0)
#         else:
#             # assume already (N, act_dim)
#             self._actions = np.zeros_like(sample, dtype=np.float32)

#         # Cache last obs (VecEnv may call render etc. before first step)
#         self._last_obs = obs

#     def reset(self) -> VecEnvObs:
#         try:
#             obs, _ = self.env.reset()
#         except Exception:
#             obs = self.env.reset()
#         obs = np.asarray(obs)
#         if obs.ndim == 1:
#             obs = obs.reshape(1, -1)
#         self._last_obs = obs
#         return obs

#     def step_async(self, actions: np.ndarray) -> None:
#         self._actions = np.asarray(actions, dtype=np.float32)

#     def step_wait(self) -> VecEnvStepReturn:
#         # Env must return: (obs, rewards, terminated, truncated, infos) in batched shape
#         obs, rewards, terminated, truncated, infos = self.env.step(self._actions)

#         obs = np.asarray(obs)
#         if obs.ndim == 1:
#             obs = obs.reshape(1, -1)
#         rewards   = np.asarray(rewards).reshape(-1)          # (N,)
#         terminated = np.asarray(terminated).reshape(-1)      # (N,)
#         truncated  = np.asarray(truncated).reshape(-1)       # (N,)
#         dones      = np.logical_or(terminated, truncated)    # (N,)

#         # Coerce infos into list of dicts length N
#         if isinstance(infos, dict):
#             infos = [infos for _ in range(self.num_envs)]
#         elif not isinstance(infos, list):
#             infos = list(infos)
#         if len(infos) != self.num_envs:
#             infos = (infos + [{}] * self.num_envs)[: self.num_envs]

#         # Auto-reset when all done
#         if dones.all():
#             if self.store_terminal_observation_info:
#                 for i in range(self.num_envs):
#                     if isinstance(infos[i], dict):
#                         infos[i] = dict(infos[i])
#                         infos[i]["terminal_observation"] = obs[i].copy()
#             try:
#                 obs, _ = self.env.reset()
#             except Exception:
#                 obs = self.env.reset()
#             obs = np.asarray(obs)
#             if obs.ndim == 1:
#                 obs = obs.reshape(1, -1)

#         self._last_obs = obs
#         return obs, rewards, dones, infos

#     # Required stubs
#     def close(self) -> None:
#         try:
#             self.env.close()
#         except Exception:
#             pass

#     def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
#         return []

#     def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
#         pass

#     def env_method(self, method_name: str, *args, indices: VecEnvIndices = None, **kwargs) -> List[Any]:
#         return []

#     def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
#         return [False] * self.num_envs

#     def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
#         try:
#             self.env.reset(seed=seed)
#         except TypeError:
#             # older gym signature without seed kw
#             pass
#         return [seed] * self.num_envs

#     def get_images(self) -> Sequence[np.ndarray]:
#         return []

class HyperparameterConfig:
    """Centralized configuration for all hyperparameters"""
    def __init__(self, 
                 # Environment params
                 terminal_time: float = 1.0,
                 n_steps: int = 100,
                 arrival_rate: float = 10.0,
                 sigma: float = 0.1,
                 phi: float = 0.5,
                 alpha: float = 0.001,
                 initial_inventory: Tuple[int, int] = (-4, 5),
                 initial_price: float = 100.0,
                 fill_exponent: float = 1.0,
                 
                 # Wrapper params
                 state_indices: List[int] = None,
                 
                 # Training params
                 num_trajectories: int = 1000,
                 normalize_obs: bool = False,
                 normalize_actions: bool = False,
                 
                 # RL algorithm params (for reference)
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2):
        
        # Environment
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.arrival_rate = arrival_rate
        self.sigma = sigma
        self.phi = phi
        self.alpha = alpha
        self.initial_inventory = initial_inventory
        self.initial_price = initial_price
        self.fill_exponent = fill_exponent
        
        # Wrapper
        self.state_indices = state_indices or [INVENTORY_INDEX, TIME_INDEX]
        
        # Training
        self.num_trajectories = num_trajectories
        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions
        
        # RL algorithm
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class FlexibleReduceStateSizeWrapper(gym.Wrapper):
    """
    Enhanced wrapper with hyperparameter tuning support
    """
    
    def __init__(self, env, config: HyperparameterConfig = None):
        super(FlexibleReduceStateSizeWrapper, self).__init__(env)
        
        self.config = config or HyperparameterConfig()
        
        assert type(env.observation_space) == gym.spaces.box.Box, "Observation space must be Box"
        
        # Dynamic state indices based on config
        self.list_of_state_indices = self.config.state_indices
        
        self.observation_space = gym.spaces.box.Box(
            low=env.observation_space.low[self.list_of_state_indices],
            high=env.observation_space.high[self.list_of_state_indices],
            dtype=np.float64,
        )
        
        # For hyperparameter tuning tracking
        self.episode_returns = []
        self.current_episode_return = 0

    def reset(self):
        """Reset the environment and tracking"""
        obs = self.env.reset()
        self.current_episode_return = 0
        return obs[:, self.list_of_state_indices]

    def step(self, action):
        """Step with enhanced tracking for hyperparameter tuning"""
        obs, reward, done, info = self.env.step(action)
        
        # Track episode returns for performance monitoring
        self.current_episode_return += np.mean(reward)
        if done.any():
            self.episode_returns.append(self.current_episode_return)
            self.current_episode_return = 0
            
        return obs[:, self.list_of_state_indices], reward, done, info

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for hyperparameter tuning"""
        if not self.episode_returns:
            return {}
        
        returns = np.array(self.episode_returns)
        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'num_episodes': len(returns)
        }

    @property
    def spec(self):
        return self.env.spec
    
class TunableStableBaselinesTradingEnvironment(VecEnv):
    """
    Enhanced SB3 environment adapter with hyperparameter tuning support
    """
    
    def __init__(
        self,
        trading_env: TradingEnv,
        config: HyperparameterConfig = None,
        store_terminal_observation_info: bool = True,
        track_performance: bool = True
    ):
        self.env = trading_env
        self.config = config or HyperparameterConfig()
        self.store_terminal_observation_info = store_terminal_observation_info
        self.track_performance = track_performance
        self.actions: np.ndarray = self.env.action_space.sample()
        
        # Performance tracking for hyperparameter tuning
        self.episode_returns = []
        self.current_episode_returns = np.zeros(self.env.num_trajectories)
        
        super().__init__(self.env.num_trajectories, self.env.observation_space, self.env.action_space)

    def reset(self) -> VecEnvObs:
        reset_obs = self.env.reset()
        if self.track_performance:
            self.current_episode_returns = np.zeros(self.env.num_trajectories)
        return reset_obs

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.env.step(self.actions)
        
        # Track performance metrics
        if self.track_performance:
            self.current_episode_returns += rewards
            for i, done in enumerate(dones):
                if done:
                    self.episode_returns.append(self.current_episode_returns[i])
                    self.current_episode_returns[i] = 0
        
        # Handle terminal states (SB3 convention)
        if dones.min():
            if self.store_terminal_observation_info:
                infos = self._add_terminal_observations(infos, obs)
            obs = self.env.reset()
            if self.track_performance:
                self.current_episode_returns = np.zeros(self.env.num_trajectories)
                
        return obs, rewards, dones, infos

    def _add_terminal_observations(self, infos: List[Dict], obs: np.ndarray) -> List[Dict]:
        """Add terminal observations to info dicts"""
        infos = infos.copy()
        if isinstance(infos, list):
            for count, info in enumerate(infos):
                info["terminal_observation"] = obs[count, :]
        elif isinstance(infos, dict):
            infos["terminal_observation"] = obs[0, :]
        return infos

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for hyperparameter tuning"""
        if not self.episode_returns:
            return {}
        
        returns = np.array(self.episode_returns)
        return {
            'mean_episode_return': float(np.mean(returns)),
            'std_episode_return': float(np.std(returns)),
            'min_episode_return': float(np.min(returns)),
            'max_episode_return': float(np.max(returns)),
            'total_episodes': len(returns)
        }

    def close(self) -> None:
        if hasattr(self.env, 'close'):
            self.env.close()

    # Required VecEnv methods with basic implementations
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        indices = self._get_indices(indices)
        return [getattr(self.env, attr_name) for _ in indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        indices = self._get_indices(indices)
        for _ in indices:
            setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        indices = self._get_indices(indices)
        return [getattr(self.env, method_name)(*method_args, **method_kwargs) for _ in indices]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        indices = self._get_indices(indices)
        return [isinstance(self.env, wrapper_class) for _ in indices]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.env.seed(seed)

    def get_images(self) -> Sequence[np.ndarray]:
        return []

    def _get_indices(self, indices: VecEnvIndices) -> List[int]:
        """Convert VecEnvIndices to list of indices"""
        if indices is None:
            return list(range(self.num_trajectories))
        elif isinstance(indices, int):
            return [indices]
        else:
            return indices

    @property
    def num_trajectories(self):
        return self.env.num_trajectories

    @property
    def n_steps(self):
        return self.env.n_steps