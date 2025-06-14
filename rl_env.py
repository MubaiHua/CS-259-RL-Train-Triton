# rl_env.py
import torch
import triton
import triton.language as tl
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from triton_kernels import rl_managed_matmul_kernel, matmul_bench
from config import ACTION_CHOICES, ACTION_PARAM_NAMES, FIXED_GROUP_SIZE_M, \
                   R_W_TFLOPS, R_W_VRAM, R_ERROR_PENALTY, R_MAX_VRAM_PENALTY_VAL

class TritonMatmulEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, train_sizes, render_mode=None):
        super(TritonMatmulEnv, self).__init__()
        
        self.train_sizes = train_sizes
        # Initialize with the first size in the list
        self.M, self.N, self.K = self.train_sizes[0]
        
        self.render_mode = render_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        action_dims = [len(ACTION_CHOICES[name]) for name in ACTION_PARAM_NAMES]
        self.action_space = spaces.MultiDiscrete(action_dims)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.max_dim = 16384.0

        self.w_tflops, self.w_vram = R_W_TFLOPS, R_W_VRAM
        self.error_penalty, self.max_vram_penalty_val = R_ERROR_PENALTY, R_MAX_VRAM_PENALTY_VAL

        self.current_step_in_episode = 0
        self.action_history = {}

        self._reconfigure_and_init(self.M, self.N, self.K)

    def _reconfigure_and_init(self, M, N, K):
        self.M, self.N, self.K = M, N, K
        self.size_key = f"{M}-{N}-{K}"
        if self.size_key not in self.action_history:
            self.action_history[self.size_key] = set()

        if self.device == 'cuda':
            self.A_gpu = torch.randn((self.M, self.K), dtype=torch.float32, device=self.device)
            self.B_gpu = torch.randn((self.K, self.N), dtype=torch.float32, device=self.device)
            self.C_gpu = torch.empty((self.M, self.N), dtype=torch.float32, device=self.device)

    # === START MODIFICATION ===
    def set_test_size(self, M, N, K):
        """Method specifically for testing to set a non-random problem size."""
        self._reconfigure_and_init(M, N, K)
        return self._get_obs(), self._get_info()
    # === END MODIFICATION ===
    
    def _decode_action(self, discrete_action: np.ndarray):
        config = {name: ACTION_CHOICES[name][discrete_action[i]] for i, name in enumerate(ACTION_PARAM_NAMES)}
        config['GROUP_SIZE_M'] = FIXED_GROUP_SIZE_M
        return config

    def _get_obs(self):
        m_norm, n_norm, k_norm = self.M / self.max_dim, self.N / self.max_dim, self.K / self.max_dim
        log_total_size = np.log2(self.M * self.N * self.K) / np.log2(self.max_dim**3)
        ratio_mn = np.tanh((self.M / self.N) if self.N > 0 else 1.0)
        ratio_mk = np.tanh((self.M / self.K) if self.K > 0 else 1.0)
        ratio_nk = np.tanh((self.N / self.K) if self.K > 0 else 1.0)
        return np.array([m_norm, n_norm, k_norm, log_total_size, ratio_mn, ratio_mk, ratio_nk], dtype=np.float32)

    def _get_info(self, config=None, metrics=None, error=None):
        info = {"step": self.current_step_in_episode, "M": self.M, "N": self.N, "K": self.K}
        if config: info["config"] = config
        if metrics: info["metrics"] = metrics
        if error: info["error"] = str(error)
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly pick a new size for the next training episode
        new_M, new_N, new_K = random.choice(self.train_sizes)
        self._reconfigure_and_init(new_M, new_N, new_K)

        self.current_step_in_episode = 0
        return self._get_obs(), {}

    def step(self, action):
        config_dict = self._decode_action(action)
        obs = self._get_obs()
        terminated = True
        truncated = False
        reward = self.error_penalty
        runtime_ms, vram_mb, tflops = float('inf'), self.max_vram_penalty_val, 0
        current_error = None
        
        try:
            runtime_ms = matmul_bench(self.M, self.N, self.K, self.A_gpu, self.B_gpu, self.C_gpu, config_dict)
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            tflops = (2 * self.M * self.N * self.K) / (runtime_ms * 1e-3) / 1e12 if runtime_ms > 0 else 0
            
            reward = (self.w_tflops * tflops) - (self.w_vram * vram_mb)

            action_tuple = tuple(action)
            if action_tuple not in self.action_history[self.size_key]:
                reward += 0.5
                self.action_history[self.size_key].add(action_tuple)
        except Exception as e:
            current_error = e
        
        metrics_log = {"runtime_ms": runtime_ms, "vram_mb": vram_mb, "tflops": tflops, "reward": reward}
        info = self._get_info(config=config_dict, metrics=metrics_log, error=current_error)
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass