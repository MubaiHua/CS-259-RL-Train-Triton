# rl_env.py
import torch
import triton
import triton.language as tl
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Assuming triton_kernels.py and config.py are in the same directory or accessible in PYTHONPATH
from triton_kernels import rl_managed_matmul_kernel
from config import ACTION_CHOICES, ACTION_PARAM_NAMES, FIXED_GROUP_SIZE_M, \
                   R_W_TFLOPS, R_W_VRAM, R_ERROR_PENALTY, R_MAX_VRAM_PENALTY_VAL

class TritonMatmulEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, M=1024, N=1024, K=1024, render_mode=None):
        super(TritonMatmulEnv, self).__init__()
        self.M, self.N, self.K = M, N, K
        self.render_mode = render_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define action space (MultiDiscrete)
        action_dims = [len(ACTION_CHOICES[name]) for name in ACTION_PARAM_NAMES]
        self.action_space = spaces.MultiDiscrete(action_dims)

        # Define observation space (M, N, K)
        # Max values are placeholders, adjust if needed
        self.observation_space = spaces.Box(low=16, high=16384, shape=(3,), dtype=np.float32)

        # Reward weights
        self.w_tflops = R_W_TFLOPS
        self.w_vram = R_W_VRAM

        self.error_penalty = R_ERROR_PENALTY
        self.max_vram_penalty_val = R_MAX_VRAM_PENALTY_VAL

        # For logging best results for the current M,N,K configuration
        self.best_reward_current_size = -float('inf')
        self.best_config_current_size = None
        self.best_metrics_current_size = {}

        self.current_step_in_episode = 0 # Tracks steps for current M,N,K

        # Prepare tensors (will be resized in reset if M,N,K change)
        self._init_tensors()

    def _init_tensors(self):
        """Initializes or resizes tensors based on self.M, self.N, self.K."""
        self.A_gpu = torch.randn((self.M, self.K), dtype=torch.float32, device=self.device)
        self.B_gpu = torch.randn((self.K, self.N), dtype=torch.float32, device=self.device)
        self.C_gpu = torch.empty((self.M, self.N), dtype=torch.float32, device=self.device)
        if self.render_mode == "human":
            print(f"Initialized tensors for M={self.M}, N={self.N}, K={self.K} on {self.device}")

    def reconfigure_size(self, M, N, K):
        """Allows changing the matrix dimensions for the environment."""
        self.M, self.N, self.K = M, N, K
        self._init_tensors()
        # Reset bests for the new size
        self.best_reward_current_size = -float('inf')
        self.best_config_current_size = None
        self.best_metrics_current_size = {}
        if self.render_mode == "human":
            print(f"Environment reconfigured for M={M}, N={N}, K={K}")
        return self.reset()


    def _decode_action(self, discrete_action: np.ndarray):
        config = {}
        for i, param_name in enumerate(ACTION_PARAM_NAMES):
            choice_index = discrete_action[i]
            config[param_name] = ACTION_CHOICES[param_name][choice_index]

        config['GROUP_SIZE_M'] = FIXED_GROUP_SIZE_M
        return config

    def _get_obs(self):
        return np.array([self.M, self.N, self.K], dtype=np.float32)

    def _get_info(self, config=None, metrics=None, error=None):
        info = {"step": self.current_step_in_episode, "M": self.M, "N": self.N, "K": self.K}
        if config: info["config"] = config
        if metrics: info["metrics"] = metrics
        if error: info["error"] = str(error)
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_in_episode = 0
        # Tensors are already initialized or reconfigured by reconfigure_size or __init__
        if self.render_mode == "human" and options is None: # Avoid double print if called by reconfigure
             print(f"Environment reset for M={self.M}, N={self.N}, K={self.K}.")
        return self._get_obs(), self._get_info()

    def step(self, action): # action is now a discrete numpy array of indices
        self.current_step_in_episode += 1
        config_dict = self._decode_action(action)

        A, B, C = self.A_gpu, self.B_gpu, self.C_gpu

        grid_fn = lambda META: (triton.cdiv(self.M, META['BLOCK_SIZE_M']) * triton.cdiv(self.N, META['BLOCK_SIZE_N']),)
        obs = self._get_obs()
        terminated = True # Each step is an episode for this tuning task
        truncated = False

        runtime_ms, vram_mb, tflops = 0, self.max_vram_penalty_val, 0
        current_error = None

        if self.device == 'cpu': # Cannot run Triton kernel on CPU
            reward = self.error_penalty
            current_error = "CUDA device not available"
            runtime_ms, vram_mb, tflops = float('inf'), self.max_vram_penalty_val, 0
        else:
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                kernel_fn = lambda: rl_managed_matmul_kernel[grid_fn](
                    A, B, C, self.M, self.N, self.K,
                    A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                    **config_dict
                )
                runtime_ms = triton.testing.do_bench(kernel_fn, warmup=10, rep=50)
                torch.cuda.synchronize()

                vram_bytes = torch.cuda.max_memory_allocated()
                vram_mb = vram_bytes / (1024 * 1024)
                if runtime_ms < 1e-6 : runtime_ms = 1e-6
                tflops = (2 * self.M * self.N * self.K) / (runtime_ms * 1e-3) / 1e12

                reward = (self.w_tflops * tflops) - (self.w_vram * vram_mb)

                if reward > self.best_reward_current_size:
                    self.best_reward_current_size = reward
                    self.best_config_current_size = config_dict.copy()
                    self.best_metrics_current_size = {"tflops": tflops, "vram_mb": vram_mb, "runtime_ms": runtime_ms}

            except Exception as e:
                if self.render_mode == "human": print(f"Error for config {config_dict} (M={self.M},N={self.N},K={self.K}): {e}")
                reward = self.error_penalty
                current_error = e
                runtime_ms, vram_mb, tflops = float('inf'), self.max_vram_penalty_val, 0

        metrics_log = {"runtime_ms": runtime_ms, "vram_mb": vram_mb, "tflops": tflops, "reward": reward}
        info = self._get_info(config=config_dict, metrics=metrics_log, error=current_error)

        if self.render_mode == "human":
            print(f"Size M={self.M},N={self.N},K={self.K} | Step: {self.current_step_in_episode}, Config: {config_dict}, Metrics: {metrics_log}")
            if current_error: print(f"Error: {current_error}")

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"\n--- Current Best for M={self.M},N={self.N},K={self.K} ---")
            print(f"Reward: {self.best_reward_current_size:.4f}")
            if self.best_config_current_size:
                print(f"Config: {self.best_config_current_size}")
                print(f"Metrics: {self.best_metrics_current_size}")
            print("------------------------------------")

    def close(self):
        if self.render_mode == "human": print("Environment closed.")
