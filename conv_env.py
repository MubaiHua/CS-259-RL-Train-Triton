# conv_env.py
import torch
import triton
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from triton_kernels import conv2d_bench
from config_conv2d import ACTION_CHOICES, ACTION_PARAM_NAMES,  \
                   R_W_TFLOPS, R_W_VRAM, R_ERROR_PENALTY, R_MAX_VRAM_PENALTY_VAL

class TritonConv2dEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, train_configs, render_mode=None):
        super(TritonConv2dEnv, self).__init__()
        
        self.train_configs = train_configs
        self.shape_params = self.train_configs[0]
        
        self.render_mode = render_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        action_dims = [len(ACTION_CHOICES[name]) for name in ACTION_PARAM_NAMES]
        self.action_space = gym.spaces.MultiDiscrete(action_dims)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)
        self.max_obs_values = np.array([
            128, 1024, 1024, 1024, 1024, 15, 15, 4, 4, 15, 15,
            1024, 1024, 128*1024*1024, 128*1024*1024, 1024*1024*15*15
        ], dtype=np.float32)
        
        self.w_tflops, self.w_vram = R_W_TFLOPS, R_W_VRAM
        self.error_penalty, self.max_vram_penalty_val = R_ERROR_PENALTY, R_MAX_VRAM_PENALTY_VAL

        self.best_reward_current_size = -float('inf')
        self.best_config_current_size, self.best_metrics_current_size = None, {}
        self.current_step_in_episode = 0
        
        self.action_history = {}
        self._reconfigure_and_init(self.shape_params)

    def _reconfigure_and_init(self, shape_params):
        self.shape_params = shape_params
        self.size_key = str(shape_params)
        if self.size_key not in self.action_history:
            self.action_history[self.size_key] = set()

    # === START MODIFICATION ===
    def set_test_shape(self, shape_params):
        """Method specifically for testing to set a non-random problem shape."""
        self._reconfigure_and_init(shape_params)
        return self._get_obs(), self._get_info()
    # === END MODIFICATION ===

    def _decode_action(self, discrete_action: np.ndarray):
        return {name: ACTION_CHOICES[name][int(discrete_action[i])] for i, name in enumerate(ACTION_PARAM_NAMES)}

    def _get_obs(self):
        N, C_in, H, W, C_out, KH, KW, stride, padding = self.shape_params
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        H_out = (H + 2 * padding_h - KH) // stride_h + 1
        W_out = (W + 2 * padding_w - KW) // stride_w + 1
        
        raw_obs = np.array([
            N, C_in, H, W, C_out, KH, KW, stride_h, stride_w, padding_h, padding_w,
            H_out, W_out, float(N*C_in*H*W), float(N*C_out*H_out*W_out), float(C_out*C_in*KH*KW)
        ], dtype=np.float32)
        
        return np.clip(raw_obs / self.max_obs_values, 0.0, 1.0)
    
    def _get_info(self, config=None, metrics=None, error=None):
        info = {"step": self.current_step_in_episode, "shape": self.shape_params}
        if config: info["config"] = config
        if metrics: info["metrics"] = metrics
        if error: info["error"] = str(error)
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        new_shape_params = random.choice(self.train_configs)
        self._reconfigure_and_init(new_shape_params)

        self.current_step_in_episode = 0
        return self._get_obs(), {}
    
    def step(self, action):
        config_dict = self._decode_action(action)
        obs = self._get_obs()
        terminated = True
        truncated = False
        reward = self.error_penalty
        runtime_ms, vram_mb, tflops = float('inf'), self.max_vram_penalty_val, 0
        
        try:
            runtime_ms = conv2d_bench(*self.shape_params, config_dict=config_dict)
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            N, C_in, H, W, C_out, KH, KW, stride, padding = self.shape_params
            H_out = (H + 2*padding[0] - KH) // stride[0] + 1
            W_out = (W + 2*padding[1] - KW) // stride[1] + 1
            total_ops = 2.0 * N * C_out * H_out * W_out * (C_in * KH * KW)
            tflops = total_ops / (runtime_ms * 1e-3) / 1e12 if runtime_ms > 0 else 0
            
            reward = (self.w_tflops * tflops) - (self.w_vram * vram_mb)

            action_tuple = tuple(action)
            if action_tuple not in self.action_history[self.size_key]:
                reward += 0.5
                self.action_history[self.size_key].add(action_tuple)

            if reward > self.best_reward_current_size:
                self.best_reward_current_size, self.best_config_current_size = reward, config_dict.copy()
                self.best_metrics_current_size = {"tflops": tflops, "vram_mb": vram_mb, "runtime_ms": runtime_ms}
        except Exception as e:
            pass
        
        metrics_log = {"runtime_ms": runtime_ms, "vram_mb": vram_mb, "tflops": tflops, "reward": reward}
        info = self._get_info(config=config_dict, metrics=metrics_log)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass