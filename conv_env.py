import torch
import triton
import triton.language as tl
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Assuming triton_kernels.py and config.py are in the same directory or accessible in PYTHONPATH
from triton_kernels import rl_managed_matmul_kernel, matmul_bench, conv2d_bench
from config_conv2d import ACTION_CHOICES, ACTION_PARAM_NAMES,  \
                   R_W_TFLOPS, R_W_VRAM, R_ERROR_PENALTY, R_MAX_VRAM_PENALTY_VAL



class TritonConv2dEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 N=1, C_in=3, H=32, W=32,
                 C_out=8, KH=3, KW=3,
                 stride=(1, 1), padding=(1, 1),
                 render_mode=None):
        """
        Environment to tune Triton Conv2D kernel hyperparameters via RL.
        Observations: the convolution shape and parameters.
        Actions: select from discrete choices for BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_CIN, BLOCK_SIZE_COUT, NUM_WARPS, NUM_STAGES.
        Reward: based on measured TFLOPS minus VRAM penalty.
        """
        super(TritonConv2dEnv, self).__init__()
        self.render_mode = render_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Convolution dimensions
        self.N = N
        self.C_in = C_in
        self.H = H
        self.W = W
        self.C_out = C_out
        self.KH = KH
        self.KW = KW
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding

        # Define action space
        # ACTION_PARAM_NAMES and ACTION_CHOICES should be defined globally or imported
        action_dims = [len(ACTION_CHOICES[name]) for name in ACTION_PARAM_NAMES]
        self.action_space = gym.spaces.MultiDiscrete(action_dims)

        # Observation space: we include all conv parameters as floats
        # [N, C_in, H, W, C_out, KH, KW, stride_h, stride_w, padding_h, padding_w]
        low = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float32)
        high = np.array([
            16384, 16384, 65536, 65536, 16384, 15, 15, 16, 16, 16, 16
        ], dtype=np.float32)
        # The above high values are placeholders; adjust as needed.
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Reward weights (assumed defined globally)
        self.w_tflops = R_W_TFLOPS
        self.w_vram = R_W_VRAM
        self.error_penalty = R_ERROR_PENALTY
        self.max_vram_penalty_val = R_MAX_VRAM_PENALTY_VAL

        # Best tracking for current shape
        self.best_reward_current_size = -float('inf')
        self.best_config_current_size = None
        self.best_metrics_current_size = {}

        self.current_step_in_episode = 0

        # Prepare input/output/weight tensors
        self._init_tensors()

    def _init_tensors(self):
        """Initialize or resize tensors based on self.N, self.C_in, self.H, self.W, self.C_out, KH, KW."""
        # Input activation x, weight w, output y placeholder
        # We use float16 inputs/weights but can adjust dtype if desired
        if self.device == 'cuda':
            # For benchmarking Triton conv2d, typically input dtype is float16
            # But you can choose float16 or float32; adjust conv2d kernel accordingly.
            self.x = torch.randn((self.N, self.C_in, self.H, self.W),
                                 dtype=torch.float16, device=self.device)
            self.w = torch.randn((self.C_out, self.C_in, self.KH, self.KW),
                                 dtype=torch.float16, device=self.device)
            self.H_out = (self.H + 2*self.padding_h - self.KH) // self.stride_h + 1
            self.W_out = (self.W + 2*self.padding_w - self.KW) // self.stride_w + 1
            #self.y = torch.empty((self.N, self.C_out, self.H_out, self.W_out), device=self.device, dtype=torch.float16)
            # Output will be allocated in conv2d_bench each time
            if self.render_mode == "human":
                print(f"Initialized conv2d tensors: x[{self.N},{self.C_in},{self.H},{self.W}], "
                      f"w[{self.C_out},{self.C_in},{self.KH},{self.KW}] on {self.device}")
        else:
            assert 0, "CUDA not available; environment initialized on CPU for reference only."

    def reconfigure_size(self, N, C_in, H, W, C_out, KH, KW, stride=(1,1), padding=(0,0)):
        """
        Change convolution dimensions. Resets best tracking for the new shape.
        """
        self.N = N
        self.C_in = C_in
        self.H = H
        self.W = W
        self.C_out = C_out
        self.KH = KH
        self.KW = KW
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding
        self.best_reward_current_size = -float('inf')
        self.best_config_current_size = None
        self.best_metrics_current_size = {}
        self._init_tensors()
        if self.render_mode == "human":
            print(f"Environment reconfigured for Conv2D: N={N},C_in={C_in},H={H},W={W},"
                  f"C_out={C_out},KH={KH},KW={KW},stride={stride},padding={padding}")
        return self.reset()

    def _decode_action(self, discrete_action: np.ndarray):
        """
        Map discrete action indices to Triton conv2d config dict:
        {'BLOCK_SIZE_H': ..., 'BLOCK_SIZE_W': ..., 'BLOCK_SIZE_CIN': ..., 'BLOCK_SIZE_COUT': ..., 'NUM_WARPS': ..., 'NUM_STAGES': ...}
        """
        config = {}
        for i, param_name in enumerate(ACTION_PARAM_NAMES):
            choice_index = int(discrete_action[i])
            config[param_name] = ACTION_CHOICES[param_name][choice_index]
        # No GROUP_SIZE_M for conv2d; if present, ignore or set to a default:
        # config['GROUP_SIZE_M'] = FIXED_GROUP_SIZE_M  # not used in conv2d kernel
        return config

    def _get_obs(self):
        """
        Observation: numpy array of floats representing the conv dimensions.
        """
        return np.array([
            float(self.N), float(self.C_in), float(self.H), float(self.W),
            float(self.C_out), float(self.KH), float(self.KW),
            float(self.stride_h), float(self.stride_w),
            float(self.padding_h), float(self.padding_w)
        ], dtype=np.float32)

    def _get_info(self, config=None, metrics=None, error=None):
        info = {
            "step": self.current_step_in_episode,
            "N": self.N, "C_in": self.C_in, "H": self.H, "W": self.W,
            "C_out": self.C_out, "KH": self.KH, "KW": self.KW,
            "stride": (self.stride_h, self.stride_w),
            "padding": (self.padding_h, self.padding_w),
        }
        if config is not None:
            info["config"] = config
        if metrics is not None:
            info["metrics"] = metrics
        if error is not None:
            info["error"] = str(error)
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_in_episode = 0
        # Tensors already initialized or reconfigured
        if self.render_mode == "human" and options is None:
            print(f"Conv2dEnv reset for shape: N={self.N},C_in={self.C_in},H={self.H},W={self.W},"
                  f"C_out={self.C_out},KH={self.KH},KW={self.KW},"
                  f"stride=({self.stride_h},{self.stride_w}),padding=({self.padding_h},{self.padding_w})")
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Each action corresponds to trying one Triton conv2d configuration. We run one benchmark and return reward.
        Episodes are single-step (terminated=True after each).
        """
        self.current_step_in_episode += 1
        config_dict = self._decode_action(action)
        obs = self._get_obs()
        terminated = True
        truncated = False

        runtime_ms = float('inf')
        vram_mb = self.max_vram_penalty_val
        tflops = 0.0
        current_error = None

        if self.device != 'cuda':
            # Cannot run Triton conv2d on CPU
            reward = self.error_penalty
            current_error = "CUDA device not available"
        else:
            try:
                # Reset and sync before timing
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                # Run Triton conv2d benchmark
                # self.x, self.w are float16 tensors on CUDA
                #print("start bench")
                runtime_ms = conv2d_bench(self.N, self.C_in, self.H, self.W,
                                            self.C_out, self.KH, self.KW,
                                          stride=(self.stride_h, self.stride_w),
                                          padding=(self.padding_h, self.padding_w),
                                          config_dict=config_dict)
                #print("end bench")
                #print(runtime_ms)
                torch.cuda.synchronize()
                # Measure peak memory
                vram_bytes = torch.cuda.max_memory_allocated()
                vram_mb = vram_bytes / (1024 * 1024)
                if runtime_ms < 1e-6:
                    runtime_ms = 1e-6
                # Compute TFLOPS for conv2d:
                # Total multiply-add operations: N * C_out * H_out * W_out * (2 * C_in * KH * KW)
                H_out = (self.H + 2*self.padding_h - self.KH) // self.stride_h + 1
                W_out = (self.W + 2*self.padding_w - self.KW) // self.stride_w + 1
                total_ops = 2.0 * self.N * self.C_out * H_out * W_out * (self.C_in * self.KH * self.KW)
                tflops = total_ops / (runtime_ms * 1e-3) / 1e12

                # Reward combining TFLOPS (higher is better) and VRAM penalty (lower is better)
                reward = (self.w_tflops * tflops) - (self.w_vram * vram_mb)

                # Update best for this shape if improved
                # if reward > self.best_reward_current_size:
                #     self.best_reward_current_size = reward
                #     self.best_config_current_size = config_dict.copy()
                #     self.best_metrics_current_size = {
                #         "tflops": tflops,
                #         "vram_mb": vram_mb,
                #         "runtime_ms": runtime_ms
                #     }

            except Exception as e:
                # On error (e.g., invalid config causing compile/runtime failure), apply penalty
                if self.render_mode == "human":
                    print(f"Error for conv2d config {config_dict}: {e}")
                reward = self.error_penalty
                # current_error = e
                #raise e

        metrics_log = {
            "runtime_ms": runtime_ms,
            "vram_mb": vram_mb,
            "tflops": tflops,
            "reward": reward
        }
        info = self._get_info(config=config_dict, metrics=metrics_log, error=current_error)

        if self.render_mode == "human":
            print(f"Conv2D Step {self.current_step_in_episode} | Shape N={self.N},C_in={self.C_in},H={self.H},W={self.W},"
                  f"C_out={self.C_out},KH={self.KH},KW={self.KW}, stride=({self.stride_h},{self.stride_w}),"
                  f"padding=({self.padding_h},{self.padding_w})\n"
                  f"  Config: {config_dict}\n"
                  f"  Metrics: {metrics_log}\n"
                  f"  Best so far: reward={self.best_reward_current_size:.4f}, config={self.best_config_current_size}, "
                  f"metrics={self.best_metrics_current_size}")

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"\n--- Conv2D Current Best for shape N={self.N},C_in={self.C_in},H={self.H},W={self.W},"
                  f"C_out={self.C_out},KH={self.KH},KW={self.KW} ---")
            print(f"  Best Reward: {self.best_reward_current_size:.4f}")
            if self.best_config_current_size:
                print(f"  Best Config: {self.best_config_current_size}")
                print(f"  Best Metrics: {self.best_metrics_current_size}")
            print("------------------------------------")

    def close(self):
        if self.render_mode == "human":
            print("TritonConv2dEnv closed.")