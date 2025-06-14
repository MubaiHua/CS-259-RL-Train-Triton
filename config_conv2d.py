# config.py
from random import randint
import numpy as np

def range_list(start, end, step=1):
    """Generates a list of integers from start to end with a specified step."""
    return list(range(start, end + 1, step))

def power_of_two_range(start_exp, end_exp):
    """Generates a list of powers of two from 2^start_exp to 2^end_exp."""
    return [2 ** i for i in range(start_exp, end_exp + 1)]

# Action choices for the RL agent and the discrete search space for Triton's autotuner
ACTION_CHOICES = {
    'BLOCK_SIZE_H': power_of_two_range(2,5),
    'BLOCK_SIZE_W': power_of_two_range(2,5),
    'BLOCK_SIZE_CIN': power_of_two_range(4,6),
    'BLOCK_SIZE_COUT': power_of_two_range(4,6),
    'num_warps': power_of_two_range(2,5),
    'num_stages': [1, 2, 3, 4, 5]
}

ACTION_PARAM_NAMES = ['BLOCK_SIZE_H', 'BLOCK_SIZE_W', 'BLOCK_SIZE_CIN', 'BLOCK_SIZE_COUT', 'num_warps', 'num_stages']

AUTOTUNER_BENCHMARK_ACTION_CHOICES = {
    'BLOCK_SIZE_H': power_of_two_range(2,5),
    'BLOCK_SIZE_W': power_of_two_range(2,5),
    'BLOCK_SIZE_CIN': power_of_two_range(4,6),
    'BLOCK_SIZE_COUT': power_of_two_range(4,6),
    'num_warps': power_of_two_range(2,5),
    'num_stages': [1, 2, 3, 4, 5]
}

# Fixed parameters for the kernel
# FIXED_GROUP_SIZE_M = 8

# Default matrix sizes for training and testing
# List of tuples (M, N, K)
# DEFAULT_TRAIN_SIZES = [
#     (256, 256, 256),   # Smaller power of 2
#     (512, 512, 512),
#     (1024, 1024, 1024),
#     (2048, 2048, 2048), # Larger power of 2
#     (768, 768, 768),   # Non-power of 2, square
#     (1536, 1536, 1536), # Non-power of 2, square, larger
#     (2048, 1024, 512), # Rectangular (as before)
#     (512, 2048, 1024), # Rectangular
#     (1024, 512, 2048), # Rectangular
#     (4096, 512, 1024), # Very rectangular, larger M
#     (512, 4096, 1024), # Very rectangular, larger N
#     (1024, 1024, 4096)  # Very rectangular, larger K
# ]

# RAND_SIZE_MIN = 128
# RAND_SIZE_MAX = 2048

def generate_random_conv_configs(n,
                                 N_min=1, N_max=16,  # batch sizes
                                 C_min=8, C_max=32, C_step=8,  # channel counts multiples of 8
                                 H_min=16, H_max=256, H_step=16,  # spatial dims multiples of 16
                                 kernel_choices=(1, 3, 5, 7),
                                 stride_choices=(1, 2)):
    """
    Generate n random convolution configurations as dicts:
    {
      "N": int,
      "C_in": int,
      "H": int,
      "W": int,
      "C_out": int,
      "KH": int,
      "KW": int,
      "stride": (stride_h, stride_w),
      "padding": (padding_h, padding_w)
    }
    - N: from N_min to N_max
    - C_in, C_out: multiples of C_step between C_min and C_max
    - H, W: multiples of H_step between H_min and H_max
    - KH, KW: chosen from kernel_choices (e.g., 1,3,5,7)
    - stride: chosen from stride_choices (same for h and w)
    - padding: floor(K/2) for 'same' padding
    """
    configs = []
    for _ in range(n):
        N = int(np.random.randint(N_min, N_max + 1))
        C_in = int((np.random.randint(C_min // C_step, C_max // C_step + 1)) * C_step)
        C_out = int((np.random.randint(C_min // C_step, C_max // C_step + 1)) * C_step)
        H = int((np.random.randint(H_min // H_step, H_max // H_step + 1)) * H_step)
        W = int((np.random.randint(H_min // H_step, H_max // H_step + 1)) * H_step)
        KH = int(np.random.choice(kernel_choices))
        KW = KH
        stride = int(np.random.choice(stride_choices))
        padding_h = 1 + KH // 2
        padding_w = 1 + KW // 2
        configs.append((N, C_in, H, W, C_out, KH, KW, (stride, stride), (padding_h, padding_w)))
    return configs

random_train_sizes = lambda n: generate_random_conv_configs(n)



DEFAULT_TEST_SIZES = [
    (1, 3, 224, 224, 64, 7, 7, (2, 2), (3, 3)),
    (4, 64, 56, 56, 128, 3, 3, (1, 1), (1, 1)),
    (8, 128, 28, 28, 256, 3, 3, (1, 1), (1, 1)),
    (1, 256, 14, 14, 512, 3, 3, (1, 1), (1, 1)),
    (1, 512, 7, 7, 512, 3, 3, (1, 1), (1, 1)),
]

random_test_sizes = lambda n: generate_random_conv_configs(n)


# RL Agent training parameters
PPO_N_STEPS = 64
PPO_BATCH_SIZE = 32
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.95
DEFAULT_TOTAL_TRAINING_TIMESTEPS_PER_SIZE = 1000 # Timesteps for each matrix size during training

# Reward weights
R_W_TFLOPS = 1.0
R_W_VRAM = 0.05

# Penalties
R_ERROR_PENALTY = -200.0
R_MAX_VRAM_PENALTY_VAL = 20000 # MB
