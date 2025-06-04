# config.py

def range_list(start, end, step=1):
    """Generates a list of integers from start to end with a specified step."""
    return list(range(start, end + 1, step))

def power_of_two_range(start_exp, end_exp):
    """Generates a list of powers of two from 2^start_exp to 2^end_exp."""
    return [2 ** i for i in range(start_exp, end_exp + 1)]

# Action choices for the RL agent and the discrete search space for Triton's autotuner
ACTION_CHOICES = {
    'BLOCK_SIZE_M': power_of_two_range(2,8),
    'BLOCK_SIZE_N': power_of_two_range(2,8),
    'BLOCK_SIZE_K': power_of_two_range(2,8),
    'num_warps': power_of_two_range(2,5),
    'num_stages': [1, 2, 3, 4, 5]
}

ACTION_PARAM_NAMES = ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'num_warps', 'num_stages']

AUTOTUNER_BENCHMARK_ACTION_CHOICES = {
    'BLOCK_SIZE_M': [32, 64, 128],
    'BLOCK_SIZE_N': [32, 64, 128],
    'BLOCK_SIZE_K': [32, 64],
    'num_warps': [4, 8],
    'num_stages': [2, 3, 4]
}

# Names of parameters in the order they appear in the MultiDiscrete action vector
# This order must be consistent.
ACTION_PARAM_NAMES = ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'num_warps', 'num_stages']

# Fixed parameters for the kernel
FIXED_GROUP_SIZE_M = 8

# Default matrix sizes for training and testing
# List of tuples (M, N, K)
DEFAULT_TRAIN_SIZES = [
    (256, 256, 256),   # Smaller power of 2
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048), # Larger power of 2
    (768, 768, 768),   # Non-power of 2, square
    (1536, 1536, 1536), # Non-power of 2, square, larger
    (2048, 1024, 512), # Rectangular (as before)
    (512, 2048, 1024), # Rectangular
    (1024, 512, 2048), # Rectangular
    (4096, 512, 1024), # Very rectangular, larger M
    (512, 4096, 1024), # Very rectangular, larger N
    (1024, 1024, 4096)  # Very rectangular, larger K
]

DEFAULT_TEST_SIZES = [
    (1024, 1024, 1024), # Keep one common with train for direct comparison
    (2048, 2048, 2048),
    (512, 2048, 1024),
    (3000, 1000, 2000), # Larger, non-power of 2 test case
    (4096, 4096, 4096)  # Even larger test case
]

# RL Agent training parameters
PPO_N_STEPS = 64
PPO_BATCH_SIZE = 32
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.95
DEFAULT_TOTAL_TRAINING_TIMESTEPS_PER_SIZE = 500 # Timesteps for each matrix size during training

# Reward weights
R_W_TFLOPS = 1.0
R_W_VRAM = 0.05

# Penalties
R_ERROR_PENALTY = -200.0
R_MAX_VRAM_PENALTY_VAL = 20000 # MB
