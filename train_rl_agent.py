# train_rl_agent.py
import torch
import time
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback # For custom logging if needed

# Assuming rl_env.py and config.py are accessible
from rl_env import TritonMatmulEnv
from conv_env import TritonConv2dEnv
from config import DEFAULT_TRAIN_SIZES, PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_EPOCHS, \
                   PPO_GAMMA, DEFAULT_TOTAL_TRAINING_TIMESTEPS_PER_SIZE
from config import random_train_sizes as random_train_sizes_matmul
from config_conv2d import random_train_sizes as random_train_sizes_conv2d

# (Optional) Custom Callback for more detailed logging during training
class TrainingProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.best_reward_overall = -float('inf')
        self.best_config_overall = None
        self.best_metrics_overall = None

    def _on_step(self) -> bool:
        # Access environment attributes
        # For VecEnv, use get_attr; for single env, access directly
        if hasattr(self.training_env, 'get_attr'): # VecEnv
            current_best_reward_env = self.training_env.get_attr("best_reward_current_size")[0]
            current_best_config_env = self.training_env.get_attr("best_config_current_size")[0]
            current_best_metrics_env = self.training_env.get_attr("best_metrics_current_size")[0]
            current_M = self.training_env.get_attr("M")[0]
            current_N = self.training_env.get_attr("N")[0]
            current_K = self.training_env.get_attr("K")[0]
        else: # Single Env
            current_best_reward_env = self.training_env.best_reward_current_size
            current_best_config_env = self.training_env.best_config_current_size
            current_best_metrics_env = self.training_env.best_metrics_current_size
            current_M = self.training_env.M
            current_N = self.training_env.N
            current_K = self.training_env.K

        # Log to TensorBoard (SB3 logger)
        self.logger.record("custom/current_env_best_reward", current_best_reward_env)
        if current_best_metrics_env and "tflops" in current_best_metrics_env:
            self.logger.record("custom/current_env_best_tflops", current_best_metrics_env["tflops"])

        # Track overall best across all sizes if desired (more complex logic needed if model is reused)
        # For simplicity, this callback logs per-step, assuming env resets bests per size.
        return True

def train_agent(train_sizes, total_timesteps_per_size, kernel_type, model_save_path="ppo_triton_tuner.zip",
                load_existing_model=False, existing_model_path=None, render_mode=None):

    if not torch.cuda.is_available():
        print("CUDA is not available. Training requires an NVIDIA GPU.")
        return

    print(f"Starting training for sizes: {train_sizes}")

    # Initialize environment - it will be reconfigured for each size
    # The first size in the list is used for initial setup
    if kernel_type == "matmul":
        initial_M, initial_N, initial_K = train_sizes[0]
        env = TritonMatmulEnv(M=initial_M, N=initial_N, K=initial_K, render_mode=render_mode)
    elif kernel_type == "conv2d":
        env = TritonConv2dEnv(
                N=1, C_in=3, H=32, W=32,
                C_out=8, KH=3, KW=3,
                stride=(1, 1), padding=(1, 1)
              )


    if load_existing_model and existing_model_path and os.path.exists(existing_model_path):
        print(f"Loading existing model from {existing_model_path}")
        model = PPO.load(existing_model_path, env=env)
        print("Model loaded.")
    else:
        print("Creating a new PPO model.")
        model = PPO("MlpPolicy", env, verbose=1, device="cpu",
                    n_steps=PPO_N_STEPS,
                    batch_size=PPO_BATCH_SIZE,
                    n_epochs=PPO_N_EPOCHS,
                    gamma=PPO_GAMMA,
                    tensorboard_log="./ppo_triton_tensorboard/"
                    )

    overall_best_reward_across_all_sizes = -float('inf')
    overall_best_config_details = {} # Store best config per size

    training_start_time = time.time()

    for i, config_tuple in enumerate(train_sizes):
        print(f"\n--- Training for size: {config_tuple} (Size {i+1}/{len(train_sizes)}) ---")

        # Reconfigure the environment for the new size and reset it
        env.reconfigure_size(*config_tuple)
        # If model is reused, its internal state is preserved.
        # SB3 PPO model's `set_env` can be used if you want to ensure the model is
        # strictly aware of the new env instance, though for PPO, just changing env's state
        # and calling learn again often works for fine-tuning.
        # model.set_env(env) # Recommended if env instance changes significantly or for other algos

        size_train_start_time = time.time()
        try:
            # The callback can be used for more detailed logging
            # callback = TrainingProgressCallback()
            model.learn(total_timesteps=total_timesteps_per_size,
                        reset_num_timesteps=False, # Continue global timestep count if model is reused
                        progress_bar=True
                        # callback=callback
                        )
        except Exception as e:
            print(f"An error occurred during model training for size {config_tuple}: {e}")
            import traceback
            traceback.print_exc()
            continue # Move to next size if error occurs

        size_train_end_time = time.time()
        print(f"Training for size {config_tuple} finished in {size_train_end_time - size_train_start_time:.2f} seconds.")
        env.render() # Show best for this size

        # Store the best result for this size
        if env.best_reward_current_size > -float('inf'): # Check if any valid config was found
             overall_best_config_details[config_tuple] = {
                "reward": env.best_reward_current_size,
                "config": env.best_config_current_size,
                "metrics": env.best_metrics_current_size
            }
        if env.best_reward_current_size > overall_best_reward_across_all_sizes:
            overall_best_reward_across_all_sizes = env.best_reward_current_size


    training_end_time = time.time()
    print(f"\n--- Overall RL Training Complete ---")
    print(f"Total training time: {training_end_time - training_start_time:.2f} seconds.")

    print("\nBest configurations found per size by RL agent:")
    for size_key, results in overall_best_config_details.items():
        print(f"  Size {size_key}: Reward={results['reward']:.4f}, TFLOPs={results['metrics'].get('tflops',0):.2f}")
        # print(f"    Config: {results['config']}")


    if model_save_path:
        print(f"\nSaving trained model to {model_save_path}...")
        model.save(model_save_path)
        print("Model saved.")

    env.close()
    return model_save_path # Return path for testing script




if __name__ == "__main__":
    # To run training:
    # python train_rl_agent.py
    import argparse
    parser = argparse.ArgumentParser(description="Train RL agent for Triton matmul tuning")
    parser.add_argument("--random-train", action="store_true",
                        help="Use random sizes instead of default training sizes")
    parser.add_argument("-n", "--num-random-sizes", type=int, default=100,
                        help="Number of random sizes to generate if --random is specified")
    parser.add_argument("--kernel-type", type=str, default="matmul",
                        choices=["matmul", "conv2d"],
                        help="Type of kernel to tune (default: matmul)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.kernel_type == "matmul":
        random_train_sizes = random_train_sizes_matmul(args.num_random_sizes)
    elif args.kernel_type == "conv2d":
        random_train_sizes = random_train_sizes_conv2d(args.num_random_sizes)
    else:
        raise ValueError(f"Unsupported kernel type: {args.kernel_type}. Choose 'matmul' or 'conv2d'.")

    train_sizes = random_train_sizes if args.random_train else DEFAULT_TRAIN_SIZES
    trained_model_path = train_agent(
        train_sizes=train_sizes,
        total_timesteps_per_size=DEFAULT_TOTAL_TRAINING_TIMESTEPS_PER_SIZE,
        kernel_type=args.kernel_type,
        model_save_path=f"ppo_triton_{args.kernel_type}_tuner_multisize.zip",
        load_existing_model=False, # Set to True to continue training
        render_mode="human" if args.verbose else None # or None for less output
    )
    print(f"\nTraining finished. Model saved at: {trained_model_path}")
    print("You can now run test_rl_agent.py with this model.")
