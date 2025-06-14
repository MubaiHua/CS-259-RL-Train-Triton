# train_rl_agent.py
import torch
import time
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_env import TritonMatmulEnv
from conv_env import TritonConv2dEnv
from config import DEFAULT_TRAIN_SIZES, PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_EPOCHS, \
                   PPO_GAMMA, DEFAULT_TOTAL_TRAINING_TIMESTEPS_PER_SIZE
from config import random_train_sizes as random_train_sizes_matmul
from config_conv2d import random_train_sizes as random_train_sizes_conv2d

def train_agent(train_sizes, total_timesteps, kernel_type, model_save_path="ppo_triton_tuner.zip",
                load_existing_model=False, existing_model_path=None, render_mode=None):

    if not torch.cuda.is_available():
        print("CUDA is not available. Training requires an NVIDIA GPU.")
        return

    print(f"Starting training for {len(train_sizes)} sizes with a total of {total_timesteps} timesteps.")

    if kernel_type == "matmul":
        env = TritonMatmulEnv(train_sizes=train_sizes, render_mode=render_mode)
    elif kernel_type == "conv2d":
        env = TritonConv2dEnv(train_configs=train_sizes, render_mode=render_mode)

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

    if load_existing_model and existing_model_path and os.path.exists(existing_model_path):
        print(f"Loading existing model from {existing_model_path}")
        model = PPO.load(existing_model_path, env=env)
    else:
        print("Creating a new PPO model with exploration-focused hyperparameters.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            # === START MODIFICATION ===
            # Changed device from "cuda" to "cpu" as per the warning's advice
            device="cpu",
            # === END MODIFICATION ===
            policy_kwargs=policy_kwargs,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            learning_rate=1e-4,
            ent_coef=0.02,
            tensorboard_log="./ppo_triton_tensorboard/"
        )

    training_start_time = time.time()

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    training_end_time = time.time()
    print(f"\n--- Overall RL Training Complete ---")
    print(f"Total training time: {training_end_time - training_start_time:.2f} seconds.")

    if model_save_path:
        print(f"\nSaving trained model to {model_save_path}...")
        model.save(model_save_path)
        print("Model saved.")

    env.close()
    return model_save_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train RL agent for Triton tuning")
    parser.add_argument("--random-train", action="store_true", help="Use random sizes instead of default")
    parser.add_argument("-n", "--num-random-sizes", type=int, default=50, help="Number of random sizes")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps to train")
    parser.add_argument("--kernel-type", type=str, default="matmul", choices=["matmul", "conv2d"], help="Kernel to tune")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.kernel_type == "matmul":
        train_sizes = random_train_sizes_matmul(args.num_random_sizes) if args.random_train else DEFAULT_TRAIN_SIZES
    elif args.kernel_type == "conv2d":
        train_sizes = random_train_sizes_conv2d(args.num_random_sizes)
    else:
        raise ValueError("Unsupported kernel type")

    trained_model_path = train_agent(
        train_sizes=train_sizes,
        total_timesteps=args.timesteps,
        kernel_type=args.kernel_type,
        model_save_path=f"ppo_triton_{args.kernel_type}_tuner_multisize.zip",
        load_existing_model=False,
        render_mode="human" if args.verbose else None
    )
    print(f"\nTraining finished. Model saved at: {trained_model_path}")