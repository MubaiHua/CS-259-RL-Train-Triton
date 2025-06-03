# test_rl_agent.py
import torch
import time
import os
import numpy as np
from stable_baselines3 import PPO

# Assuming rl_env.py, autotuner_benchmark.py and config.py are accessible
from rl_env import TritonMatmulEnv 
from autotuner_benchmark import benchmark_triton_autotuner
from config import DEFAULT_TEST_SIZES, ACTION_CHOICES, FIXED_GROUP_SIZE_M, AUTOTUNER_BENCHMARK_ACTION_CHOICES

def test_agent_vs_autotuner(model_path, test_sizes, render_mode=None):
    if not torch.cuda.is_available():
        print("CUDA is not available. Testing requires an NVIDIA GPU.")
        return

    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please train the agent first.")
        return

    print(f"Loading trained PPO model from {model_path}...")
    # We need to instantiate a dummy env first to load the model, 
    # then we'll reconfigure it for each test size.
    # Or, we can pass None for env if the model was saved with env parameters.
    # For PPO, it's generally safer to provide an env with matching action/obs spaces.
    dummy_M, dummy_N, dummy_K = test_sizes[0] if test_sizes else (512,512,512)
    dummy_env = TritonMatmulEnv(M=dummy_M, N=dummy_N, K=dummy_K, render_mode=None) # No render during dummy load
    
    try:
        model = PPO.load(model_path, env=dummy_env) # Provide env for space checks
    except Exception as e:
        print(f"Error loading model: {e}. Ensure the environment used for saving matches.")
        # Fallback: try loading without env, assuming policy architecture is independent
        try:
            print("Attempting to load model without providing an environment instance...")
            model = PPO.load(model_path) 
            print("Model loaded (without env instance during load).")
            # We will set the env later if needed, or rely on predict not needing it.
        except Exception as e2:
            print(f"Failed to load model even without env instance: {e2}")
            return

    print("Model loaded successfully.")

    # Test environment instance
    test_env = TritonMatmulEnv(M=dummy_M, N=dummy_N, K=dummy_K, render_mode=render_mode)

    print(f"\n--- Starting Testing for Sizes: {test_sizes} ---")

    for i, (M, N, K) in enumerate(test_sizes):
        print(f"\n--- Testing Size: M={M}, N={N}, K={K} (Test {i+1}/{len(test_sizes)}) ---")
        
        # Reconfigure environment for the current test size
        obs, info = test_env.reconfigure_size(M, N, K)
        # If model was loaded without env, or if you want to be explicit:
        # model.set_env(test_env) # This might be needed if predict relies on env properties

        # 1. Get RL Agent's Best Action and Performance
        print("--- RL Agent Prediction ---")
        rl_agent_best_tflops = 0
        rl_agent_best_config_for_size = None
        rl_agent_metrics_for_size = {}
        rl_agent_reward_for_size = -float('inf')

        try:
            # For discrete actions, predict directly gives the action array
            action_indices, _states = model.predict(obs, deterministic=True)
            
            # Simulate a step with this action to get metrics
            # The environment's step method will use its internal A_gpu, B_gpu, C_gpu
            _obs, reward, _terminated, _truncated, info = test_env.step(action_indices)
            
            rl_agent_best_config_for_size = info.get("config")
            rl_agent_metrics_for_size = info.get("metrics", {})
            rl_agent_reward_for_size = reward

            print(f"RL Agent Predicted Config: {rl_agent_best_config_for_size}")
            print(f"RL Agent Metrics: TFLOPs={rl_agent_metrics_for_size.get('tflops',0):.2f}, "
                  f"VRAM={rl_agent_metrics_for_size.get('vram_mb',0):.2f}MB, "
                  f"Runtime={rl_agent_metrics_for_size.get('runtime_ms',0):.4f}ms")
            print(f"RL Agent Reward: {rl_agent_reward_for_size:.4f}")

        except Exception as e:
            print(f"Error during RL agent prediction or evaluation for M={M},N={N},K={K}: {e}")
            import traceback
            traceback.print_exc()

        # 2. Get Triton Autotuner's Performance
        # The test_env already has A_gpu, B_gpu, C_gpu for the current size
        autotuner_results = benchmark_triton_autotuner(
            M, N, K, 
            test_env.A_gpu, test_env.B_gpu, test_env.C_gpu,
            action_choices_for_autotuner=AUTOTUNER_BENCHMARK_ACTION_CHOICES,
            device=test_env.device,
            fixed_group_size_m=FIXED_GROUP_SIZE_M
        )
        
        # 3. Print Comparison for the current size
        print(f"\n--- Comparison for M={M}, N={N}, K={K} ---")
        if rl_agent_best_config_for_size:
            print(f"RL Agent TFLOPs: {rl_agent_metrics_for_size.get('tflops',0):.2f}")
        else:
            print("RL Agent: No valid configuration found or error.")
        
        if autotuner_results and autotuner_results.get("best_config"):
            print(f"Triton Autotuner TFLOPs: {autotuner_results.get('tflops',0):.2f}")
        else:
            print("Triton Autotuner: No valid configuration found or error.")
        print("------------------------------------------")

    test_env.close()


if __name__ == "__main__":
    # To run testing:
    # python test_rl_agent.py
    
    # Ensure the model path matches where you saved your trained model
    trained_model_file = "ppo_triton_matmul_tuner_multisize.zip" 
    
    if not os.path.exists(trained_model_file):
         print(f"Model file '{trained_model_file}' not found. Please run 'train_rl_agent.py' first.")
    else:
        test_agent_vs_autotuner(
            model_path=trained_model_file,
            test_sizes=DEFAULT_TEST_SIZES,
            render_mode="human" # or None for less output
        )

