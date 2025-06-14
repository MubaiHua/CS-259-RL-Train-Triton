# test_rl_agent.py
import torch
import time
import os
import json
import numpy as np
from stable_baselines3 import PPO

from rl_env import TritonMatmulEnv
from autotuner_benchmark import benchmark_triton_autotuner
from config import DEFAULT_TEST_SIZES, ACTION_CHOICES, FIXED_GROUP_SIZE_M, AUTOTUNER_BENCHMARK_ACTION_CHOICES, random_test_sizes

def test_agent_vs_autotuner(model_path, test_sizes, render_mode=None):
    if not torch.cuda.is_available():
        print("CUDA is not available. Testing requires an NVIDIA GPU.")
        return

    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please train the agent first.")
        return

    print(f"Loading trained PPO model from {model_path}...")

    dummy_env = TritonMatmulEnv(train_sizes=test_sizes, render_mode=None)

    try:
        model = PPO.load(model_path, env=dummy_env)
    except Exception as e:
        print(f"Error loading model: {e}. Ensure the environment used for saving matches.")
        return

    print("Model loaded successfully.")

    test_env = TritonMatmulEnv(train_sizes=test_sizes, render_mode=render_mode)


    print(f"\n--- Starting Testing for Sizes: {test_sizes} ---")
    comparision_results = {"test_sizes": test_sizes, "num_test_sizes":len(test_sizes), "results": []}
    for i, (M, N, K) in enumerate(test_sizes):
        size_result = {"size": (M, N, K), "rl_agent": {}, "autotuner": {}}
        print(f"\n--- Testing Size: M={M}, N={N}, K={K} (Test {i+1}/{len(test_sizes)}) ---")

        obs, info = test_env.set_test_size(M, N, K)

        print("--- RL Agent Prediction ---")
        rl_agent_metrics_for_size = {}
        
        try:
            action_indices, _states = model.predict(obs, deterministic=True)
            # The environment's step method will use the size set by set_test_size
            _obs, reward, _terminated, _truncated, info = test_env.step(action_indices)

            rl_config = info.get("config", {})
            rl_metrics = info.get("metrics", {})
            
            print(f"RL Agent Predicted Config: {rl_config}")
            print(f"RL Agent Metrics: TFLOPs={rl_metrics.get('tflops', 0):.2f}, "
                  f"VRAM={rl_metrics.get('vram_mb', 0):.2f}MB, "
                  f"Runtime={rl_metrics.get('runtime_ms', 0):.4f}ms")
            print(f"RL Agent Reward: {reward:.4f}")

            size_result['rl_agent'] = {
                "best_config": rl_config,
                "tflops": rl_metrics.get('tflops', 0),
                "runtime_ms": rl_metrics.get('runtime_ms', 0),
                "vram_mb": rl_metrics.get('vram_mb', 0),
                "reward": reward
            }

        except Exception as e:
            print(f"Error during RL agent prediction or evaluation for M={M},N={N},K={K}: {e}")
            import traceback
            traceback.print_exc()

        # Get Triton Autotuner's Performance
        autotuner_results = benchmark_triton_autotuner(
            M, N, K,
            test_env.A_gpu, test_env.B_gpu, test_env.C_gpu,
            action_choices_for_autotuner=AUTOTUNER_BENCHMARK_ACTION_CHOICES,
            device=test_env.device,
            fixed_group_size_m=FIXED_GROUP_SIZE_M
        )


        size_result['autotuner'] = autotuner_results

        # Print Comparison
        print(f"\n--- Comparison for M={M}, N={N}, K={K} ---")
        print(f"RL Agent TFLOPs: {size_result['rl_agent'].get('tflops', 0):.2f}")
        print(f"Triton Autotuner TFLOPs: {autotuner_results.get('tflops', 0):.2f}")
        print("------------------------------------------")

        comparision_results["results"].append(size_result)
    test_env.close()

    results_file = "rl_vs_autotuner_comparison_results_vram.json"
    with open(results_file, 'w') as f:
        json.dump(comparision_results, f, indent=4)
    print(f"\n--- Testing Complete. Results saved to {results_file} ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test RL agent for Triton matmul tuning")
    parser.add_argument("--random-test", action="store_true",
                        help="Use random sizes instead of default test sizes")
    parser.add_argument("-n", "--num-random-sizes", type=int, default=100,
                        help="Number of random sizes to generate if --random is specified")
    args = parser.parse_args()

    test_sizes = random_test_sizes(args.num_random_sizes) if args.random_test else DEFAULT_TEST_SIZES

    trained_model_file = "ppo_triton_matmul_tuner_multisize.zip"
    if not os.path.exists(trained_model_file):
         print(f"Model file '{trained_model_file}' not found. Please run 'train_rl_agent.py' first.")
    else:
        test_agent_vs_autotuner(
            model_path=trained_model_file,
            test_sizes=test_sizes,
            render_mode="human"
        )