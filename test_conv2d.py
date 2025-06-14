# test_conv2d.py
import torch
import triton
import triton.language as tl
import time
import os
import json
import numpy as np
import itertools
from stable_baselines3 import PPO

# Assuming necessary files are accessible
from conv_env import TritonConv2dEnv
from triton_kernels import conv2d_nchw_kernel
from config_conv2d import (
    DEFAULT_TEST_SIZES,
    AUTOTUNER_BENCHMARK_ACTION_CHOICES,
    ACTION_PARAM_NAMES,
    random_test_sizes,
)


def benchmark_conv2d_autotuner(
    shape_params, action_choices_for_autotuner, device="cuda"
):
    """
    Benchmarks Triton's built-in autotuner for a given Conv2D shape.
    """
    (
        N,
        C_in,
        H,
        W,
        C_out,
        KH,
        KW,
        stride,
        padding,
    ) = shape_params
    stride_h, stride_w = stride
    padding_h, padding_w = padding

    print(
        f"\n--- Running Triton Autotuner Benchmark for Conv2D Shape: "
        f"N={N},C_in={C_in},H={H},W={W},C_out={C_out},K={KH}x{KW},S={stride},P={padding} ---"
    )

    if device == "cpu":
        return {"best_config": None, "tflops": 0, "vram_mb": 0, "runtime_ms": float('inf'), "error": "CUDA device not available"}

    param_names_ordered = ACTION_PARAM_NAMES
    value_lists = [action_choices_for_autotuner[name] for name in param_names_ordered]

    autotune_configs = [
        triton.Config(
            kwargs=dict(zip(param_names_ordered, values)),
            num_warps=values[param_names_ordered.index("num_warps")],
            num_stages=values[param_names_ordered.index("num_stages")],
        )
        for values in itertools.product(*value_lists)
    ]

    if not autotune_configs:
        return {"best_config": None, "tflops": 0, "vram_mb": 0, "runtime_ms": float('inf'), "error": "No valid autotuner configs"}

    print(f"Triton autotuner will evaluate {len(autotune_configs)} configurations.")

    x = torch.randn((N, C_in, H, W), device="cuda", dtype=torch.float16)
    w = torch.randn((C_out, C_in, KH, KW), device="cuda", dtype=torch.float16)
    H_out = (H + 2 * padding_h - KH) // stride_h + 1
    W_out = (W + 2 * padding_w - KW) // stride_w + 1
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float16)

    key_for_autotune = ['N', 'C_in', 'H', 'W', 'C_out', 'H_out', 'W_out']

    @triton.autotune(configs=autotune_configs, key=key_for_autotune)
    @triton.jit
    def _autotuned_conv2d_kernel(
        x_ptr, w_ptr, y_ptr, N, C_in, H, W, C_out,
        stride_h: tl.constexpr, stride_w: tl.constexpr,
        padding_h: tl.constexpr, padding_w: tl.constexpr,
        KH: tl.constexpr, KW: tl.constexpr,
        H_out, W_out,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_y_n, stride_y_co, stride_y_h, stride_y_w,
        BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
        BLOCK_SIZE_CIN: tl.constexpr, BLOCK_SIZE_COUT: tl.constexpr,
        num_warps: tl.constexpr, num_stages: tl.constexpr
    ):
        conv2d_nchw_kernel(
            x_ptr, w_ptr, y_ptr, N, C_in, H, W, C_out, stride_h, stride_w, padding_h, padding_w,
            KH, KW, H_out, W_out, stride_x_n, stride_x_c, stride_x_h, stride_x_w,
            stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
            stride_y_n, stride_y_co, stride_y_h, stride_y_w,
            BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_CIN, BLOCK_SIZE_COUT,
            num_warps, num_stages)

    def grid(META):
        num_co_tiles = triton.cdiv(C_out, META['BLOCK_SIZE_COUT'])
        num_h_tiles = triton.cdiv(H_out, META['BLOCK_SIZE_H'])
        num_w_tiles = triton.cdiv(W_out, META['BLOCK_SIZE_W'])
        return (N * num_co_tiles * num_h_tiles * num_w_tiles,)

    runtime_ms, vram_mb, tflops, best_config_dict, error_message = float('inf'), 0, 0, None, None

    try:
        _autotuned_conv2d_kernel[grid](
            x, w, y, N, C_in, H, W, C_out, stride_h, stride_w, padding_h, padding_w, KH, KW, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3))
        torch.cuda.synchronize()

        best_config_obj = _autotuned_conv2d_kernel.best_config
        best_config_dict = best_config_obj.kwargs.copy()
        
        runtime_ms = triton.testing.do_bench(lambda: _autotuned_conv2d_kernel[grid](
            x, w, y, N, C_in, H, W, C_out, stride_h, stride_w, padding_h, padding_w, KH, KW, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3)), warmup=10, rep=50)
        torch.cuda.synchronize()

        vram_bytes = torch.cuda.max_memory_allocated()
        vram_mb = vram_bytes / (1024 * 1024)
        total_ops = 2.0 * N * C_out * H_out * W_out * (C_in * KH * KW)
        tflops = total_ops / (runtime_ms * 1e-3) / 1e12 if runtime_ms > 1e-6 else 0

    except Exception as e:
        error_message = str(e)

    results = {"best_config": best_config_dict, "tflops": tflops, "vram_mb": vram_mb, "runtime_ms": runtime_ms, "error": error_message}
    print("--- Triton Autotuner Benchmark Complete ---")
    print(f"Autotuner Best Metrics: TFLOPs={results['tflops']:.2f}, VRAM={results['vram_mb']:.2f}MB, Runtime={results['runtime_ms']:.4f}ms")
    return results


def test_agent_vs_autotuner_conv2d(model_path, test_sizes, render_mode=None):
    if not torch.cuda.is_available():
        print("CUDA is not available. Testing requires an NVIDIA GPU.")
        return

    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please train the agent first.")
        return

    print(f"Loading trained PPO model from {model_path}...")
    
    # === START MODIFICATION ===
    # Initialize the dummy environment with the test_sizes list to match the new constructor
    dummy_env = TritonConv2dEnv(train_configs=test_sizes)
    # === END MODIFICATION ===
    
    model = PPO.load(model_path, env=dummy_env)
    print("Model loaded successfully.")

    # === START MODIFICATION ===
    # Initialize the test environment instance with the list of sizes
    test_env = TritonConv2dEnv(train_configs=test_sizes, render_mode=render_mode)
    # === END MODIFICATION ===


    print(f"\n--- Starting Conv2D Testing for {len(test_sizes)} Sizes ---")
    comparison_results = {"test_sizes": test_sizes, "num_test_sizes": len(test_sizes), "results": []}

    for i, shape_params in enumerate(test_sizes):
        size_result = {"size": shape_params, "rl_agent": {}, "autotuner": {}}
        print(f"\n--- Testing Shape: {shape_params} (Test {i+1}/{len(test_sizes)}) ---")

        # === START MODIFICATION ===
        # Reconfigure environment using the new dedicated testing method
        obs, info = test_env.set_test_shape(shape_params)
        # === END MODIFICATION ===

        print("--- RL Agent Prediction ---")
        try:
            action_indices, _ = model.predict(obs, deterministic=True)
            _obs, reward, _, _, info = test_env.step(action_indices)
            
            rl_config = info.get("config", {})
            rl_metrics = info.get("metrics", {})

            print(f"RL Agent Predicted Config: {rl_config}")
            print(f"RL Agent Metrics: TFLOPs={rl_metrics.get('tflops', 0):.2f}, "
                  f"VRAM={rl_metrics.get('vram_mb', 0):.2f}MB, "
                  f"Runtime={rl_metrics.get('runtime_ms', 0):.4f}ms")
            print(f"RL Agent Reward: {reward:.4f}")

            size_result['rl_agent'] = {"best_config": rl_config, **rl_metrics}

        except Exception as e:
            print(f"Error during RL agent evaluation for shape {shape_params}: {e}")
            import traceback
            traceback.print_exc()

        autotuner_results = benchmark_conv2d_autotuner(
            shape_params=shape_params,
            action_choices_for_autotuner=AUTOTUNER_BENCHMARK_ACTION_CHOICES,
            device=test_env.device,
        )
        size_result['autotuner'] = autotuner_results

        print(f"\n--- Comparison for Shape: {shape_params} ---")
        print(f"RL Agent TFLOPs: {size_result['rl_agent'].get('tflops', 0):.2f}")
        print(f"Triton Autotuner TFLOPs: {autotuner_results.get('tflops', 0):.2f}")
        print("------------------------------------------")

        comparison_results["results"].append(size_result)

    test_env.close()

    results_file = "rl_vs_autotuner_comparison_results_conv2d.json"
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    print(f"\n--- Testing Complete. Results saved to {results_file} ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test RL agent for Triton Conv2D tuning")
    parser.add_argument("--random-test", action="store_true",
                        help="Use random sizes instead of default test sizes")
    parser.add_argument("-n", "--num-random-sizes", type=int, default=10,
                        help="Number of random sizes to generate if --random-test is specified")
    args = parser.parse_args()

    test_sizes = random_test_sizes(args.num_random_sizes) if args.random_test else DEFAULT_TEST_SIZES

    trained_model_file = "ppo_triton_conv2d_tuner_multisize.zip"
    if not os.path.exists(trained_model_file):
        print(f"Model file '{trained_model_file}' not found. Please run 'train_rl_agent.py --kernel-type conv2d' first.")
    else:
        test_agent_vs_autotuner_conv2d(
            model_path=trained_model_file,
            test_sizes=test_sizes,
            render_mode="human"
        )