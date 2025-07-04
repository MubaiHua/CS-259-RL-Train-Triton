# autotuner_benchmark.py
import torch
import triton
import triton.language as tl
import itertools
import time

from triton_kernels import matmul_kernel_logic 
from config import ACTION_PARAM_NAMES

def benchmark_triton_autotuner(M, N, K, A_gpu, B_gpu, C_gpu,
                               action_choices_for_autotuner, # Specific choices for this benchmark
                               fixed_group_size_m,
                               device='cuda'):

    print(f"\n--- Running Triton Autotuner Benchmark for M={M},N={N},K={K} ---")

    if device == 'cpu':
        print("Cannot run Triton autotuner on CPU. Skipping.")
        return {
            "best_config": None, "tflops": 0, "vram_mb": 0, "runtime_ms": float('inf'),
            "error": "CUDA device not available"
        }

    # Use ACTION_PARAM_NAMES from config.py to ensure consistent ordering
    param_names_ordered = ACTION_PARAM_NAMES
    try:
        value_lists = [action_choices_for_autotuner[name] for name in param_names_ordered]
    except KeyError as e:
        print(f"Error: Parameter name {e} from ACTION_PARAM_NAMES not found in action_choices_for_autotuner.")
        print("Ensure AUTOTUNER_BENCHMARK_ACTION_CHOICES in config.py contains all keys from ACTION_PARAM_NAMES.")
        return { "best_config": None, "tflops": 0, "vram_mb": 0, "runtime_ms": float('inf'), "error": str(e) }


    autotune_configs = []
    for values_tuple in itertools.product(*value_lists):
        cfg_kwargs = {}
        for i, name in enumerate(param_names_ordered):
            cfg_kwargs[name] = values_tuple[i]
        cfg_kwargs['GROUP_SIZE_M'] = fixed_group_size_m

        # Ensure num_warps and num_stages are present in cfg_kwargs for the Config constructor
        if 'num_warps' not in cfg_kwargs or 'num_stages' not in cfg_kwargs:
            print(f"Error: 'num_warps' or 'num_stages' missing in generated cfg_kwargs: {cfg_kwargs}")
            print("Ensure they are part of ACTION_PARAM_NAMES and AUTOTUNER_BENCHMARK_ACTION_CHOICES.")
            # Skip this invalid config or handle error appropriately
            continue

        autotune_configs.append(
            triton.Config(kwargs=cfg_kwargs,
                          num_warps=cfg_kwargs['num_warps'],
                          num_stages=cfg_kwargs['num_stages'])
        )

    if not autotune_configs:
        print("Error: No valid configurations generated for Triton autotuner. Check ACTION_CHOICES and PARAM_NAMES.")
        return { "best_config": None, "tflops": 0, "vram_mb": 0, "runtime_ms": float('inf'), "error": "No valid autotuner configs" }

    print(f"Triton autotuner will evaluate {len(autotune_configs)} configurations.")

    # Key for autotuning, using distinct names for M, N, K arguments to the kernel
    key_for_autotune = ['M_key', 'N_key', 'K_key']

    @triton.autotune(configs=autotune_configs, key=key_for_autotune)
    @triton.jit
    def _autotuned_kernel_for_benchmark(
        A_param, B_param, C_param, M_key, N_key, K_key, # Kernel args for autotune key
        stride_am_param, stride_ak_param, stride_bk_param, stride_bn_param,
        stride_cm_param, stride_cn_param, # Stride args
        # Constexpr args that will be filled by autotuner from triton.Config.kwargs
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, num_warps: tl.constexpr, num_stages: tl.constexpr
    ):
         matmul_kernel_logic(A_param, B_param, C_param, M_key, N_key, K_key,
                             stride_am_param, stride_ak_param, stride_bk_param, stride_bn_param,
                             stride_cm_param, stride_cn_param,
                             BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                             GROUP_SIZE_M, num_warps, num_stages)

    grid_fn = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    runtime_ms_autotune = float('inf')
    vram_mb_autotune = 0
    tflops_autotune = 0
    best_config_dict_autotune = None
    error_message = None

    try:
        # First call: This triggers the autotuning process. Triton runs benchmarks for all `autotune_configs`
        # and determines `_autotuned_kernel_for_benchmark.best_config`.
        _autotuned_kernel_for_benchmark[grid_fn](
            A_gpu, B_gpu, C_gpu, M, N, K, # Actual M,N,K values for the autotune key
            A_gpu.stride(0), A_gpu.stride(1), B_gpu.stride(0), B_gpu.stride(1),
            C_gpu.stride(0), C_gpu.stride(1)
        )
        torch.cuda.synchronize() # Wait for autotuning to complete

        # Check if a best_config was actually found
        if _autotuned_kernel_for_benchmark.best_config is None:
            raise ValueError("Triton autotuner did not find a best configuration. All configs might have failed.")

        print(f"Triton autotuner selected best config: {_autotuned_kernel_for_benchmark.best_config.kwargs}")
        print(f"  with num_warps={_autotuned_kernel_for_benchmark.best_config.num_warps}, num_stages={_autotuned_kernel_for_benchmark.best_config.num_stages}")


        # Second call: Benchmark the kernel, which now implicitly uses its determined best configuration.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        runtime_ms_autotune = triton.testing.do_bench(lambda: _autotuned_kernel_for_benchmark[grid_fn](
            A_gpu, B_gpu, C_gpu, M, N, K, # Pass M,N,K again for the key
            A_gpu.stride(0), A_gpu.stride(1), B_gpu.stride(0), B_gpu.stride(1),
            C_gpu.stride(0), C_gpu.stride(1)
        ),
            warmup=10,
            rep=50
        )
        torch.cuda.synchronize()

        vram_bytes_autotune = torch.cuda.max_memory_allocated()
        vram_mb_autotune = vram_bytes_autotune / (1024 * 1024)

        best_config_obj = _autotuned_kernel_for_benchmark.best_config
        best_config_dict_autotune = best_config_obj.kwargs.copy()

        if runtime_ms_autotune < 1e-6: runtime_ms_autotune = 1e-6 # Avoid division by zero
        tflops_autotune = (2 * M * N * K) / (runtime_ms_autotune * 1e-3) / 1e12

    except Exception as e:
        print(f"Error during Triton Autotuner benchmark execution for M={M},N={N},K={K}: {e}")
        error_message = str(e)
        runtime_ms_autotune = float('inf')
        tflops_autotune = 0

    results = {
        "best_config": best_config_dict_autotune,
        "tflops": tflops_autotune,
        "vram_mb": vram_mb_autotune,
        "runtime_ms": runtime_ms_autotune,
        "error": error_message
    }

    if error_message:
        print(f"Triton Autotuner for M={M},N={N},K={K} finished with an error.")
    else:
        print("--- Triton Autotuner Benchmark Complete ---")
        print(f"Autotuner Best Config for M={M},N={N},K={K}: {results['best_config']}")
        print(f"Autotuner Best Metrics: TFLOPs={results['tflops']:.2f}, VRAM={results['vram_mb']:.2f}MB, Runtime={results['runtime_ms']:.4f}ms")

    return results
