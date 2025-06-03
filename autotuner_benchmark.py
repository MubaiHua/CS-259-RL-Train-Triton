# autotuner_benchmark.py
import torch
import triton
import triton.language as tl
import itertools
import time

# Assuming triton_kernels.py and config.py are accessible
from triton_kernels import matmul_kernel_logic # Use the core logic
from config import ACTION_CHOICES, ACTION_PARAM_NAMES, FIXED_GROUP_SIZE_M

def benchmark_triton_autotuner(M, N, K, A_gpu, B_gpu, C_gpu, device='cuda'):
    print(f"\n--- Running Triton Autotuner Benchmark for M={M},N={N},K={K} ---")
    
    if device == 'cpu':
        print("Cannot run Triton autotuner on CPU. Skipping.")
        return {
            "best_config": None, "tflops": 0, "vram_mb": 0, "runtime_ms": float('inf'),
            "error": "CUDA device not available"
        }

    param_names_from_config_file = ACTION_PARAM_NAMES # Ensure this order is used
    value_lists = [ACTION_CHOICES[name] for name in param_names_from_config_file]
    
    autotune_configs = []
    for values_tuple in itertools.product(*value_lists):
        cfg_kwargs = {}
        for i, name in enumerate(param_names_from_config_file):
            cfg_kwargs[name] = values_tuple[i]
        cfg_kwargs['GROUP_SIZE_M'] = FIXED_GROUP_SIZE_M
        
        autotune_configs.append(
            triton.Config(kwargs=cfg_kwargs, 
                          num_warps=cfg_kwargs['num_warps'], 
                          num_stages=cfg_kwargs['num_stages'])
        )

    key_for_autotune = ['M_key', 'N_key', 'K_key'] 
    
    # Define the autotuned kernel locally to ensure it captures the current autotune_configs
    # This kernel needs to be defined each time benchmark_triton_autotuner is called
    # to ensure it gets the correct `autotune_configs` if they were to change (though they don't here).
    @triton.autotune(configs=autotune_configs, key=key_for_autotune)
    @triton.jit # Jit decorator is good practice even with autotune
    def _autotuned_kernel_for_benchmark(
        A_param, B_param, C_param, M_key, N_key, K_key, 
        stride_am_param, stride_ak_param, stride_bk_param, stride_bn_param, stride_cm_param, stride_cn_param,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, num_warps: tl.constexpr, num_stages: tl.constexpr
    ):
         matmul_kernel_logic(A_param, B_param, C_param, M_key, N_key, K_key, 
                             stride_am_param, stride_ak_param, stride_bk_param, stride_bn_param, 
                             stride_cm_param, stride_cn_param,
                             BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, 
                             GROUP_SIZE_M, num_warps, num_stages)

    grid_fn = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    try:
        # First call: triggers the autotuning process, finds and sets the best config internally.
        _autotuned_kernel_for_benchmark[grid_fn](
            A_gpu, B_gpu, C_gpu, M, N, K, # M,N,K are used as key values
            A_gpu.stride(0), A_gpu.stride(1), B_gpu.stride(0), B_gpu.stride(1), 
            C_gpu.stride(0), C_gpu.stride(1)
        )
        torch.cuda.synchronize()
        
        # Second call: benchmark the kernel now using its determined best configuration.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        runtime_ms_autotune = triton.testing.do_bench(lambda: _autotuned_kernel_for_benchmark[grid_fn](
            A_gpu, B_gpu, C_gpu, M, N, K,
            A_gpu.stride(0), A_gpu.stride(1), B_gpu.stride(0), B_gpu.stride(1), 
            C_gpu.stride(0), C_gpu.stride(1)
        ))
        torch.cuda.synchronize()
        
        vram_bytes_autotune = torch.cuda.max_memory_allocated()
        vram_mb_autotune = vram_bytes_autotune / (1024 * 1024)

        best_config_obj = _autotuned_kernel_for_benchmark.best_config
        best_config_dict_autotune = best_config_obj.kwargs.copy()
        
        if runtime_ms_autotune < 1e-6: runtime_ms_autotune = 1e-6
        tflops_autotune = (2 * M * N * K) / (runtime_ms_autotune * 1e-3) / 1e12

        results = {
            "best_config": best_config_dict_autotune,
            "tflops": tflops_autotune,
            "vram_mb": vram_mb_autotune,
            "runtime_ms": runtime_ms_autotune
        }
        print("--- Triton Autotuner Benchmark Complete ---")
        print(f"Autotuner Best Config for M={M},N={N},K={K}: {results['best_config']}")
        print(f"Autotuner Best Metrics: TFLOPs={results['tflops']:.2f}, VRAM={results['vram_mb']:.2f}MB, Runtime={results['runtime_ms']:.4f}ms")
    
    except Exception as e:
        print(f"Error during Triton Autotuner benchmark for M={M},N={N},K={K}: {e}")
        results = {
            "best_config": None, "tflops": 0, "vram_mb": 0, "runtime_ms": float('inf'),
            "error": str(e)
        }
    return results
