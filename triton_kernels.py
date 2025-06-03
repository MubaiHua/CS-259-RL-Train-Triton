# triton_kernels.py
import triton
import triton.language as tl

# Adapted from https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
@triton.jit
def matmul_kernel_logic(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tunable parameters that will be constexpr
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m_actual)
    pid_n = (pid % num_pid_in_group) // group_size_m_actual

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am
    offs_ak = tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_bn
    offs_bk = tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk

    a_ptrs = A + (offs_am + offs_ak)
    b_ptrs = B + (offs_bn + offs_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_loop_var in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_k_mask = (tl.arange(0, BLOCK_SIZE_K)[None,:] + k_loop_var * BLOCK_SIZE_K) < K
        b_k_mask = (tl.arange(0, BLOCK_SIZE_K)[:,None] + k_loop_var * BLOCK_SIZE_K) < K

        a_load_mask = a_k_mask & ( (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:,None]) < M )
        b_load_mask = b_k_mask & ( (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None,:]) < N )

        a = tl.load(a_ptrs, mask=a_load_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_load_mask, other=0.0)
        
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_val = accumulator.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    c_ptrs = C + stride_cm * offs_cm + stride_cn * offs_cn
    c_mask = (offs_cm < M) & (offs_cn < N)
    tl.store(c_ptrs, c_val, mask=c_mask)

# Kernel to be used by the RL environment
@triton.jit
def rl_managed_matmul_kernel(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, num_warps: tl.constexpr, num_stages: tl.constexpr
):
    matmul_kernel_logic(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages)

