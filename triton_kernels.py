# triton_kernels.py
import triton
import triton.language as tl
import torch

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

def ceil_div(x, y):
    return (x + y - 1) // y

@triton.jit
def conv2d_nchw_kernel(
    # Pointers
    x_ptr, w_ptr, y_ptr,
    # Runtime dims (N, C_in, H, W, C_out)
    N, C_in, H, W, C_out,
    # Stride and padding are now constexpr so loops can use them at compile time
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    padding_h: tl.constexpr, padding_w: tl.constexpr,
    # Kernel size as constexpr
    KH: tl.constexpr, KW: tl.constexpr,
    # Computed output sizes (could be runtime but often derived)
    H_out, W_out,
    # Strides for pointer arithmetic (element strides)
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_co, stride_y_h, stride_y_w,
    # Tunable parameters
    BLOCK_SIZE_H: tl.constexpr,      # tile size on output height
    BLOCK_SIZE_W: tl.constexpr,      # tile size on output width
    BLOCK_SIZE_CIN: tl.constexpr,    # tile size on input channels
    BLOCK_SIZE_COUT: tl.constexpr,   # tile size on output channels
    num_warps: tl.constexpr,    # number of warps per program
    num_stages: tl.constexpr,   # pipeline depth
):
    """
    Conv2D forward in NCHW with fixed KH, KW, stride_h, stride_w, padding_h, padding_w as constexpr.
    Each program computes a tile [BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W].
    """

    pid = tl.program_id(axis=0)
    # Number of tiles along each dimension
    num_co_tiles = tl.cdiv(C_out, BLOCK_SIZE_COUT)
    num_h_tiles = tl.cdiv(H_out, BLOCK_SIZE_H)
    num_w_tiles = tl.cdiv(W_out, BLOCK_SIZE_W)
    tiles_per_batch = num_co_tiles * num_h_tiles * num_w_tiles

    # Map pid to (n, co_block, h_block, w_block)
    n = pid // tiles_per_batch
    rem = pid % tiles_per_batch
    co_block = rem // (num_h_tiles * num_w_tiles)
    rem2 = rem % (num_h_tiles * num_w_tiles)
    h_block = rem2 // num_w_tiles
    w_block = rem2 % num_w_tiles

    co_start = co_block * BLOCK_SIZE_COUT
    h_start = h_block * BLOCK_SIZE_H
    w_start = w_block * BLOCK_SIZE_W

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over input channels in chunks
    for ci_start in range(0, C_in, BLOCK_SIZE_CIN):
        # input-channel offsets
        offs_ci = ci_start + tl.arange(0, BLOCK_SIZE_CIN)  # [BLOCK_SIZE_CIN]
        mask_ci = offs_ci < C_in  # [BLOCK_SIZE_CIN]

        # Loop over kernel spatial dims: KH, KW are compile-time constants
        # so these loops are unrolled by Triton
        for kh in range(0, KH):
            for kw in range(0, KW):
                # Compute input spatial positions for each output tile coordinate
                offs_out_h = h_start + tl.arange(0, BLOCK_SIZE_H)  # [BLOCK_SIZE_H]
                in_h = offs_out_h * stride_h - padding_h + kh  # [BLOCK_SIZE_H]
                offs_out_w = w_start + tl.arange(0, BLOCK_SIZE_W)   # [BLOCK_SIZE_W]
                in_w = offs_out_w * stride_w - padding_w + kw  # [BLOCK_SIZE_W]

                mask_h = (in_h >= 0) & (in_h < H)   # [BLOCK_SIZE_H]
                mask_w = (in_w >= 0) & (in_w < W)   # [BLOCK_SIZE_W]
                mask_hw = mask_h[:, None] & mask_w[None, :]  # [BLOCK_SIZE_H, BLOCK_SIZE_W]

                # Build pointers for input: shape [BLOCK_SIZE_CIN, BLOCK_SIZE_H, BLOCK_SIZE_W]
                offs_ci_b = offs_ci[:, None, None]  # [BLOCK_SIZE_CIN,1,1]
                offs_h_b = in_h[None, :, None]     # [1,BLOCK_SIZE_H,1]
                offs_w_b = in_w[None, None, :]     # [1,1,BLOCK_SIZE_W]
                ptrs_x = x_ptr \
                    + n * stride_x_n \
                    + offs_ci_b * stride_x_c \
                    + offs_h_b * stride_x_h \
                    + offs_w_b * stride_x_w  # [BLOCK_SIZE_CIN, BLOCK_SIZE_H, BLOCK_SIZE_W]
                load_mask = mask_ci[:, None, None] & mask_hw[None, :, :]
                x_block = tl.load(ptrs_x, mask=load_mask, other=0.0)  # [BLOCK_SIZE_CIN, BLOCK_SIZE_H, BLOCK_SIZE_W]

                # Load weight: shape [BLOCK_SIZE_COUT, BLOCK_SIZE_CIN]
                offs_co = co_start + tl.arange(0, BLOCK_SIZE_COUT)  # [BLOCK_SIZE_COUT]
                offs_ic = offs_ci  # [BLOCK_SIZE_CIN]
                ptrs_w = w_ptr \
                    + (offs_co[:, None] * stride_w_co) \
                    + (offs_ic[None, :] * stride_w_ci) \
                    + kh * stride_w_kh \
                    + kw * stride_w_kw  # [BLOCK_SIZE_COUT, BLOCK_SIZE_CIN]
                mask_w = (offs_co < C_out)[:, None] & (offs_ic < C_in)[None, :]
                weights_block = tl.load(ptrs_w, mask=mask_w, other=0.0)  # [BLOCK_SIZE_COUT, BLOCK_SIZE_CIN]

                # Compute partial: reshape and dot
                x_flat = tl.reshape(x_block, (BLOCK_SIZE_CIN, BLOCK_SIZE_H * BLOCK_SIZE_W))  # [BLOCK_SIZE_CIN, H*W tile]
                acc_flat = tl.reshape(acc, (BLOCK_SIZE_COUT, BLOCK_SIZE_H * BLOCK_SIZE_W))  # [BLOCK_SIZE_COUT, H*W tile]
                acc_flat = tl.dot(weights_block, x_flat, acc_flat)  # [BLOCK_SIZE_COUT, H*W tile]
                acc = tl.reshape(acc_flat, (BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W))

    # Cast accumulator to FP16 for output
    y_block = acc.to(tl.float16)

    # Write back
    offs_co = co_start + tl.arange(0, BLOCK_SIZE_COUT)  # [BLOCK_SIZE_COUT]
    offs_out_h = h_start + tl.arange(0, BLOCK_SIZE_H)   # [BLOCK_SIZE_H]
    offs_out_w = w_start + tl.arange(0, BLOCK_SIZE_W)   # [BLOCK_SIZE_W]

    ptrs_y = y_ptr \
        + n * stride_y_n \
        + (offs_co[:, None, None] * stride_y_co) \
        + (offs_out_h[None, :, None] * stride_y_h) \
        + (offs_out_w[None, None, :] * stride_y_w)  # [BLOCK_SIZE_COUT, BLOCK_SIZE_H, BLOCK_SIZE_W]
    mask_write = (offs_co < C_out)[:, None, None] & \
                 (offs_out_h[None, :, None] < H_out) & \
                 (offs_out_w[None, None, :] < W_out)
    tl.store(ptrs_y, y_block, mask=mask_write)

def conv2d_nchw_triton(x, w, stride=(1, 1), padding=(0, 0)):
    """
    x: [N, C_in, H, W], w: [C_out, C_in, KH, KW], contiguous, on CUDA
    stride, padding: tuples
    """
    assert x.is_contiguous() and w.is_contiguous()
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = w.shape
    assert C_in == C_in_w
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    # Compute output spatial sizes
    H_out = (H + 2*padding_h - KH) // stride_h + 1
    W_out = (W + 2*padding_w - KW) // stride_w + 1
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float16)

    # Strides in element units
    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw = w.stride()
    stride_y_n, stride_y_co, stride_y_h, stride_y_w = y.stride()

    # Grid lambda uses META[...]
    def grid(META):
        bc = META['BLOCK_SIZE_COUT']
        bh = META['BLOCK_SIZE_H']
        bw = META['BLOCK_SIZE_W']
        num_co = ceil_div(C_out, bc)
        num_h = ceil_div(H_out, bh)
        num_w = ceil_div(W_out, bw)
        return (N * num_co * num_h * num_w,)

    conv2d_nchw_kernel[grid](
        x, w, y,
        # runtime dims
        N, C_in, H, W, C_out,
        # constexpr: stride_h, stride_w, padding_h, padding_w, KH, KW
        stride_h, stride_w, padding_h, padding_w, KH, KW,
        # computed output sizes
        H_out, W_out,
        # strides
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_y_n, stride_y_co, stride_y_h, stride_y_w,
        # Tunable params are provided by META
    )
    return y


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

def matmul_bench(M, N, K, A, B, C, config_dict):
    grid_fn = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    kernel_fn = lambda: rl_managed_matmul_kernel[grid_fn](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            **config_dict
        )
    runtime_ms = triton.testing.do_bench(kernel_fn, warmup=10, rep=50)
    return runtime_ms

def conv2d_bench(N, C_in, H, W, C_out, KH, KW, stride=(1, 1), padding=(0, 0), config_dict=None):
    """
    x: [N, C_in, H, W], w: [C_out, C_in, KH, KW], contiguous, on CUDA
    stride, padding: tuples
    """
    assert isinstance(config_dict, dict), "config_dict must be a dictionary of Triton config hyperparameters"
    #assert x.is_contiguous() and w.is_contiguous()
    # N, C_in, H, W = x.shape
    # C_out, C_in_w, KH, KW = w.shape
    #assert C_in == C_in_w
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    # Compute output spatial sizes
    H_out = (H + 2*padding_h - KH) // stride_h + 1
    W_out = (W + 2*padding_w - KW) // stride_w + 1

    x = torch.randn((N, C_in, H, W),
                                 dtype=torch.float16, device="cuda")
    w = torch.randn((C_out, C_in, KH, KW),dtype=torch.float16,device="cuda")
    y = torch.empty((N, C_out, H_out, W_out), device="cuda", dtype=torch.float16)
    # Strides in element units
    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw = w.stride()
    stride_y_n, stride_y_co, stride_y_h, stride_y_w = y.stride()

    # Grid lambda uses META[...]
    def grid(META):
        bc = META['BLOCK_SIZE_COUT']
        bh = META['BLOCK_SIZE_H']
        bw = META['BLOCK_SIZE_W']
        num_co = ceil_div(C_out, bc)
        num_h = ceil_div(H_out, bh)
        num_w = ceil_div(W_out, bw)
        return (N * num_co * num_h * num_w,)
    # print("config dict:", config_dict)
    kernel_fn = lambda: conv2d_nchw_kernel[grid](
        x, w, y,
        # runtime dims
        N, C_in, H, W, C_out,
        # constexpr: stride_h, stride_w, padding_h, padding_w, KH, KW
        stride_h, stride_w, padding_h, padding_w, KH, KW,
        # computed output sizes
        H_out, W_out,
        # strides
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_y_n, stride_y_co, stride_y_h, stride_y_w,
        config_dict['BLOCK_SIZE_H'], config_dict['BLOCK_SIZE_W'], config_dict['BLOCK_SIZE_CIN'], config_dict['BLOCK_SIZE_COUT'],
        config_dict['num_warps'], config_dict['num_stages']
    )

    runtime_ms = triton.testing.do_bench(kernel_fn, warmup=10, rep=50)

    return runtime_ms