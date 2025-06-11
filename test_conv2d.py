import torch
import triton
import triton.language as tl
import torch.nn.functional as F



@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16,
                'BLOCK_SIZE_CIN': 32, 'BLOCK_SIZE_COUT': 32
            },
            num_warps=4, num_stages=2
        ),
        triton.Config(
            {
                'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 8,
                'BLOCK_SIZE_CIN': 32, 'BLOCK_SIZE_COUT': 16
            },
            num_warps=4, num_stages=3
        ),
        triton.Config(
            {
                'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 32,
                'BLOCK_SIZE_CIN': 16, 'BLOCK_SIZE_COUT': 32
            },
            num_warps=4, num_stages=2
        ),
        triton.Config(
            {
                'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 8,
                'BLOCK_SIZE_CIN': 64, 'BLOCK_SIZE_COUT': 64
            },
            num_warps=8, num_stages=3
        ),
        triton.Config(
            {
                'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32,
                'BLOCK_SIZE_CIN': 16, 'BLOCK_SIZE_COUT': 16
            },
            num_warps=8, num_stages=4
        ),
        # ... add/prune configs as appropriate ...
    ],
    key=['N', 'C_in', 'H', 'W', 'C_out',
         # Note: KH and KW are constexpr, but we keep them in key so autotuner re-runs if different values:
         'stride_h', 'stride_w', 'padding_h', 'padding_w', 'H_out', 'W_out', 'C_in', 'C_out'
         # We do NOT include KH, KW here because they are constexpr arguments: Triton autotuner
         # uses META['KH'], etc., but runtime key does not need them as python args once constexpr.
         ],
)



# Ensure conv2d_nchw_triton is defined in your session
def test_conv2d(shape_params, dtype=torch.float16, tol=5e-2):
    N, C_in, H, W, C_out, KH, KW, stride, padding = shape_params
    x = torch.randn((N, C_in, H, W), device='cuda', dtype=dtype)
    w = torch.randn((C_out, C_in, KH, KW), device='cuda', dtype=dtype)
    y_t = conv2d_nchw_triton(x, w, stride=stride, padding=padding)
    y_ref = F.conv2d(x, w, bias=None, stride=stride, padding=padding)
    diff = (y_t.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Test shape N={N},C_in={C_in},H={H},W={W},C_out={C_out},KH={KH},KW={KW},stride={stride},padding={padding}")
    print(f"  Max abs diff: {max_diff:.4f}, Mean abs diff: {mean_diff:.4f}")
    if max_diff < tol:
        print("  PASS: within tolerance")
    else:
        print("  FAIL: difference exceeds tolerance")
    return max_diff, mean_diff

# Define test cases
test_cases = [
    (16, 16, 64, 64, 8, 3, 3, (3, 3), (1, 1)),
    (1, 3, 32, 32, 8, 3, 3, (1, 1), (1, 1)),
    (2, 16, 64, 64, 32, 5, 5, (1, 1), (2, 2)),
    (4, 64, 128, 128, 64, 3, 3, (2, 2), (1, 1)),
    (1, 16, 31, 31, 16, 3, 3, (1, 1), (1, 1)),  # odd spatial
    (1, 8, 28, 28, 8, 1, 1, (1, 1), (0, 0)),    # 1x1 conv
]

for params in test_cases:
    test_conv2d(params)