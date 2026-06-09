"""Minimal Triton launcher fixture exercising the triton/profile.sh proton contract.

Exposes the launcher ABI profile.sh expects:
  make_inputs()  -> tuple of args
  forward(*args) -> runs the @triton.jit kernel once
  BYTES / FLOPS  -> per-invocation roofline annotation (drives dram_pct / sm_pct)

Only imported on a CUDA+triton box (the GPU-tier integration test skips otherwise).
"""
import torch
import triton
import triton.language as tl

N = 1 << 20
BLOCK = 1024
BYTES = 3 * N * 4   # read x, read y, write out (fp32)
FLOPS = N           # one add per element
KERNEL_NAME = "add_kernel"


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)


def make_inputs():
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    o = torch.empty_like(x)
    return (x, y, o)


def forward(x, y, o):
    grid = (triton.cdiv(N, BLOCK),)
    add_kernel[grid](x, y, o, N, BLOCK=BLOCK)
    return o
