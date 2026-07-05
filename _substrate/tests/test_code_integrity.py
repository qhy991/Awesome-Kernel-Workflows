import os, sys, json, subprocess, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import code_integrity as ci

# A real, complete CUDA kernel — must pass clean.
GOOD_CUDA = r"""
#include <cuda_runtime.h>
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"""

# A real, complete Triton kernel — must pass clean.
GOOD_TRITON = r"""
import triton, triton.language as tl
@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x * 2.0, mask=mask)
"""

# 018 case: model stopped mid-kernel — unbalanced braces, dangling opener.
TRUNCATED_CUDA = r"""
__global__ void rmsnorm(float* out, float* x, float* w, int N) {
    int row = blockIdx.x;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float xi = x[row * N + i];
        acc += xi * xi;
    // ran out of output budget here — never closes the for, the kernel, etc.
"""

# L2-054 case: a balanced skeleton whose body is empty.
STUB_CUDA = r"""
__global__ void rmsnorm(float* out, float* x, float* w, int N) {
}
"""

STUB_TRITON = "@triton.jit\ndef k(x_ptr, n):\n    pass\n"

PLACEHOLDER_SRC = "__global__ void k(float* x) {\n  // your code goes here\n  return;\n}\n"


class TestCodeIntegrityUnit(unittest.TestCase):
    def test_good_cuda_passes(self):
        res = ci.evaluate(GOOD_CUDA)
        self.assertTrue(res["valid"], res)
        self.assertEqual(res["flags"], [])

    def test_good_triton_passes(self):
        res = ci.evaluate(GOOD_TRITON)
        self.assertTrue(res["valid"], res)
        self.assertEqual(res["flags"], [])

    def test_truncated_cuda_flagged(self):
        res = ci.evaluate(TRUNCATED_CUDA)
        self.assertFalse(res["valid"])
        self.assertTrue(any(f["type"] == "truncated_code" for f in res["flags"]), res)

    def test_stub_cuda_empty_body_flagged(self):
        res = ci.evaluate(STUB_CUDA)
        self.assertFalse(res["valid"])
        self.assertTrue(any(f["type"] == "empty_body" for f in res["flags"]), res)

    def test_stub_triton_pass_body_flagged(self):
        res = ci.evaluate(STUB_TRITON)
        self.assertFalse(res["valid"])
        self.assertTrue(any(f["type"] == "empty_body" for f in res["flags"]), res)

    def test_placeholder_flagged_but_not_blocking(self):
        # A placeholder in an otherwise-complete kernel is a warning, not a block.
        src = "__global__ void k(float* x) {\n  x[0] = x[0] + 1.0f;\n  // your code goes here\n}\n"
        res = ci.evaluate(src)
        self.assertTrue(any(f["type"] == "placeholder" for f in res["flags"]))
        self.assertTrue(res["valid"], "placeholder alone must not block a complete kernel")

    def test_string_with_brace_does_not_confuse_balance(self):
        src = '__global__ void k() { const char* s = "}"; printf("%s", s); }\n'
        res = ci.evaluate(src)
        self.assertTrue(res["valid"], res)
        self.assertEqual(res["flags"], [])

    def test_min_lines_gate(self):
        # below_min_lines is a WARNING, not a block — a small but complete kernel
        # is still valid; min_lines only signals "suspiciously small for this op."
        res = ci.evaluate(GOOD_CUDA, min_lines=50)
        self.assertTrue(res["valid"])
        self.assertTrue(any(f["type"] == "below_min_lines" for f in res["flags"]))

    def test_empty_source_invalid(self):
        res = ci.evaluate("")
        self.assertFalse(res["valid"])
        self.assertTrue(any(f["type"] == "empty_body" for f in res["flags"]))


class TestCodeIntegrityCli(unittest.TestCase):
    def _run(self, src, *extra):
        return subprocess.run(
            [sys.executable, os.path.join(SUB, "code_integrity.py"), "--source-text", src, *extra],
            capture_output=True, text=True,
        )

    def test_cli_good_kernel_exit_zero(self):
        r = self._run(GOOD_CUDA)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertTrue(out["valid"])

    def test_cli_truncated_exit_nonzero(self):
        r = self._run(TRUNCATED_CUDA)
        self.assertEqual(r.returncode, 1, r.stderr)
        out = json.loads(r.stdout)
        self.assertFalse(out["valid"])
        self.assertIn("truncated_code", [f["type"] for f in out["flags"]])


if __name__ == "__main__":
    unittest.main()
