import os, sys, json, subprocess, tempfile, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import anti_cheat

# Metal-flavoured source: an MPS library fallback AND a C++ empty-return stub.
METAL_SRC = (
    "#import <MetalPerformanceShaders.h>\n"
    "MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] init];\n"
    "void stub() { return; }\n"
)

# A Metal vendor-patterns file body: [fallback] + [skip], one regex per line,
# optional "| label". Comments and blank lines ignored.
METAL_PATTERNS = (
    "# Metal vendor cheat patterns\n"
    "[fallback]\n"
    "MPSMatrixMultiplication | MPS matmul fallback\n"
    "MPSNDArray\n"
    "MPSNNGraph\n"
    "\n"
    "[skip]\n"
    "//\\s*TODO | C++ TODO placeholder\n"
    "return\\s*; | C++ empty return stub\n"
    "__builtin_unreachable\n"
)

GOOD_METRICS = {
    "compiled": True, "correct": True,
    "candidate_latency_ms": 0.5, "eager_latency_ms": 1.0,
    "compile_latency_ms": 0.8, "claimed_speedup": 2.0,
}


class TestAntiCheatGolden(unittest.TestCase):
    """Characterization: lock the CURRENT static_flags / evaluate behaviour. These
    MUST stay byte-identical after the --vendor-patterns-file edit (CUDA invariant)."""

    def test_flags_cublas(self):
        self.assertEqual(
            anti_cheat.static_flags("auto y = cublasSgemm(handle);"),
            [{"type": "library_fallback", "detail": "cuBLAS delegation"}],
        )

    def test_flags_torch_matmul(self):
        self.assertEqual(
            anti_cheat.static_flags("out = torch.matmul(a, b)"),
            [{"type": "library_fallback", "detail": "torch.matmul fallback"}],
        )

    def test_flags_empty_pass(self):
        self.assertEqual(
            anti_cheat.static_flags("def f():\n    pass\n"),
            [{"type": "skipped_compute", "detail": "empty pass body"}],
        )

    def test_flags_not_implemented(self):
        self.assertEqual(
            anti_cheat.static_flags("raise NotImplementedError"),
            [{"type": "skipped_compute", "detail": "NotImplementedError"}],
        )

    def test_clean_kernel_has_no_flags(self):
        self.assertEqual(
            anti_cheat.static_flags("__global__ void k(float* x){ x[threadIdx.x] *= 2.0f; }"),
            [],
        )

    def test_evaluate_clean_fast_kernel_valid(self):
        res = anti_cheat.evaluate("__global__ void k(){}", GOOD_METRICS)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reward"], 3)
        self.assertEqual(res["recorded_speedup"], 2.0)
        self.assertEqual(res["blocking_flags"], [])

    def test_evaluate_cublas_fallback_invalid(self):
        res = anti_cheat.evaluate("cublasSgemm(h);", GOOD_METRICS)
        self.assertFalse(res["valid"])
        self.assertEqual(res["recorded_speedup"], 0.0)
        self.assertEqual(res["blocking_flags"], ["library_fallback"])

    def test_metal_source_not_flagged_by_default(self):
        # Today the CUDA/Python defaults never match MPS or a C++ return; stub.
        self.assertEqual(anti_cheat.static_flags(METAL_SRC), [])


class TestVendorPatternsUnit(unittest.TestCase):
    """NEW (red on current code): static_flags must accept appended vendor patterns
    so the Metal source is flagged on BOTH the fallback and skip lists."""

    def test_metal_flagged_with_vendor_patterns(self):
        flags = anti_cheat.static_flags(
            METAL_SRC,
            extra_fallback=[("MPSMatrixMultiplication", "MPS matmul fallback")],
            extra_skip=[(r"return\s*;", "C++ empty return stub")],
        )
        types = sorted(f["type"] for f in flags)
        self.assertIn("library_fallback", types)
        self.assertIn("skipped_compute", types)


class TestVendorPatternsCli(unittest.TestCase):
    """NEW (red on current code): --vendor-patterns-file does not exist yet, so
    argparse rejects it (exit 2). After the edit the Metal source is invalid."""

    def test_cli_metal_invalid_with_patterns_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".patterns", delete=False) as pf:
            pf.write(METAL_PATTERNS)
            patterns_path = pf.name
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SUB, "anti_cheat.py"),
                 "--source-text", METAL_SRC,
                 "--vendor-patterns-file", patterns_path,
                 "--metrics", "-"],
                input=json.dumps(GOOD_METRICS), capture_output=True, text=True,
            )
        finally:
            os.unlink(patterns_path)
        # invalid -> exit 1; stdout JSON shows both blocking flag types.
        self.assertEqual(r.returncode, 1, r.stderr)
        out = json.loads(r.stdout)
        self.assertFalse(out["valid"])
        self.assertEqual(
            sorted(set(out["blocking_flags"])),
            ["library_fallback", "skipped_compute"],
        )


if __name__ == "__main__":
    unittest.main()
