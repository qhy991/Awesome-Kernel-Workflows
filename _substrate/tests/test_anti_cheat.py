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

    def test_measured_speedup_separate_from_recorded(self):
        # O3: a valid compile+correct run that merely doesn't beat the reward
        # threshold (e.g. ~1.0x SMOKE, or a sub-path win losing on aggregate)
        # must still report its measured speedup, so consumers don't misread a
        # valid run as a 0/failure. recorded_speedup stays 0 (beam gate intact).
        no_win = {**GOOD_METRICS, "candidate_latency_ms": 1.1, "eager_latency_ms": 1.0,
                  "compile_latency_ms": 1.2, "claimed_speedup": 1.02}
        res = anti_cheat.evaluate("__global__ void k(){}", no_win)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reward"], 0)            # no real win over baselines
        self.assertEqual(res["recorded_speedup"], 0.0)  # beam/memory gate intact
        self.assertEqual(res["measured_speedup"], 1.02)  # but the measured value is preserved
        # A real win sets both.
        win = {**GOOD_METRICS, "claimed_speedup": 2.0}
        res_win = anti_cheat.evaluate("__global__ void k(){}", win)
        self.assertEqual(res_win["recorded_speedup"], 2.0)
        self.assertEqual(res_win["measured_speedup"], 2.0)
        # An invalid run zeros both.
        res_bad = anti_cheat.evaluate("cublasSgemm(h);", GOOD_METRICS)
        self.assertEqual(res_bad["recorded_speedup"], 0.0)
        self.assertEqual(res_bad["measured_speedup"], 0.0)

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


class TestLoadVendorPatterns(unittest.TestCase):
    """Direct unit tests for anti_cheat.load_vendor_patterns."""

    def _write_temp(self, text):
        fd, path = tempfile.mkstemp(suffix=".patterns")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
        except Exception:
            os.unlink(path)
            raise
        return path

    def test_well_formed_file(self):
        # Uses METAL_PATTERNS which has 3 fallback lines and 3 skip lines.
        path = self._write_temp(METAL_PATTERNS)
        try:
            fallback, skip = anti_cheat.load_vendor_patterns(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(fallback), 3)
        self.assertEqual(len(skip), 3)
        # First fallback line has an explicit label via "| label" syntax.
        pat0, label0 = fallback[0]
        self.assertEqual(pat0, "MPSMatrixMultiplication")
        self.assertEqual(label0, "MPS matmul fallback")

    def test_no_pipe_label_uses_pattern_as_label(self):
        # "MPSNDArray" has no "|" -> label == pattern
        path = self._write_temp(METAL_PATTERNS)
        try:
            fallback, skip = anti_cheat.load_vendor_patterns(path)
        finally:
            os.unlink(path)
        pat1, label1 = fallback[1]
        self.assertEqual(pat1, "MPSNDArray")
        self.assertEqual(label1, "MPSNDArray")

    def test_line_before_any_section_is_dropped(self):
        text = (
            "orphan_regex\n"
            "[fallback]\n"
            r"\btorch\.matmul\b" + " | torch matmul\n"
        )
        path = self._write_temp(text)
        try:
            fallback, skip = anti_cheat.load_vendor_patterns(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(fallback), 1)
        self.assertEqual(len(skip), 0)

    def test_unknown_section_lines_dropped(self):
        text = (
            "[fallback]\n"
            r"\btorch\.matmul\b" + "\n"
            "[unknown]\n"
            "some_pattern\n"
        )
        path = self._write_temp(text)
        try:
            fallback, skip = anti_cheat.load_vendor_patterns(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(fallback), 1)
        self.assertEqual(len(skip), 0)

    def test_invalid_regex_raises_value_error(self):
        text = (
            "[fallback]\n"
            "[unclosed\n"
        )
        path = self._write_temp(text)
        try:
            with self.assertRaises(ValueError):
                anti_cheat.load_vendor_patterns(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
