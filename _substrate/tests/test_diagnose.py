import os, sys, json, subprocess, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import diagnose


class TestDiagnoseGolden(unittest.TestCase):
    """Characterization: lock the CURRENT (class, evidence) for every all-measured
    case and every partial case that already works today. These MUST stay
    byte-identical after the §5.3 vendor-profile + null-rule edit (NVIDIA invariant)."""

    def test_memory_bound_both_measured(self):
        self.assertEqual(
            diagnose.classify({"dram_pct": 80, "sm_pct": 30}),
            ("memory_bound", ["dram 80% high, sm 30% low"]),
        )

    def test_compute_bound_both_measured(self):
        self.assertEqual(
            diagnose.classify({"dram_pct": 30, "sm_pct": 85}),
            ("compute_bound", ["sm 85% high"]),
        )

    def test_overhead_bound_both_measured(self):
        self.assertEqual(
            diagnose.classify({"dram_pct": 20, "sm_pct": 20, "occupancy": 0.9}),
            ("overhead_bound", ["both utilizations low (dram 20%, sm 20%)"]),
        )

    def test_unknown_both_measured_no_dominant_signal(self):
        # measured-both, no branch fires -> the :.0f / occ-None evidence MUST be byte-identical
        self.assertEqual(
            diagnose.classify({"dram_pct": 80, "sm_pct": 60}),
            ("unknown", ["no dominant signal (dram 80%, sm 60%, occ None)"]),
        )

    def test_latency_occupancy_partial_occ_only(self):
        # already-working partial case: dram/sm null, occ low
        self.assertEqual(
            diagnose.classify({"occupancy": 0.2}),
            ("latency_occupancy", ["occupancy 0.20 < 0.40 (launch/occupancy limited)"]),
        )

    def test_unknown_all_null(self):
        self.assertEqual(
            diagnose.classify({}),
            ("unknown", ["no profiler metrics available"]),
        )


class TestDiagnoseNullRule(unittest.TestCase):
    """Decided spec §4.6 rule: a two-sided branch (memory/overhead) fires only when
    BOTH dram and sm are measured; single-signal branches (latency_occupancy on occ,
    compute_bound on sm) may fire alone. New cases."""

    def test_high_dram_unmeasured_sm_is_unknown(self):
        # KEYSTONE: current code coerces sm->0.0 and WRONGLY returns memory_bound.
        self.assertEqual(
            diagnose.classify({"dram_pct": 80, "sm_pct": None}),
            ("unknown", ["no dominant signal (insufficient measured metrics)"]),
        )

    def test_high_sm_unmeasured_dram_is_compute_bound(self):
        # Regression guard: current code already returns compute_bound (sm>=70 fires on
        # the sm-alone branch). Stays compute_bound after the edit.
        self.assertEqual(
            diagnose.classify({"dram_pct": None, "sm_pct": 80}),
            ("compute_bound", ["sm 80% high"]),
        )

    def test_partial_sm_low_occ_high_is_unknown(self):
        # current code coerces dram->0.0 and WRONGLY returns overhead_bound.
        self.assertEqual(
            diagnose.classify({"dram_pct": None, "sm_pct": 30, "occupancy": 0.6}),
            ("unknown", ["no dominant signal (insufficient measured metrics)"]),
        )


class TestDiagnoseCli(unittest.TestCase):
    """Integration: the --metrics - stdin path prints the canonical JSON envelope."""

    def _run(self, metrics):
        return subprocess.run(
            [sys.executable, os.path.join(SUB, "diagnose.py"), "--metrics", "-"],
            input=json.dumps(metrics), capture_output=True, text=True,
        )

    def test_cli_memory_bound_envelope(self):
        r = self._run({"dram_pct": 80, "sm_pct": 30})
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["bottleneck_class"], "memory_bound")
        self.assertEqual(out["evidence"], ["dram 80% high, sm 30% low"])

    def test_cli_high_dram_unmeasured_sm_is_unknown(self):
        r = self._run({"dram_pct": 80, "sm_pct": None})
        out = json.loads(r.stdout)
        self.assertEqual(out["bottleneck_class"], "unknown")


if __name__ == "__main__":
    unittest.main()
