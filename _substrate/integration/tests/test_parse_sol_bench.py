# tests/test_parse_sol_bench.py
import json, math, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARSE = HERE.parent / "parse_sol_bench.py"

def _rec(status, sf):
    return json.dumps({"definition": "025", "workload": {"axes": {"batch_size": 1}},
                       "solution": "cand",
                       "evaluation": {"status": status,
                                      "performance": {"latency_ms": 1.0, "reference_latency_ms": sf,
                                                      "speedup_factor": sf}}})

class ParseSolTests(unittest.TestCase):
    def _run(self, lines, *, normalized=False, reduction=None):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "bench.jsonl"
            f.write_text("\n".join(lines) + "\n")
            out = Path(d) / "result.json"
            command = [sys.executable, str(PARSE), str(f)]
            if reduction is not None:
                contract = Path(d) / "contract.env"
                contract.write_text(f"aggregate_reduction={reduction}\n")
                command.extend(["--contract", str(contract)])
            if normalized:
                command.extend(["--out", str(out)])
            result = subprocess.run(command, capture_output=True, text=True)
            payload = json.loads(out.read_text()) if out.exists() else None
            return result, payload

    def test_geomean_of_passed(self):
        # two PASSED with speedup 2.0 and 8.0 -> geomean 4.0
        r, _ = self._run([_rec("PASSED", 2.0), _rec("PASSED", 8.0)])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stdout, r"SPEEDUP=4(?:\.0+)?\b")
        self.assertIn("WORKLOADS=2/2", r.stdout)
        self.assertIn("STATUS=PASS", r.stdout)

    def test_skips_nonjson_header_lines(self):
        r, _ = self._run(["Problem: 025", "Config: {...}", "⠼ Evaluating...", _rec("PASSED", 3.0)])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("WORKLOADS=1/1", r.stdout)

    def test_failed_workloads_counted(self):
        r, _ = self._run([_rec("PASSED", 2.0), _rec("RUNTIME_ERROR", 0.0)])
        self.assertEqual(r.returncode, 1)
        self.assertIn("WORKLOADS=1/2", r.stdout)
        self.assertIn("STATUS=FAIL", r.stdout)

    def test_all_failed_exits_1(self):
        r, _ = self._run([_rec("RUNTIME_ERROR", 0.0)])
        self.assertEqual(r.returncode, 1)
        self.assertIn("STATUS=FAIL", r.stdout)

    def test_writes_canonical_all_workload_measurement(self):
        r, payload = self._run(
            [_rec("PASSED", 2.0), _rec("PASSED", 8.0)],
            normalized=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(payload["schema_version"], 1)
        self.assertTrue(payload["compiled"])
        self.assertTrue(payload["correct"])
        self.assertEqual(payload["n_pass"], 2)
        self.assertEqual(payload["n_total"], 2)
        self.assertEqual(payload["aggregate_reduction"], "geomean")
        self.assertAlmostEqual(payload["speedup"], 4.0)

    def test_sum_reduction_uses_ratio_of_latency_sums(self):
        # Per-workload speedups are 2 and 8, but weighted SUM is (2+8)/(1+1)=5.
        r, payload = self._run(
            [_rec("PASSED", 2.0), _rec("PASSED", 8.0)],
            normalized=True,
            reduction="sum",
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stdout, r"SPEEDUP=5(?:\.0+)?\b")
        self.assertIn("REDUCTION=sum", r.stdout)
        self.assertEqual(payload["aggregate_reduction"], "sum")
        self.assertAlmostEqual(payload["speedup"], 5.0)
        self.assertNotIn("geomean_speedup", payload)

    def test_unknown_reduction_fails_closed(self):
        r, _ = self._run([_rec("PASSED", 2.0)], reduction="median")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unsupported aggregate_reduction", r.stderr)

    def test_sum_reduction_rejects_non_positive_latency(self):
        rec = json.loads(_rec("PASSED", 2.0))
        rec["evaluation"]["performance"]["latency_ms"] = 0.0
        r, _ = self._run([json.dumps(rec)], reduction="sum")
        self.assertEqual(r.returncode, 1)
        self.assertIn("STATUS=FAIL", r.stdout)
        self.assertIn("WORKLOADS=0/1", r.stdout)

if __name__ == "__main__":
    unittest.main()
