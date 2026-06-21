import os, sys, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, SUB)

import method_gate


class TestMethodGateConfirmIntent(unittest.TestCase):
    def test_unknown_includes_confirm_methods(self):
        out = method_gate.gate("unknown")
        self.assertIn("baseline_confirm", out["allowed_methods"])
        self.assertIn("noop_validate", out["allowed_methods"])

    def test_confirm_intent_prioritizes_confirm_methods(self):
        out = method_gate.gate("unknown", op_description="CONFIRM v5 baseline noop — do not optimize")
        self.assertEqual(out["allowed_methods"][:2], ["baseline_confirm", "noop_validate"])
        self.assertIn("confirm/noop intent", out["rationale"])

    def test_non_confirm_unknown_unchanged_order(self):
        out = method_gate.gate("unknown", op_description="optimize q8_0 gemm for 1.05x")
        self.assertEqual(out["allowed_methods"][0], "profile_first")
        self.assertNotIn("confirm/noop intent", out["rationale"])

    def test_memory_bound_unaffected_by_confirm_intent(self):
        out = method_gate.gate("memory_bound", op_description="confirm baseline")
        self.assertNotIn("baseline_confirm", out["allowed_methods"])
        self.assertIn("memory_coalescing", out["allowed_methods"])


if __name__ == "__main__":
    unittest.main()
