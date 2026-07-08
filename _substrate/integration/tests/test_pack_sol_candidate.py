# tests/test_pack_sol_candidate.py
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent / "pack_sol_candidate.py"

CONTRACT = """op=reduction
backend=cuda
kernel_language=cuda
task_name=025_rmsnorm_h4096
kernel_path=/abs/025/kernel.cu
"""

# A kernel that already carries a pybind run() (like the KerSor-Best 025 seed).
KERNEL_WITH_BINDING = '''#include <torch/extension.h>
torch::Tensor run(torch::Tensor x, torch::Tensor w, double eps) { return x; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
'''

class PackSolTests(unittest.TestCase):
    def _run(self, kernel_text, contract_text):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "kernel.cu").write_text(kernel_text)
            (d / "contract.env").write_text(contract_text)
            out = d / "solution.json"
            r = subprocess.run([sys.executable, str(PACK), "--kernel", str(d / "kernel.cu"),
                                "--contract", str(d / "contract.env"), "--out", str(out)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            return json.loads(out.read_text())

    def test_required_top_level_keys(self):
        sol = self._run(KERNEL_WITH_BINDING, CONTRACT)
        for k in ("name", "definition", "author", "spec", "sources", "description"):
            self.assertIn(k, sol)

    def test_definition_is_task_name(self):
        sol = self._run(KERNEL_WITH_BINDING, CONTRACT)
        self.assertEqual(sol["definition"], "025_rmsnorm_h4096")

    def test_spec_contract_fields(self):
        sol = self._run(KERNEL_WITH_BINDING, CONTRACT)
        self.assertEqual(sol["spec"]["binding"], "torch")
        self.assertEqual(sol["spec"]["languages"], ["cuda"])  # cuda backend
        self.assertTrue(sol["spec"]["entry_point"])  # non-empty
        self.assertIn("compile_options", sol["spec"])

    def test_kernel_with_binding_is_single_source(self):
        sol = self._run(KERNEL_WITH_BINDING, CONTRACT)
        paths = [s["path"] for s in sol["sources"]]
        self.assertIn("kernel.cu", paths)
        # kernel already has PYBIND11_MODULE -> no separate main.cpp binding shell
        self.assertNotIn("main.cpp", paths)
        self.assertTrue(any("PYBIND11_MODULE" in s["content"] for s in sol["sources"]))

    def test_bare_kernel_gets_binding_shell(self):
        bare = "__global__ void k(){}\n"  # no pybind
        sol = self._run(bare, CONTRACT)
        paths = [s["path"] for s in sol["sources"]]
        self.assertIn("main.cpp", paths)  # a binding shell was added

if __name__ == "__main__":
    unittest.main()