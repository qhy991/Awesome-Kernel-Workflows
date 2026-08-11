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
    def _run(self, kernel_text, contract_text, filename="kernel.cu"):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            kernel = d / filename
            kernel.write_text(kernel_text)
            (d / "contract.env").write_text(contract_text)
            out = d / "solution.json"
            r = subprocess.run([sys.executable, str(PACK), "--kernel", str(kernel),
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

    def test_missing_contract_uses_safe_metadata_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            kernel = d / "candidate.cu"
            out = d / "solution.json"
            kernel.write_text(KERNEL_WITH_BINDING)
            r = subprocess.run(
                [sys.executable, str(PACK), "--kernel", str(kernel),
                 "--contract", str(d / "missing-contract.env"), "--out", str(out)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            sol = json.loads(out.read_text())
            self.assertEqual(sol["definition"], "candidate")
            self.assertEqual(sol["spec"]["languages"], ["cuda_cpp"])
            self.assertEqual(sol["spec"]["entry_point"], "candidate.cu::run")

    def test_generic_python_staging_name_is_normalized_for_cuda_cpp(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            kernel = d / "kernel.py"
            out = d / "solution.json"
            kernel.write_text(KERNEL_WITH_BINDING)
            r = subprocess.run(
                [sys.executable, str(PACK), "--kernel", str(kernel),
                 "--contract", str(d / "missing-contract.env"), "--out", str(out)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            sol = json.loads(out.read_text())
            self.assertEqual(sol["spec"]["entry_point"], "kernel.cu::run")
            self.assertEqual(sol["sources"][0]["path"], "kernel.cu")

    def test_spec_contract_fields(self):
        sol = self._run(KERNEL_WITH_BINDING, CONTRACT)
        self.assertEqual(sol["spec"]["binding"], "torch")
        self.assertEqual(sol["spec"]["languages"], ["cuda_cpp"])  # sol-execbench enum
        self.assertTrue(sol["spec"]["entry_point"])  # non-empty
        self.assertIn("compile_options", sol["spec"])

    def test_kernel_with_binding_is_single_source(self):
        sol = self._run(KERNEL_WITH_BINDING, CONTRACT)
        paths = [s["path"] for s in sol["sources"]]
        self.assertIn("kernel.cu", paths)
        # kernel already has PYBIND11_MODULE -> no separate main.cpp binding shell
        self.assertNotIn("main.cpp", paths)
        self.assertTrue(any("PYBIND11_MODULE" in s["content"] for s in sol["sources"]))

    def test_bare_kernel_fails_loudly(self):
        bare = "__global__ void k(){}\n"  # no pybind
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "kernel.cu").write_text(bare)
            (d / "contract.env").write_text(CONTRACT)
            out = d / "solution.json"
            r = subprocess.run([sys.executable, str(PACK), "--kernel", str(d / "kernel.cu"),
                                "--contract", str(d / "contract.env"), "--out", str(out)],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertTrue(
                "PYBIND11_MODULE" in r.stderr or "binding" in r.stderr,
                f"Expected PYBIND11_MODULE or binding in stderr; got: {r.stderr!r}"
            )

    def test_triton_python_candidate_uses_native_python_transport(self):
        source = "import torch\nimport triton\nimport triton.language as tl\n\ndef run(a, b):\n    return torch.matmul(a, b.T)\n"
        sol = self._run(source, CONTRACT, "candidate.py")
        self.assertEqual(sol["spec"]["languages"], ["triton"])
        self.assertEqual(sol["spec"]["entry_point"], "candidate.py::run")
        self.assertNotIn("binding", sol["spec"])
        self.assertNotIn("compile_options", sol["spec"])

    def test_pytorch_candidate_uses_pytorch_transport(self):
        source = "import torch\n\ndef run(a, b):\n    return torch.matmul(a, b.T)\n"
        sol = self._run(source, CONTRACT, "candidate.py")
        self.assertEqual(sol["spec"]["languages"], ["pytorch"])
        self.assertEqual(sol["spec"]["destination_passing_style"], False)

    def test_python_candidate_without_module_run_fails_and_removes_stale_output(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            kernel = d / "candidate.py"
            contract = d / "contract.env"
            out = d / "solution.json"
            kernel.write_text("import torch\n\ndef helper():\n    pass\n")
            contract.write_text(CONTRACT)
            out.write_text('{"stale": true}\n')
            r = subprocess.run(
                [sys.executable, str(PACK), "--kernel", str(kernel),
                 "--contract", str(contract), "--out", str(out)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("module-level run", r.stderr)
            self.assertFalse(out.exists())

    def test_contract_comment_stripping(self):
        # contract.env values can carry trailing '#' comments (real 025 contract.env does)
        contract = "op=reduction\nbackend=cuda\ntask_name=025_rmsnorm_h4096  # trailing comment\nkernel_path=/abs/k.cu\n"
        sol = self._run(KERNEL_WITH_BINDING, contract)
        self.assertEqual(sol["definition"], "025_rmsnorm_h4096")  # comment stripped, no trailing spaces

if __name__ == "__main__":
    unittest.main()
