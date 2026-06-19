import os
import glob
import unittest

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

VALID_PORTABILITY = {'clean', 'vendor_locked', 'method_intrinsic'}
VALID_MATRIX = {True, False, 'partial'}


def _workflow_manifests():
    """Per-workflow manifest.yaml files (SoT since manifest consolidation)."""
    paths = []
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, '*', 'manifest.yaml'))):
        parent = os.path.basename(os.path.dirname(p))
        if parent.startswith('_'):
            continue
        paths.append(p)
    return paths


def _check_backend_block_shape(testcase, manifest, source_path):
    backend = manifest['backend']
    testcase.assertIsInstance(backend, dict, f"{source_path}: backend: must be a mapping")
    testcase.assertIn('supported', backend, f"{source_path}: backend.supported required")
    testcase.assertIsInstance(backend['supported'], list,
                              f"{source_path}: backend.supported must be a list")
    testcase.assertIn('portability', backend, f"{source_path}: backend.portability required")
    testcase.assertIn(backend['portability'], VALID_PORTABILITY,
                      f"{source_path}: backend.portability must be one of {VALID_PORTABILITY}")
    testcase.assertIn('matrix_eligible', backend,
                      f"{source_path}: backend.matrix_eligible required")
    testcase.assertIn(backend['matrix_eligible'], VALID_MATRIX,
                      f"{source_path}: backend.matrix_eligible must be true|false|'partial'")


@unittest.skipIf(yaml is None, "pyyaml not installed")
class TestManifestParseV11(unittest.TestCase):

    def test_all_workflow_manifests_parse_under_v11(self):
        manifests = _workflow_manifests()
        self.assertGreater(len(manifests), 0, "expected at least one workflow manifest.yaml")
        for p in manifests:
            with self.subTest(manifest=p):
                with open(p) as f:
                    data = yaml.safe_load(f)
                self.assertIsInstance(data, dict, f"{p}: top-level must be a mapping")
                self.assertIn('backend', data, f"{p}: backend: block required")
                _check_backend_block_shape(self, data, p)

    def test_v11_minimal_fixture_parses_and_validates(self):
        p = os.path.join(FIXTURE_DIR, 'manifest_v11_minimal.yaml')
        with open(p) as f:
            data = yaml.safe_load(f)
        self.assertIn('backend', data, "v1.1 fixture must populate backend:")
        _check_backend_block_shape(self, data, p)
        self.assertEqual(data['backend']['supported'], ['cuda'])
        self.assertEqual(data['backend']['default'], 'cuda')
        self.assertEqual(data['backend']['portability'], 'clean')
        self.assertTrue(data['backend']['matrix_eligible'])


if __name__ == '__main__':
    unittest.main()
