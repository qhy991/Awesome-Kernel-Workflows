import os
import glob
import unittest

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MANIFEST_DIR = os.path.join(REPO_ROOT, '_meta', 'manifests')
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

V10_REQUIRED_KEYS = {'source', 'workflow', 'method', 'topology', 'phases', 'args'}
VALID_PORTABILITY = {'clean', 'vendor_locked', 'method_intrinsic'}
VALID_MATRIX = {True, False, 'partial'}


def _existing_manifests():
    paths = []
    for p in sorted(glob.glob(os.path.join(MANIFEST_DIR, '*.yaml'))):
        if os.path.basename(p) == 'schema.yaml':
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

    def test_all_existing_manifests_parse_under_v11(self):
        manifests = _existing_manifests()
        self.assertGreater(len(manifests), 0, "expected at least one manifest")
        for p in manifests:
            with self.subTest(manifest=p):
                with open(p) as f:
                    data = yaml.safe_load(f)
                self.assertIsInstance(data, dict, f"{p}: top-level must be a mapping")
                missing = V10_REQUIRED_KEYS - set(data.keys())
                self.assertFalse(missing, f"{p}: missing v1.0 keys {missing}")
                if 'backend' in data:
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

    def test_accelopt_yaml_v10_still_parses_under_v11(self):
        p = os.path.join(MANIFEST_DIR, 'accelopt.yaml')
        self.assertTrue(os.path.exists(p), "AccelOpt manifest must exist as pinned fixture")
        with open(p) as f:
            data = yaml.safe_load(f)
        self.assertIsInstance(data, dict)
        for key in V10_REQUIRED_KEYS:
            self.assertIn(key, data, f"AccelOpt manifest missing v1.0 key {key}")
        self.assertNotIn('backend', data,
                         "AccelOpt pinned as legacy back-compat fixture; "
                         "no backend: block until its P5b retrofit lands")


if __name__ == '__main__':
    unittest.main()
