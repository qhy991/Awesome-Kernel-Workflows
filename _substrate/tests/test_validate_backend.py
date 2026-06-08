import os, sys, json, subprocess, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

VALIDATOR = os.path.join(SUB, 'backends', 'validate_backend.py')
FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def run_validator(driver_dir):
    """Shell out to validate_backend.py <dir>; return (returncode, parsed_stdout_json)."""
    proc = subprocess.run(
        [sys.executable, VALIDATOR, os.path.join(FIXTURES, driver_dir)],
        capture_output=True, text=True)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"_raw_stdout": proc.stdout, "_stderr": proc.stderr}
    return proc.returncode, payload


class TestValidateBackendGood(unittest.TestCase):
    def test_good_driver_exits_zero_and_ok_true(self):
        code, payload = run_validator('good')
        self.assertEqual(code, 0, msg=f"expected exit 0, got {code}; payload={payload}")
        self.assertEqual(payload.get('ok'), True, msg=f"payload={payload}")
        self.assertEqual(payload.get('errors'), [], msg=f"payload={payload}")

    def test_bottleneck_classes_may_list_unknown(self):
        # `unknown` is an allowed bottleneck_classes entry (ALLOWED_BCLASSES = the 4
        # meaningful classes | {"unknown"}); listing it must NOT be rejected.
        code, payload = run_validator('good_unknown_bclass')
        self.assertEqual(code, 0, msg=f"expected exit 0, got {code}; payload={payload}")
        self.assertEqual(payload.get('ok'), True, msg=f"payload={payload}")
        self.assertEqual(payload.get('errors'), [], msg=f"payload={payload}")


class TestValidateBackendBad(unittest.TestCase):
    def assert_bad(self, driver_dir, substring):
        code, payload = run_validator(driver_dir)
        self.assertNotEqual(code, 0, msg=f"expected non-zero exit; payload={payload}")
        self.assertEqual(payload.get('ok'), False, msg=f"payload={payload}")
        joined = " | ".join(payload.get('errors', []))
        self.assertIn(substring, joined,
                      msg=f"expected '{substring}' in errors; got: {joined}")

    def test_backend_id_mismatch(self):
        self.assert_bad('bad_backend_id', 'backend_id')

    def test_idiom_references_unknown_method(self):
        self.assert_bad('bad_idiom_method', 'not_a_real_method')

    def test_non_canonical_metric_key(self):
        self.assert_bad('bad_metric_key', 'gpu_temp')

    def test_bogus_bottleneck_class(self):
        self.assert_bad('bad_bottleneck_class', 'register_bound')

    def test_invoke_field_must_be_fixed_name(self):
        self.assert_bad('bad_invoke_compiler', 'compiler.invoke')

    def test_substrate_script_not_edited(self):
        self.assert_bad('driver_edits_substrate', 'must not contain a copy of _substrate/')

    def test_threshold_profile_required(self):
        self.assert_bad('missing_threshold_profile', 'threshold_profile')


class TestValidateBackendBadArgs(unittest.TestCase):
    def test_nonexistent_dir_exits_3(self):
        code, payload = run_validator('__no_such_dir__')
        self.assertEqual(code, 3, payload)
        self.assertEqual(payload.get('ok'), False)


if __name__ == '__main__':
    unittest.main()
