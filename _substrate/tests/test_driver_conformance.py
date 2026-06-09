import os, sys, json, subprocess, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import method_gate  # live table — single source of truth for real method names

VALIDATOR = os.path.join(SUB, 'backends', 'validate_backend.py')
BACKENDS = os.path.join(SUB, 'backends')

# All real method_gate method names, sourced live (can never drift from the table).
KNOWN_METHODS = {m for methods in method_gate.TABLE.values() for m in methods}

# The drivers this part introduces. Both run on NVIDIA hardware (hw_vendor "nvidia") and
# share the nvidia diagnose.py thresholds. cuda is profiled by ncu; triton is profiled by
# Proton (triton.profiler) — ncu can't target Triton's mangled JIT symbols without elevated
# perf-counter access — but both lower their evidence onto the same canonical metric keys.
REAL_DRIVERS = ['cuda', 'triton']


def run_validator(driver_dir_abspath):
    """Shell out to validate_backend.py <dir>; return (returncode, parsed_stdout_json)."""
    proc = subprocess.run(
        [sys.executable, VALIDATOR, driver_dir_abspath],
        capture_output=True, text=True)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"_raw_stdout": proc.stdout, "_stderr": proc.stderr}
    return proc.returncode, payload


def load_driver_json(driver, fname):
    with open(os.path.join(BACKENDS, driver, fname), encoding="utf-8") as fh:
        return json.load(fh)


class TestRealDriversValidate(unittest.TestCase):
    def test_each_real_driver_exits_zero_ok_true_no_errors(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                code, payload = run_validator(os.path.join(BACKENDS, driver))
                self.assertEqual(code, 0,
                                 msg=f"{driver}: expected exit 0, got {code}; payload={payload}")
                self.assertEqual(payload.get('ok'), True, msg=f"{driver}: payload={payload}")
                self.assertEqual(payload.get('errors'), [], msg=f"{driver}: payload={payload}")


class TestRealDriversMethodNames(unittest.TestCase):
    def test_every_idiom_method_is_a_real_method_gate_name(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                idioms = load_driver_json(driver, 'idioms.json')
                methods = idioms.get('methods', {})
                self.assertIsInstance(methods, dict, msg=f"{driver}: methods not an object")
                self.assertTrue(methods, msg=f"{driver}: methods is empty")
                for name in methods:
                    self.assertIn(name, KNOWN_METHODS,
                                  msg=f"{driver}: idiom method '{name}' not in method_gate.TABLE")
                for name in idioms.get('unsupported_methods', []):
                    self.assertIn(name, KNOWN_METHODS,
                                  msg=f"{driver}: unsupported '{name}' not in method_gate.TABLE")

    def test_idioms_cover_a_gated_method_for_each_meaningful_class(self):
        # Each manifest lists the 4 meaningful classes; idioms.json should reference at least
        # one real gated method from each of those four classes (so the prompt layer always
        # has a concrete idiom to surface whatever bottleneck the gate picks).
        meaningful = ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"]
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                named = set(load_driver_json(driver, 'idioms.json').get('methods', {}))
                for bclass in meaningful:
                    gated = set(method_gate.TABLE[bclass])
                    self.assertTrue(named & gated,
                                    msg=f"{driver}: no idiom covers any method of '{bclass}' "
                                        f"(class methods={sorted(gated)})")


class TestRealDriversBackendId(unittest.TestCase):
    def test_backend_id_equals_directory_name(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                manifest = load_driver_json(driver, 'manifest.json')
                self.assertEqual(manifest.get('backend_id'), driver,
                                 msg=f"{driver}: manifest.backend_id != dir name")
                idioms = load_driver_json(driver, 'idioms.json')
                self.assertEqual(idioms.get('backend_id'), driver,
                                 msg=f"{driver}: idioms.backend_id != dir name")

    def test_both_drivers_are_nvidia_vendor_and_experimental(self):
        # cuda and triton both run on NVIDIA hardware (cuda profiled by ncu, triton by
        # Proton), so both share hw_vendor "nvidia" and the nvidia threshold profile; both
        # are honestly marked status "experimental".
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                manifest = load_driver_json(driver, 'manifest.json')
                self.assertEqual(manifest.get('hw_vendor'), 'nvidia', msg=f"{driver}")
                self.assertEqual(manifest.get('threshold_profile'), 'nvidia', msg=f"{driver}")
                self.assertEqual(manifest.get('status'), 'experimental', msg=f"{driver}")


class TestRealDriversL0Extended(unittest.TestCase):
    # Re-validation of the P3 cuda + triton drivers under the P5a Task 6 invariants
    # (invoke fixed-names, threshold_profile required, substrate-shadow guard).
    def test_real_drivers_pass_extended_l0(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                code, payload = run_validator(os.path.join(BACKENDS, driver))
                self.assertEqual(code, 0,
                                 msg=f"{driver}: expected exit 0, got {code}; payload={payload}")
                self.assertEqual(payload.get('ok'), True, msg=f"{driver}: payload={payload}")
                self.assertEqual(payload.get('errors'), [], msg=f"{driver}: payload={payload}")

    def test_real_drivers_use_fixed_invoke_names(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                m = load_driver_json(driver, 'manifest.json')
                self.assertEqual(m.get('compiler', {}).get('invoke'), 'build.sh')
                self.assertEqual(m.get('runner', {}).get('invoke'), 'run.sh')
                self.assertEqual(m.get('profiler', {}).get('invoke'), 'profile.sh')
                self.assertEqual(m.get('profiler', {}).get('to_evidence'), 'to_evidence.py')

    def test_real_drivers_declare_threshold_profile(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                m = load_driver_json(driver, 'manifest.json')
                self.assertIn('threshold_profile', m,
                              msg=f"{driver}: threshold_profile required (§4.4)")

    def test_real_drivers_do_not_shadow_substrate_scripts(self):
        substrate_scripts = {'evidence_schema.py', 'anti_cheat.py', 'diagnose.py',
                             'method_gate.py', 'memory_store.py', 'verify_insight.py'}
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                files = set(os.listdir(os.path.join(BACKENDS, driver)))
                shadow = files & substrate_scripts
                self.assertFalse(shadow,
                                 msg=f"{driver}: shadows substrate scripts {sorted(shadow)}")


if __name__ == '__main__':
    unittest.main()
