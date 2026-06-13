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

# Non-NVIDIA experimental drivers. They do NOT share the nvidia hw_vendor / threshold profile,
# so they are excluded from the nvidia-specific assertions above, but they MUST still satisfy
# every vendor-agnostic L0/idiom invariant. rocm = AMD/HIP; ascend = Huawei NPU (Ascend C).
NONVIDIA_DRIVERS = ['rocm', 'ascend']


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


class TestNonNvidiaDriversConform(unittest.TestCase):
    """Vendor-agnostic conformance for the non-NVIDIA experimental drivers (rocm, ascend).

    These must pass every invariant the nvidia drivers do EXCEPT the hw_vendor=='nvidia' /
    threshold_profile=='nvidia' assertion (which is nvidia-specific by design)."""

    def test_each_nonnvidia_driver_passes_l0(self):
        for driver in NONVIDIA_DRIVERS:
            with self.subTest(driver=driver):
                code, payload = run_validator(os.path.join(BACKENDS, driver))
                self.assertEqual(code, 0, msg=f"{driver}: exit {code}; payload={payload}")
                self.assertEqual(payload.get('ok'), True, msg=f"{driver}: payload={payload}")
                self.assertEqual(payload.get('errors'), [], msg=f"{driver}: payload={payload}")

    def test_every_idiom_method_is_a_real_method_gate_name(self):
        for driver in NONVIDIA_DRIVERS:
            with self.subTest(driver=driver):
                idioms = load_driver_json(driver, 'idioms.json')
                methods = idioms.get('methods', {})
                self.assertTrue(methods, msg=f"{driver}: methods is empty")
                for name in methods:
                    self.assertIn(name, KNOWN_METHODS, msg=f"{driver}: idiom '{name}' unknown")
                for name in idioms.get('unsupported_methods', []):
                    self.assertIn(name, KNOWN_METHODS, msg=f"{driver}: unsupported '{name}' unknown")

    def test_idioms_cover_each_DECLARED_bottleneck_class(self):
        # Unlike the nvidia drivers (which declare all 4 classes), a non-nvidia driver may
        # declare a subset (ascend omits latency_occupancy — no occupancy counter). It must
        # cover a gated method for each class it ACTUALLY declares.
        for driver in NONVIDIA_DRIVERS:
            with self.subTest(driver=driver):
                manifest = load_driver_json(driver, 'manifest.json')
                declared = manifest.get('capabilities', {}).get('bottleneck_classes', [])
                self.assertTrue(declared, msg=f"{driver}: no bottleneck_classes declared")
                named = set(load_driver_json(driver, 'idioms.json').get('methods', {}))
                for bclass in declared:
                    if bclass == 'unknown':
                        continue
                    gated = set(method_gate.TABLE.get(bclass, []))
                    self.assertTrue(named & gated,
                                    msg=f"{driver}: no idiom covers declared class '{bclass}'")

    def test_backend_id_and_fixed_invoke_names(self):
        for driver in NONVIDIA_DRIVERS:
            with self.subTest(driver=driver):
                m = load_driver_json(driver, 'manifest.json')
                self.assertEqual(m.get('backend_id'), driver, msg=f"{driver}: backend_id != dir")
                self.assertEqual(load_driver_json(driver, 'idioms.json').get('backend_id'), driver)
                self.assertEqual(m.get('compiler', {}).get('invoke'), 'build.sh')
                self.assertEqual(m.get('runner', {}).get('invoke'), 'run.sh')
                self.assertEqual(m.get('profiler', {}).get('invoke'), 'profile.sh')
                self.assertEqual(m.get('profiler', {}).get('to_evidence'), 'to_evidence.py')
                self.assertIn('threshold_profile', m, msg=f"{driver}: threshold_profile required")

    def test_do_not_shadow_substrate_scripts(self):
        substrate_scripts = {'evidence_schema.py', 'anti_cheat.py', 'diagnose.py',
                             'method_gate.py', 'memory_store.py', 'verify_insight.py'}
        for driver in NONVIDIA_DRIVERS:
            with self.subTest(driver=driver):
                files = set(os.listdir(os.path.join(BACKENDS, driver)))
                self.assertFalse(files & substrate_scripts, msg=f"{driver}: shadows substrate")

    def test_ascend_threshold_profile_is_wired_into_diagnose(self):
        # ascend declares threshold_profile 'ascend'; that key MUST exist in diagnose.PROFILES
        # (and its to_evidence stamps _vendor='ascend'), else diagnose silently falls back to
        # nvidia thresholds. (rocm's 'amd_rdna' is a known pre-existing fallback, not asserted.)
        import diagnose  # noqa: E402
        m = load_driver_json('ascend', 'manifest.json')
        self.assertEqual(m.get('threshold_profile'), 'ascend')
        self.assertIn('ascend', diagnose.PROFILES,
                      msg="diagnose.PROFILES missing 'ascend' -> bottleneck thresholds fall back to nvidia")


if __name__ == '__main__':
    unittest.main()
