import os, sys, json, subprocess, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import importlib
_ev = importlib.import_module('backends._evidence_amd')

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'rocprof')
MAPPER = os.path.join(SUB, 'backends', '_evidence_amd.py')


def _csv(name):
    return os.path.join(FIXTURES, name)


def run_mapper(csv_name, source_backend='rocm', script=MAPPER):
    """Invoke the AMD mapper/wrapper via subprocess; return (rc, parsed_or_raw)."""
    proc = subprocess.run(
        [sys.executable, script, '--native', _csv(csv_name),
         '--source-backend', source_backend],
        capture_output=True, text=True)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"_raw_stdout": proc.stdout, "_stderr": proc.stderr}
    return proc.returncode, payload


class TestEvidenceAmdFull(unittest.TestCase):
    """Full rocprof CSV -> correct canonical dict, units converted, all 4 in coverage."""

    def setUp(self):
        self.rc, self.payload = run_mapper('full4.csv', source_backend='rocm')

    def test_exit_zero_and_ok_true(self):
        self.assertEqual(self.rc, 0, msg=f"payload={self.payload}")
        self.assertIs(self.payload.get('ok'), True, msg=f"payload={self.payload}")

    def test_source_backend_passed_through(self):
        self.assertEqual(self.payload['source_backend'], 'rocm')

    def test_latency_ms_is_ns_div_1e6(self):
        # KernelDuration 410000 ns / 1e6 = 0.41 ms
        self.assertAlmostEqual(self.payload['metrics']['latency_ms'], 0.41, places=9)

    def test_dram_pct_is_memunitbusy_passthrough(self):
        # AMD dram_pct = MemUnitBusy directly (a single 0-100 counter, NOT a read+write sum)
        self.assertAlmostEqual(self.payload['metrics']['dram_pct'], 62.0, places=9)

    def test_sm_pct_is_valubusy_0_to_100(self):
        # AMD sm_pct = VALUBusy (the closest analogue to NVIDIA SM throughput)
        self.assertAlmostEqual(self.payload['metrics']['sm_pct'], 48.0, places=9)

    def test_occupancy_is_wavefronts_over_max_range_0_to_1(self):
        # Wavefronts/MaxWavefronts = 51/100 = 0.51 (0-1), never 51.0
        occ = self.payload['metrics']['occupancy']
        self.assertAlmostEqual(occ, 0.51, places=9)
        self.assertGreaterEqual(occ, 0.0)
        self.assertLessEqual(occ, 1.0)

    def test_vendor_tag_is_amd(self):
        self.assertEqual(self.payload['metrics']['_vendor'], 'amd')

    def test_coverage_lists_all_four_canonical_keys(self):
        self.assertEqual(
            sorted(self.payload['coverage']),
            sorted(['latency_ms', 'dram_pct', 'sm_pct', 'occupancy']))

    def test_backend_native_carries_unmapped_counters(self):
        # L2CacheHit is free-form backend_native, not a canonical key
        self.assertIn('L2CacheHit',
                      self.payload['metrics']['backend_native'])

    def test_consumed_counters_absent_from_backend_native(self):
        # The five mapped counters must NOT leak back into backend_native
        bn = self.payload['metrics']['backend_native']
        for consumed in ('KernelDuration', 'VALUBusy', 'MemUnitBusy',
                         'Wavefronts', 'MaxWavefronts'):
            self.assertNotIn(consumed, bn)


class TestEvidenceAmdNullRule(unittest.TestCase):
    """Missing MemUnitBusy -> dram_pct is JSON null AND absent from coverage (never 0.0)."""

    def setUp(self):
        self.rc, self.payload = run_mapper('missing_membz.csv', source_backend='rocm')

    def test_exit_zero_ok_true(self):
        self.assertEqual(self.rc, 0, msg=f"payload={self.payload}")
        self.assertIs(self.payload.get('ok'), True)

    def test_dram_pct_is_json_null_not_zero(self):
        metrics = self.payload['metrics']
        self.assertIn('dram_pct', metrics)        # key present
        self.assertIsNone(metrics['dram_pct'])    # value is JSON null
        self.assertNotEqual(metrics['dram_pct'], 0.0)  # NEVER fabricated 0.0

    def test_dram_pct_absent_from_coverage(self):
        self.assertNotIn('dram_pct', self.payload['coverage'])

    def test_measured_keys_present_in_coverage(self):
        self.assertEqual(
            sorted(self.payload['coverage']),
            sorted(['latency_ms', 'sm_pct', 'occupancy']))

    def test_occupancy_still_ratio_0_to_1(self):
        # 88/100 = 0.88
        self.assertAlmostEqual(self.payload['metrics']['occupancy'], 0.88, places=9)


class TestEvidenceAmdMalformed(unittest.TestCase):
    """Empty / unparseable CSV -> exit 2, {ok: false}, JSON still printed (envelope)."""

    def test_empty_csv_exits_2_ok_false(self):
        rc, payload = run_mapper('empty.csv', source_backend='rocm')
        self.assertEqual(rc, 2, msg=f"payload={payload}")
        self.assertIs(payload.get('ok'), False, msg=f"payload={payload}")
        self.assertIn('error', payload)  # JSON still printed on stdout

    def test_garbage_csv_exits_2_ok_false(self):
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as fh:
            fh.write("this is not rocprof output\nno header columns here\n")
            path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, MAPPER, '--native', path,
                 '--source-backend', 'rocm'],
                capture_output=True, text=True)
            payload = json.loads(proc.stdout)
            self.assertEqual(proc.returncode, 2, msg=f"payload={payload}")
            self.assertIs(payload.get('ok'), False)
        finally:
            os.unlink(path)

    def test_non_utf8_file_exits_2_clean_json_no_traceback(self):
        """A binary/non-UTF-8 file must yield a clean JSON envelope (exit 2, ok=False)
        with NO traceback on stdout (UnicodeDecodeError must not propagate)."""
        import tempfile
        with tempfile.NamedTemporaryFile('wb', suffix='.csv', delete=False) as fh:
            fh.write(b'\xff\xfe\x00\x01bad')
            path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, MAPPER, '--native', path,
                 '--source-backend', 'rocm'],
                capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0,
                                msg=f"expected non-zero exit; stdout={proc.stdout!r}")
            self.assertEqual(proc.returncode, 2,
                             msg=f"expected exit 2 (parse error); stdout={proc.stdout!r}")
            payload = json.loads(proc.stdout)
            self.assertIs(payload.get('ok'), False, msg=f"payload={payload}")
            self.assertNotIn('Traceback', proc.stdout,
                             msg="traceback leaked to stdout")
        finally:
            os.unlink(path)

    def test_missing_source_backend_exits_3(self):
        # No --source-backend and no wrapper default -> bad args (exit 3)
        proc = subprocess.run(
            [sys.executable, MAPPER, '--native', _csv('full4.csv')],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 3, msg=f"stdout={proc.stdout!r}")
        payload = json.loads(proc.stdout)
        self.assertIs(payload.get('ok'), False)


ROCM_WRAPPER = os.path.join(SUB, 'backends', 'rocm', 'to_evidence.py')


class TestRocmWrapper(unittest.TestCase):
    """rocm/to_evidence.py delegates to the shared AMD mapper and stamps its own id
    when no --source-backend flag is passed (path-invoked, no package context)."""

    def _run_path_invoked(self, script, csv_name='full4.csv'):
        proc = subprocess.run(
            [sys.executable, script, '--native', _csv(csv_name)],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0,
                         msg=f"stderr={proc.stderr}\nstdout={proc.stdout}")
        return json.loads(proc.stdout)

    def test_rocm_wrapper_stamps_rocm(self):
        self.assertEqual(self._run_path_invoked(ROCM_WRAPPER)['source_backend'], 'rocm')

    def test_rocm_wrapper_metrics_match_direct_mapper(self):
        # The wrapper must produce the same metrics/coverage as the direct mapper.
        direct = run_mapper('full4.csv', source_backend='rocm')[1]
        wrapped = self._run_path_invoked(ROCM_WRAPPER)
        self.assertEqual(wrapped['metrics'], direct['metrics'])
        self.assertEqual(wrapped['coverage'], direct['coverage'])

    def test_explicit_flag_overrides_wrapper_default(self):
        proc = subprocess.run(
            [sys.executable, ROCM_WRAPPER, '--native', _csv('full4.csv'),
             '--source-backend', 'rocm-mi300'],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['source_backend'], 'rocm-mi300')


class TestEvidenceAmdMultiKernel(unittest.TestCase):
    """Multi-kernel CSV -> first kernel wins (deterministic; documented in the module)."""

    def setUp(self):
        self.rc, self.payload = run_mapper('multi2.csv', source_backend='rocm')

    def test_exit_zero_ok_true(self):
        self.assertEqual(self.rc, 0, msg=f"payload={self.payload}")
        self.assertIs(self.payload.get('ok'), True)

    def test_first_kernel_wins_latency_not_second(self):
        """kernel_A KernelDuration=200000 ns -> 0.2 ms;
        kernel_B has 999999 ns -> ~1.0 ms. First wins -> kernel_A."""
        latency = self.payload['metrics']['latency_ms']
        self.assertAlmostEqual(latency, 0.2, places=9,
                               msg="expected kernel_A latency (first-wins); "
                                   f"got {latency} (looks like kernel_B?)")
        self.assertNotAlmostEqual(latency, 0.999999, places=3,
                                  msg="kernel_B latency leaked into result (first-wins broken)")


if __name__ == '__main__':
    unittest.main()
