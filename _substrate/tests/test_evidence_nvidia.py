import os, sys, json, subprocess, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import importlib
_ev = importlib.import_module('backends._evidence_nvidia')

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'ncu')
MAPPER = os.path.join(SUB, 'backends', '_evidence_nvidia.py')


def _csv(name):
    return os.path.join(FIXTURES, name)


def run_mapper(csv_name, source_backend='cuda', script=MAPPER):
    """Invoke a mapper/wrapper script via subprocess; return (rc, parsed_or_raw)."""
    proc = subprocess.run(
        [sys.executable, script, '--native', _csv(csv_name),
         '--source-backend', source_backend],
        capture_output=True, text=True)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"_raw_stdout": proc.stdout, "_stderr": proc.stderr}
    return proc.returncode, payload


class TestEvidenceNvidiaFull(unittest.TestCase):
    """Full 4-counter CSV -> correct canonical dict, units converted, all 4 in coverage."""

    def setUp(self):
        self.rc, self.payload = run_mapper('full4.csv', source_backend='cuda')

    def test_exit_zero_and_ok_true(self):
        self.assertEqual(self.rc, 0, msg=f"payload={self.payload}")
        self.assertIs(self.payload.get('ok'), True, msg=f"payload={self.payload}")

    def test_source_backend_passed_through(self):
        self.assertEqual(self.payload['source_backend'], 'cuda')

    def test_latency_ms_is_ns_div_1e6(self):
        # 410000 ns / 1e6 = 0.41 ms
        self.assertAlmostEqual(self.payload['metrics']['latency_ms'], 0.41, places=9)

    def test_dram_pct_is_read_plus_write_0_to_100(self):
        # 40.0 + 22.0 = 62.0, in 0-100 (NOT 0-1)
        self.assertAlmostEqual(self.payload['metrics']['dram_pct'], 62.0, places=9)

    def test_sm_pct_passes_through_0_to_100(self):
        self.assertAlmostEqual(self.payload['metrics']['sm_pct'], 48.0, places=9)

    def test_occupancy_is_warps_active_div_100_range_0_to_1(self):
        # THE error-prone line: 51.0 % -> 0.51 (0-1), never 51.0
        occ = self.payload['metrics']['occupancy']
        self.assertAlmostEqual(occ, 0.51, places=9)
        self.assertGreaterEqual(occ, 0.0)
        self.assertLessEqual(occ, 1.0)

    def test_vendor_tag_is_nvidia(self):
        self.assertEqual(self.payload['metrics']['_vendor'], 'nvidia')

    def test_coverage_lists_all_four_canonical_keys(self):
        self.assertEqual(
            sorted(self.payload['coverage']),
            sorted(['latency_ms', 'dram_pct', 'sm_pct', 'occupancy']))

    def test_backend_native_carries_unmapped_counters(self):
        # the l2 sector-hit-rate row is free-form backend_native, not a canonical key
        self.assertIn('lts__t_sector_hit_rate.pct',
                      self.payload['metrics']['backend_native'])


class TestEvidenceNvidiaNullRule(unittest.TestCase):
    """Missing dram counter -> dram_pct is JSON null AND absent from coverage (never 0.0)."""

    def setUp(self):
        self.rc, self.payload = run_mapper('missing_dram.csv', source_backend='cuda')

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

    def test_occupancy_still_divided_by_100(self):
        self.assertAlmostEqual(self.payload['metrics']['occupancy'], 0.88, places=9)


class TestEvidenceNvidiaMalformed(unittest.TestCase):
    """Empty / unparseable CSV -> exit 2, {ok: false}, JSON still printed (envelope)."""

    def test_empty_csv_exits_2_ok_false(self):
        rc, payload = run_mapper('empty.csv', source_backend='cuda')
        self.assertEqual(rc, 2, msg=f"payload={payload}")
        self.assertIs(payload.get('ok'), False, msg=f"payload={payload}")
        self.assertIn('error', payload)  # JSON still printed on stdout

    def test_garbage_csv_exits_2_ok_false(self):
        # a file that is not CSV-with-our-columns at all
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as fh:
            fh.write("this is not ncu output\nno header columns here\n")
            path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, MAPPER, '--native', path,
                 '--source-backend', 'cuda'],
                capture_output=True, text=True)
            payload = json.loads(proc.stdout)
            self.assertEqual(proc.returncode, 2, msg=f"payload={payload}")
            self.assertIs(payload.get('ok'), False)
        finally:
            os.unlink(path)


CUDA_WRAPPER = os.path.join(SUB, 'backends', 'cuda', 'to_evidence.py')
TRITON_WRAPPER = os.path.join(SUB, 'backends', 'triton', 'to_evidence.py')


class TestVendorCollapseWrappers(unittest.TestCase):
    """cuda/to_evidence.py and triton/to_evidence.py on the SAME csv -> identical
    metrics, differing ONLY in source_backend (spec 5.1 vendor-collapse)."""

    def _run_path_invoked(self, script):
        # NO --source-backend flag: the wrapper supplies its own id. This also proves
        # the path-invoked (no package context) import idiom works.
        proc = subprocess.run(
            [sys.executable, script, '--native', _csv('full4.csv')],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0,
                         msg=f"stderr={proc.stderr}\nstdout={proc.stdout}")
        return json.loads(proc.stdout)

    def test_cuda_wrapper_stamps_cuda(self):
        self.assertEqual(self._run_path_invoked(CUDA_WRAPPER)['source_backend'], 'cuda')

    def test_triton_wrapper_stamps_triton(self):
        self.assertEqual(self._run_path_invoked(TRITON_WRAPPER)['source_backend'], 'triton')

    def test_metrics_identical_except_source_backend(self):
        cuda = self._run_path_invoked(CUDA_WRAPPER)
        triton = self._run_path_invoked(TRITON_WRAPPER)
        # metrics + coverage byte-identical (both lower to PTX, same ncu mapping)
        self.assertEqual(cuda['metrics'], triton['metrics'])
        self.assertEqual(cuda['coverage'], triton['coverage'])
        # ONLY source_backend differs
        self.assertNotEqual(cuda['source_backend'], triton['source_backend'])
        cuda_norm = dict(cuda); triton_norm = dict(triton)
        cuda_norm.pop('source_backend'); triton_norm.pop('source_backend')
        self.assertEqual(cuda_norm, triton_norm)

    def test_explicit_flag_overrides_wrapper_default(self):
        # --source-backend, if passed, wins over the wrapper's baked-in id
        proc = subprocess.run(
            [sys.executable, CUDA_WRAPPER, '--native', _csv('full4.csv'),
             '--source-backend', 'triton'],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['source_backend'], 'triton')


if __name__ == '__main__':
    unittest.main()
