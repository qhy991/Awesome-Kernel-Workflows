import os, sys, json, stat, subprocess, tempfile, textwrap, unittest

SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

BACKENDS = os.path.join(SUB, 'backends')
CUDA = os.path.join(BACKENDS, 'cuda')
TRITON = os.path.join(BACKENDS, 'triton')


def _write_exec(path, body):
    """Write body to path and chmod 0755 (used for fake-tool stubs)."""
    with open(path, 'w') as fh:
        fh.write(body)
    os.chmod(path, 0o755)


def _run(argv, env=None, cwd=None):
    """Run argv; return (returncode, stdout, stderr)."""
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=cwd)
    return proc.returncode, proc.stdout, proc.stderr


def _json_or_raw(out):
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"_raw": out}


def _path_env(stub_dir):
    """Copy os.environ with stub_dir prepended to PATH."""
    env = dict(os.environ)
    env['PATH'] = stub_dir + os.pathsep + env.get('PATH', '')
    return env


# ----- fake-tool stub bodies (bash) -----
FAKE_NVCC_OK = textwrap.dedent('''\
    #!/usr/bin/env bash
    # Fake nvcc: parse for an output token (-o <file>) and touch it, then exit 0.
    out=""
    prev=""
    for a in "$@"; do
      if [ "$prev" = "-o" ]; then out="$a"; fi
      prev="$a"
    done
    [ -n "$out" ] && : > "$out"
    echo "fake nvcc ok" 1>&2
    exit 0
''')

FAKE_NVCC_FAIL = textwrap.dedent('''\
    #!/usr/bin/env bash
    echo "kernel.cu(7): error: identifier \\"foo\\" is undefined" 1>&2
    echo "1 error detected in the compilation of kernel.cu" 1>&2
    exit 1
''')

# Fake ncu: emit a canned NCU --csv profile on stdout (profile.sh captures stdout).
# NOTE: the fake's CSV deliberately includes BOTH dram read AND write rows so that the
# shared mapper's dram_pct = read + write contract (40.0 + 22.0 = 62.0) is exercised
# consistently with the Task-1 fixtures.
FAKE_NCU_CSV = textwrap.dedent('''\
    #!/usr/bin/env bash
    cat <<'CSV'
    "ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
    "0","my_kernel","gpu__time_duration.sum","ns","123456"
    "0","my_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","48.0"
    "0","my_kernel","dram__bytes_read.sum.pct_of_peak_sustained_elapsed","%","40.0"
    "0","my_kernel","dram__bytes_write.sum.pct_of_peak_sustained_elapsed","%","22.0"
    "0","my_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","51.0"
    CSV
    echo "fake ncu ok" 1>&2
    exit 0
''')


class TestSharedNvidiaMapper(unittest.TestCase):
    def test_shared_mapper_and_cuda_wrapper_exist(self):
        self.assertTrue(os.path.isfile(os.path.join(BACKENDS, '_evidence_nvidia.py')),
                        "_evidence_nvidia.py shared mapper missing (Tasks 1-2)")
        self.assertTrue(os.path.isfile(os.path.join(CUDA, 'to_evidence.py')),
                        "cuda/to_evidence.py wrapper missing (Tasks 1-2)")

    def test_mapper_occupancy_is_warps_active_over_100(self):
        # The single most error-prone line: occupancy = warps_active_pct / 100 (0..1).
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, 'native.csv')
            _write_exec(os.path.join(td, '_emit.sh'), FAKE_NCU_CSV)
            with open(csv, 'w') as fh:
                subprocess.run([os.path.join(td, '_emit.sh')], stdout=fh)
            code, out, err = _run(
                [sys.executable, os.path.join(CUDA, 'to_evidence.py'),
                 '--native', csv, '--format', 'ncu-csv'])
            self.assertEqual(code, 0, msg=f"out={out} err={err}")
            payload = _json_or_raw(out)
            self.assertEqual(payload.get('ok'), True, payload)
            m = payload['metrics']
            self.assertEqual(payload['source_backend'], 'cuda', payload)
            self.assertEqual(m['_vendor'], 'nvidia', payload)
            self.assertAlmostEqual(m['latency_ms'], 123456 / 1e6, places=9)
            self.assertAlmostEqual(m['sm_pct'], 48.0, places=3)
            self.assertAlmostEqual(m['dram_pct'], 62.0, places=3)   # read + write (canonical)
            self.assertAlmostEqual(m['occupancy'], 0.51, places=4)  # 51.0 / 100
            self.assertIn('occupancy', payload['coverage'])
