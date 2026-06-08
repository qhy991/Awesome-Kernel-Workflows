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


class TestCudaBuild(unittest.TestCase):
    SCRIPT = os.path.join(CUDA, 'build.sh')

    def test_exists_executable_and_syntax(self):
        self.assertTrue(os.path.isfile(self.SCRIPT), "cuda/build.sh missing")
        self.assertTrue(os.access(self.SCRIPT, os.X_OK), "cuda/build.sh not executable")
        code, _, err = _run(['bash', '-n', self.SCRIPT])
        self.assertEqual(code, 0, msg=f"bash -n failed: {err}")

    def test_build_ok_with_fake_nvcc(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'nvcc'), FAKE_NVCC_OK)
            src = os.path.join(td, 'kernel.cu')
            out = os.path.join(td, 'kernel.so')
            with open(src, 'w') as fh:
                fh.write("// fake cuda source\n")
            env = _path_env(td)
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out,
                                     '--arch', 'sm_80'], env=env)
            self.assertEqual(code, 0, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), True, p)
            self.assertEqual(p.get('compiled'), True, p)
            self.assertEqual(p.get('artifact'), out, p)
            self.assertTrue(os.path.isfile(out), "artifact not produced by fake nvcc")
            self.assertIn('build_latency_ms', p)
            self.assertIsInstance(p['build_latency_ms'], (int, float))
            self.assertIn('stderr_tail', p)

    def test_build_passes_lineinfo_to_nvcc(self):
        # -lineinfo is REQUIRED for ncu source attribution; assert the script emits it.
        with tempfile.TemporaryDirectory() as td:
            # fake nvcc that records its argv to a sidecar file
            rec = os.path.join(td, 'argv.txt')
            _write_exec(os.path.join(td, 'nvcc'), textwrap.dedent(f'''\
                #!/usr/bin/env bash
                echo "$@" > "{rec}"
                out=""; prev=""
                for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
                [ -n "$out" ] && : > "$out"
                exit 0
            '''))
            src = os.path.join(td, 'k.cu'); out = os.path.join(td, 'k.so')
            open(src, 'w').write("//\n")
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out],
                                    env=_path_env(td))
            self.assertEqual(code, 0, msg=f"{sout} {serr}")
            self.assertIn('-lineinfo', open(rec).read())

    def test_build_compile_failure_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'nvcc'), FAKE_NVCC_FAIL)
            src = os.path.join(td, 'kernel.cu'); out = os.path.join(td, 'kernel.so')
            open(src, 'w').write("//\n")
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out],
                                    env=_path_env(td))
            self.assertEqual(code, 2, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), False, p)
            self.assertEqual(p.get('compiled'), False, p)
            self.assertIsNone(p.get('artifact'), p)
            self.assertIn('error detected', p.get('stderr_tail', ''))

    def test_build_missing_nvcc_exit_3(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'kernel.cu'); out = os.path.join(td, 'kernel.so')
            open(src, 'w').write("//\n")
            # nvcc is genuinely absent on this macOS host, so the inherited env already exercises
            # the script's own "nvcc not found" guard (exit 3). Do NOT wipe PATH — that would break
            # the `#!/usr/bin/env bash` shebang itself (exit 127). On a GPU box where nvcc exists,
            # point PATH at a stub dir that has bash+coreutils but omits nvcc.
            env = dict(os.environ)
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out], env=env)
            self.assertEqual(code, 3, msg=f"out={sout} err={serr}")
            self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_build_missing_args_exit_3(self):
        code, sout, _ = _run([self.SCRIPT, '--source', '/x.cu'])  # no --out
        self.assertEqual(code, 3)
        self.assertEqual(_json_or_raw(sout).get('ok'), False)


class TestCudaRun(unittest.TestCase):
    SCRIPT = os.path.join(CUDA, 'run.sh')

    def _problem(self, td):
        p = os.path.join(td, 'problem.json')
        with open(p, 'w') as fh:
            json.dump({"op": "add", "shape": [128, 128]}, fh)
        return p

    def test_exists_executable_and_syntax(self):
        self.assertTrue(os.path.isfile(self.SCRIPT), "cuda/run.sh missing")
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))
        code, _, err = _run(['bash', '-n', self.SCRIPT])
        self.assertEqual(code, 0, msg=err)

    def test_missing_artifact_clean_error_envelope_exit_3(self):
        # No GPU needed: a nonexistent artifact is a preflight/bad-input failure -> clean JSON envelope, exit 3 (spec §4.5).
        with tempfile.TemporaryDirectory() as td:
            prob = self._problem(td); out = os.path.join(td, 'result.json')
            code, sout, serr = _run([self.SCRIPT, '--artifact', '/nope/x.so',
                                     '--problem', prob, '--out', out])
            self.assertEqual(code, 3, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), False, p)
            # Contract keys must still be present (anti_cheat reads these exactly).
            for k in ('compiled', 'correct', 'candidate_latency_ms', 'eager_latency_ms',
                      'compile_latency_ms', 'claimed_speedup'):
                self.assertIn(k, p, f"missing key {k}: {p}")
            self.assertEqual(p['correct'], False, p)
            self.assertLessEqual(p['claimed_speedup'], 1.0, p)  # correct:false ⇒ ≤1.0

    def test_missing_args_exit_3(self):
        code, sout, _ = _run([self.SCRIPT, '--artifact', '/x.so'])  # no --problem/--out
        self.assertEqual(code, 3)
        self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_bad_problem_file_exit_3(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'r.json')
            code, sout, _ = _run([self.SCRIPT, '--artifact', '/x.so',
                                  '--problem', '/no/problem.json', '--out', out])
            self.assertEqual(code, 3)
            self.assertEqual(_json_or_raw(sout).get('ok'), False)


class TestCudaProfile(unittest.TestCase):
    SCRIPT = os.path.join(CUDA, 'profile.sh')

    def _problem(self, td):
        p = os.path.join(td, 'problem.json')
        json.dump({"op": "add"}, open(p, 'w'))
        return p

    def test_exists_executable_and_syntax(self):
        self.assertTrue(os.path.isfile(self.SCRIPT), "cuda/profile.sh missing")
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))
        code, _, err = _run(['bash', '-n', self.SCRIPT])
        self.assertEqual(code, 0, msg=err)

    def test_profile_ok_with_fake_ncu_writes_csv_and_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'ncu'), FAKE_NCU_CSV)
            art = os.path.join(td, 'k.so'); open(art, 'w').write("")
            prob = self._problem(td); out = os.path.join(td, 'native.csv')
            code, sout, serr = _run([self.SCRIPT, '--artifact', art,
                                     '--problem', prob, '--out', out], env=_path_env(td))
            self.assertEqual(code, 0, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), True, p)
            self.assertEqual(p.get('profiler'), 'ncu', p)
            self.assertEqual(p.get('native_profile'), out, p)
            self.assertEqual(p.get('format'), 'ncu-csv', p)
            self.assertTrue(os.path.isfile(out), "csv not written")
            self.assertIn('sm__warps_active', open(out).read())

    def test_profile_requests_the_four_counters(self):
        with tempfile.TemporaryDirectory() as td:
            rec = os.path.join(td, 'argv.txt')
            _write_exec(os.path.join(td, 'ncu'), textwrap.dedent(f'''\
                #!/usr/bin/env bash
                echo "$@" > "{rec}"
                echo '"ID","Metric Name","Metric Value"' ; exit 0
            '''))
            art = os.path.join(td, 'k.so'); open(art, 'w').write("")
            prob = self._problem(td); out = os.path.join(td, 'n.csv')
            _run([self.SCRIPT, '--artifact', art, '--problem', prob, '--out', out],
                 env=_path_env(td))
            argv = open(rec).read()
            for c in ('gpu__time_duration.sum',
                      'sm__throughput.avg.pct_of_peak_sustained_elapsed',
                      'dram__bytes_read.sum.pct_of_peak_sustained_elapsed',
                      'sm__warps_active.avg.pct_of_peak_sustained_active'):
                self.assertIn(c, argv, f"profile.sh did not request {c}")

    def test_profiler_absent_exit_4(self):
        with tempfile.TemporaryDirectory() as td:
            art = os.path.join(td, 'k.so'); open(art, 'w').write("")
            prob = self._problem(td); out = os.path.join(td, 'n.csv')
            env = dict(os.environ)   # ncu genuinely absent on macOS; keep PATH so the shebang resolves (wiping it => exit 127, not the exit-4 guard)
            code, sout, serr = _run([self.SCRIPT, '--artifact', art,
                                     '--problem', prob, '--out', out], env=env)
            self.assertEqual(code, 4, msg=f"out={sout} err={serr}")
            self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_missing_args_exit_3(self):
        code, sout, _ = _run([self.SCRIPT, '--artifact', '/x.so'])
        self.assertEqual(code, 3)
