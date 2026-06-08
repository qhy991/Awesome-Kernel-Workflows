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
            with open(src, 'w') as fh:
                fh.write("//\n")
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out],
                                    env=_path_env(td))
            self.assertEqual(code, 0, msg=f"{sout} {serr}")
            with open(rec) as fh:
                self.assertIn('-lineinfo', fh.read())

    def test_build_compile_failure_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'nvcc'), FAKE_NVCC_FAIL)
            src = os.path.join(td, 'kernel.cu'); out = os.path.join(td, 'kernel.so')
            with open(src, 'w') as fh:
                fh.write("//\n")
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
            with open(src, 'w') as fh:
                fh.write("//\n")
            # Use a curated PATH that has python3 + coreutils/bash but excludes the CUDA
            # bin dir — so the script's "nvcc not found" guard fires on both macOS and GPU boxes.
            env = dict(os.environ)
            env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + '/usr/bin' + os.pathsep + '/bin'
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
        with open(p, 'w') as fh:
            json.dump({"op": "add"}, fh)
        return p

    def test_exists_executable_and_syntax(self):
        self.assertTrue(os.path.isfile(self.SCRIPT), "cuda/profile.sh missing")
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))
        code, _, err = _run(['bash', '-n', self.SCRIPT])
        self.assertEqual(code, 0, msg=err)

    def test_profile_ok_with_fake_ncu_writes_csv_and_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'ncu'), FAKE_NCU_CSV)
            art = os.path.join(td, 'k.so')
            with open(art, 'w') as fh:
                fh.write("")
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
            with open(out) as fh:
                self.assertIn('sm__warps_active', fh.read())

    def test_profile_requests_the_four_counters(self):
        with tempfile.TemporaryDirectory() as td:
            rec = os.path.join(td, 'argv.txt')
            _write_exec(os.path.join(td, 'ncu'), textwrap.dedent(f'''\
                #!/usr/bin/env bash
                echo "$@" > "{rec}"
                echo '"ID","Metric Name","Metric Value"' ; exit 0
            '''))
            art = os.path.join(td, 'k.so')
            with open(art, 'w') as fh:
                fh.write("")
            prob = self._problem(td); out = os.path.join(td, 'n.csv')
            _run([self.SCRIPT, '--artifact', art, '--problem', prob, '--out', out],
                 env=_path_env(td))
            with open(rec) as fh:
                argv = fh.read()
            for c in ('gpu__time_duration.sum',
                      'sm__throughput.avg.pct_of_peak_sustained_elapsed',
                      'dram__bytes_read.sum.pct_of_peak_sustained_elapsed',
                      'sm__warps_active.avg.pct_of_peak_sustained_active'):
                self.assertIn(c, argv, f"profile.sh did not request {c}")

    def test_profiler_absent_exit_4(self):
        with tempfile.TemporaryDirectory() as td:
            art = os.path.join(td, 'k.so')
            with open(art, 'w') as fh:
                fh.write("")
            prob = self._problem(td); out = os.path.join(td, 'n.csv')
            # Use a curated PATH that has python3 + coreutils/bash but excludes the CUDA
            # bin dir — so the script's "ncu not available" guard fires on both macOS and GPU boxes.
            env = dict(os.environ)
            env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + '/usr/bin' + os.pathsep + '/bin'
            code, sout, serr = _run([self.SCRIPT, '--artifact', art,
                                     '--problem', prob, '--out', out], env=env)
            self.assertEqual(code, 4, msg=f"out={sout} err={serr}")
            self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_missing_args_exit_3(self):
        code, sout, _ = _run([self.SCRIPT, '--artifact', '/x.so'])
        self.assertEqual(code, 3)


class TestTritonL0(unittest.TestCase):
    VALIDATOR = os.path.join(BACKENDS, 'validate_backend.py')

    def test_triton_dir_passes_l0(self):
        code, out, err = _run([sys.executable, self.VALIDATOR, TRITON])
        self.assertEqual(code, 0, msg=f"out={out} err={err}")
        self.assertEqual(_json_or_raw(out).get('ok'), True, out)

    def test_triton_to_evidence_uses_shared_mapper_source_triton(self):
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, 'n.csv')
            _write_exec(os.path.join(td, '_e.sh'), FAKE_NCU_CSV)
            with open(csv, 'w') as fh:
                subprocess.run([os.path.join(td, '_e.sh')], stdout=fh)
            code, out, err = _run([sys.executable,
                                   os.path.join(TRITON, 'to_evidence.py'),
                                   '--native', csv, '--format', 'ncu-csv'])
            self.assertEqual(code, 0, msg=f"{out} {err}")
            p = _json_or_raw(out)
            self.assertEqual(p['source_backend'], 'triton', p)
            self.assertEqual(p['metrics']['_vendor'], 'nvidia', p)
            self.assertAlmostEqual(p['metrics']['occupancy'], 0.51, places=4)
            self.assertAlmostEqual(p['metrics']['dram_pct'], 62.0, places=3)  # read + write


class TestTritonScripts(unittest.TestCase):
    BUILD = os.path.join(TRITON, 'build.sh')
    RUN = os.path.join(TRITON, 'run.sh')
    PROFILE = os.path.join(TRITON, 'profile.sh')

    def test_all_three_exist_executable_syntax(self):
        for s in (self.BUILD, self.RUN, self.PROFILE):
            self.assertTrue(os.path.isfile(s), f"{s} missing")
            self.assertTrue(os.access(s, os.X_OK), f"{s} not executable")
            code, _, err = _run(['bash', '-n', s])
            self.assertEqual(code, 0, msg=f"{s}: {err}")

    def test_build_missing_args_exit_3(self):
        code, sout, _ = _run([self.BUILD, '--source', '/k.py'])  # no --out
        self.assertEqual(code, 3)
        self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_build_envelope_keys_present_on_triton_absent(self):
        # triton is absent on macOS → build.sh's warmup must still print the envelope.
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'kernel.py'); out = os.path.join(td, 'art.json')
            with open(src, 'w') as fh:
                fh.write("# triton kernel\n")
            code, sout, serr = _run([self.BUILD, '--source', src, '--out', out])
            p = _json_or_raw(sout)
            self.assertIn('ok', p); self.assertIn('compiled', p)
            self.assertIn('build_latency_ms', p); self.assertIn('stderr_tail', p)
            # triton absent ⇒ not compiled, op-error exit 2
            self.assertEqual(p['compiled'], False, p)
            self.assertEqual(code, 2, msg=f"out={sout} err={serr}")

    def test_run_missing_artifact_full_key_set_exit_3(self):
        with tempfile.TemporaryDirectory() as td:
            prob = os.path.join(td, 'p.json')
            with open(prob, 'w') as fh:
                json.dump({"op": "add"}, fh)
            out = os.path.join(td, 'r.json')
            code, sout, _ = _run([self.RUN, '--artifact', '/nope.json',
                                  '--problem', prob, '--out', out])
            self.assertEqual(code, 3)
            p = _json_or_raw(sout)
            for k in ('compiled', 'correct', 'candidate_latency_ms', 'eager_latency_ms',
                      'compile_latency_ms', 'claimed_speedup'):
                self.assertIn(k, p, f"missing {k}: {p}")
            self.assertEqual(p['correct'], False)
            self.assertLessEqual(p['claimed_speedup'], 1.0)

    def test_profile_ok_with_fake_ncu_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'ncu'), FAKE_NCU_CSV)
            art = os.path.join(td, 'art.json')
            with open(art, 'w') as fh:
                fh.write("{}")
            prob = os.path.join(td, 'p.json')
            with open(prob, 'w') as fh:
                json.dump({"op": "add"}, fh)
            out = os.path.join(td, 'n.csv')
            code, sout, serr = _run([self.PROFILE, '--artifact', art, '--problem', prob,
                                     '--out', out, '--kernel-name', 'add_kernel'],
                                    env=_path_env(td))
            self.assertEqual(code, 0, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), True, p)
            self.assertEqual(p.get('native_profile'), out, p)
            self.assertEqual(p.get('format'), 'ncu-csv', p)
            self.assertTrue(os.path.isfile(out))

    def test_profile_ncu_absent_exit_4(self):
        with tempfile.TemporaryDirectory() as td:
            art = os.path.join(td, 'a.json')
            with open(art, 'w') as fh:
                fh.write("{}")
            prob = os.path.join(td, 'p.json')
            with open(prob, 'w') as fh:
                json.dump({"op": "add"}, fh)
            out = os.path.join(td, 'n.csv')
            env = dict(os.environ)
            env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + '/usr/bin' + os.pathsep + '/bin'
            code, sout, _ = _run([self.PROFILE, '--artifact', art, '--problem', prob,
                                  '--out', out], env=env)
            self.assertEqual(code, 4)

    def test_run_no_cuda_or_triton_deferred_envelope(self):
        # Passes a REAL (existing) artifact dir so the bash preflight clears, then enters
        # the Python block. On this CPU-only box the Python block should return a deferred
        # error envelope: ok:false, exit 2, all six anti_cheat keys present.
        with tempfile.TemporaryDirectory() as td:
            # Use a real directory as the artifact so '-e "$ARTIFACT"' preflight passes.
            art = os.path.join(td, 'artifact_dir')
            os.makedirs(art)
            prob = os.path.join(td, 'problem.json')
            with open(prob, 'w') as fh:
                json.dump({"op": "add", "shape": [128, 128]}, fh)
            out = os.path.join(td, 'result.json')
            env = dict(os.environ)
            env['PATH'] = (os.path.dirname(sys.executable)
                           + os.pathsep + '/usr/bin' + os.pathsep + '/bin')
            code, sout, _serr = _run([self.RUN, '--artifact', art,
                                      '--problem', prob, '--out', out], env=env)
            # Python block reached: GPU/triton absent → deferred error envelope, exit 2.
            self.assertNotEqual(code, 0, msg=f"expected non-zero exit, got 0; out={sout}")
            self.assertEqual(code, 2, msg=f"expected exit 2 (deferred op-error), got {code}; out={sout}")
            p = _json_or_raw(sout)
            self.assertIsInstance(p, dict, msg=f"stdout is not JSON: {sout!r}")
            self.assertEqual(p.get('ok'), False, msg=f"expected ok:false; {p}")
            # All six anti_cheat keys must be present regardless of execution path.
            for k in ('compiled', 'correct', 'candidate_latency_ms',
                      'eager_latency_ms', 'compile_latency_ms', 'claimed_speedup'):
                self.assertIn(k, p, msg=f"missing anti_cheat key '{k}': {p}")
            # On this CPU-only box torch is importable but CUDA is unavailable,
            # so the script reaches the cuda-check branch with compiled=True.
            self.assertEqual(p.get('compiled'), True,
                             msg=f"expected compiled:true on no-CUDA path; {p}")
