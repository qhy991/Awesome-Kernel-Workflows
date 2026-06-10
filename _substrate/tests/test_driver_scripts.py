import os, sys, json, stat, subprocess, tempfile, textwrap, unittest

SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

BACKENDS = os.path.join(SUB, 'backends')
CUDA = os.path.join(BACKENDS, 'cuda')
TRITON = os.path.join(BACKENDS, 'triton')
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def _triton_profiler_available():
    """True iff this box can import triton.profiler AND has a CUDA device (GPU tier)."""
    code, _out, _err = _run([sys.executable, '-c',
                             'import torch,triton.profiler;'
                             'import sys;sys.exit(0 if torch.cuda.is_available() else 1)'])
    return code == 0


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
            # bin dir — so neither ncu nor nsys resolve on macOS and GPU boxes.
            env = dict(os.environ)
            env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + '/usr/bin' + os.pathsep + '/bin'
            code, sout, serr = _run([self.SCRIPT, '--artifact', art,
                                     '--problem', prob, '--out', out], env=env)
            self.assertEqual(code, 4, msg=f"out={sout} err={serr}")
            self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_nsys_fallback_when_ncu_absent(self):
        fixture_sqlite = os.path.join(FIXTURES, 'nsys', 'vector_add.sqlite')
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'nsys'), textwrap.dedent(f'''\
                #!/usr/bin/env bash
                base=""
                while [ $# -gt 0 ]; do
                  case "$1" in
                    -o) base="$2"; shift 2 ;;
                    *) shift ;;
                  esac
                done
                cp "{fixture_sqlite}" "${{base}}.sqlite"
                exit 0
            '''))
            art = os.path.join(td, 'k.bin')
            _write_exec(art, '#!/usr/bin/env bash\nexit 0\n')
            prob = self._problem(td)
            out = os.path.join(td, 'native.sqlite')
            env = dict(os.environ)
            env['PATH'] = td + os.pathsep + os.path.dirname(sys.executable) + os.pathsep + '/usr/bin'
            code, sout, serr = _run([self.SCRIPT, '--artifact', art,
                                     '--problem', prob, '--out', out], env=env)
            self.assertEqual(code, 0, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), True, p)
            self.assertEqual(p.get('profiler'), 'nsys', p)
            self.assertEqual(p.get('format'), 'nsys-sqlite', p)
            self.assertEqual(p.get('native_profile'), out, p)
            self.assertTrue(os.path.isfile(out))

    def test_missing_args_exit_3(self):
        code, sout, _ = _run([self.SCRIPT, '--artifact', '/x.so'])
        self.assertEqual(code, 3)


class TestTritonL0(unittest.TestCase):
    VALIDATOR = os.path.join(BACKENDS, 'validate_backend.py')

    def test_triton_dir_passes_l0(self):
        code, out, err = _run([sys.executable, self.VALIDATOR, TRITON])
        self.assertEqual(code, 0, msg=f"out={out} err={err}")
        self.assertEqual(_json_or_raw(out).get('ok'), True, out)

    def test_triton_to_evidence_parses_proton_hatchet_roofline(self):
        # Triton is profiled by Proton (triton.profiler), NOT ncu. to_evidence.py maps the
        # proton .hatchet JSON -> canonical metrics: latency from device time, dram_pct/sm_pct
        # as device-derived roofline estimates from the bytes/flops scope annotation.
        fixture = os.path.join(FIXTURES, 'proton', 'add_kernel.hatchet')
        code, out, err = _run([sys.executable,
                               os.path.join(TRITON, 'to_evidence.py'),
                               '--native', fixture, '--format', 'proton-hatchet'])
        self.assertEqual(code, 0, msg=f"{out} {err}")
        p = _json_or_raw(out)
        self.assertEqual(p['source_backend'], 'triton', p)
        self.assertEqual(p['metrics']['_vendor'], 'nvidia', p)
        # latency_ms = time_ns(109985) / count(20) / 1e6
        self.assertAlmostEqual(p['metrics']['latency_ms'], 109985 / 20 / 1e6, places=9)
        # roofline: dram_pct exceeds 100 (memory-bound add), sm_pct ~1%, occupancy unavailable
        self.assertGreater(p['metrics']['dram_pct'], 100.0, p)
        self.assertLess(p['metrics']['sm_pct'], 5.0, p)
        self.assertIsNone(p['metrics']['occupancy'], p)
        self.assertNotIn('occupancy', p['coverage'], p)
        for k in ('latency_ms', 'dram_pct', 'sm_pct'):
            self.assertIn(k, p['coverage'], p)

    def test_triton_to_evidence_null_rule_without_bytes_flops(self):
        # No bytes/flops scope annotation -> dram_pct/sm_pct are null (NOT fabricated 0.0),
        # only latency_ms is covered, so diagnose.py degrades to `unknown` honestly.
        fixture = os.path.join(FIXTURES, 'proton', 'no_roofline.hatchet')
        code, out, err = _run([sys.executable,
                               os.path.join(TRITON, 'to_evidence.py'),
                               '--native', fixture])
        self.assertEqual(code, 0, msg=f"{out} {err}")
        p = _json_or_raw(out)
        self.assertIsNone(p['metrics']['dram_pct'], p)
        self.assertIsNone(p['metrics']['sm_pct'], p)
        self.assertIsNone(p['metrics']['occupancy'], p)
        self.assertEqual(p['coverage'], ['latency_ms'], p)

    def test_triton_to_evidence_rejects_ncu_csv_format(self):
        # The triton driver no longer accepts ncu-csv; only proton formats.
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, 'x.hatchet')
            with open(f, 'w') as fh:
                fh.write('[]')
            code, out, _ = _run([sys.executable, os.path.join(TRITON, 'to_evidence.py'),
                                 '--native', f, '--format', 'ncu-csv'])
            self.assertEqual(code, 3, msg=out)
            self.assertEqual(_json_or_raw(out).get('ok'), False)


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

    def test_profile_missing_args_exit_3(self):
        code, sout, _ = _run([self.PROFILE, '--artifact', '/x'])  # no --problem/--out
        self.assertEqual(code, 3)
        self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_profile_no_source_degrades_exit_4(self):
        # Without a runnable --source launcher contract (or when proton/CUDA is absent),
        # profile.sh degrades to "profiler unavailable" -> exit 4, pointer ok:false, name proton.
        with tempfile.TemporaryDirectory() as td:
            art = os.path.join(td, 'cache_dir')  # real dir so the -e preflight clears
            os.makedirs(art)
            prob = os.path.join(td, 'p.json')
            with open(prob, 'w') as fh:
                json.dump({"op": "add"}, fh)
            out = os.path.join(td, 'prof.hatchet')
            code, sout, serr = _run([self.PROFILE, '--artifact', art, '--problem', prob,
                                     '--out', out])
            self.assertEqual(code, 4, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), False, p)
            self.assertEqual(p.get('profiler'), 'proton', p)

    @unittest.skipUnless(_triton_profiler_available(),
                         "GPU tier: needs CUDA device + triton.profiler (proton)")
    def test_profile_proton_end_to_end_on_gpu(self):
        # Real Proton run over the launcher fixture, then to_evidence on the produced hatchet.
        launcher = os.path.join(FIXTURES, 'proton', 'launcher_add.py')
        with tempfile.TemporaryDirectory() as td:
            art = os.path.join(td, 'cache_dir')
            os.makedirs(art)
            prob = os.path.join(td, 'p.json')
            with open(prob, 'w') as fh:
                json.dump({"op": "add", "profile_reps": 20}, fh)
            out = os.path.join(td, 'prof.hatchet')
            code, sout, serr = _run([self.PROFILE, '--artifact', art, '--problem', prob,
                                     '--out', out, '--source', launcher,
                                     '--kernel-name', 'add_kernel'])
            self.assertEqual(code, 0, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), True, p)
            self.assertEqual(p.get('profiler'), 'proton', p)
            self.assertEqual(p.get('format'), 'proton-hatchet', p)
            self.assertTrue(os.path.isfile(out), "hatchet not written")
            # to_evidence parses the real hatchet -> latency + roofline dram_pct/sm_pct.
            code2, out2, err2 = _run([sys.executable, os.path.join(TRITON, 'to_evidence.py'),
                                      '--native', out])
            self.assertEqual(code2, 0, msg=f"{out2} {err2}")
            m = _json_or_raw(out2)['metrics']
            self.assertGreater(m['latency_ms'], 0.0)
            self.assertIsNotNone(m['dram_pct'])  # launcher annotates BYTES -> roofline
            self.assertIsNone(m['occupancy'])

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
