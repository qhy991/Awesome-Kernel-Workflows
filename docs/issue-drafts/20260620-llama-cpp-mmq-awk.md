# AWK Issue Draft — llama.cpp mmq.cuh embedded experiments (2026-06-20)

**Target repo**: https://github.com/qhy991/Awesome-Kernel-Workflows  
**Labels**: `bug`, `enhancement`, `embedded`, `generalist`, `profiling`  
**Related KerSor issue**: qhy991/KerSor (see `docs/issue-drafts/20260620-llama-cpp-mmq-kersor.md`)

## Title

Generalist embedded_inplace validated on llama.cpp mmq; remaining gaps: NCU-centric Profile, anti_cheat reporting, integration wiring scope, solver focus alignment

## Summary

End-to-end experiments on `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` (RTX 4090 sm_89,
MUL_MAT harness via `test-backend-ops`) exercised `generalist-kernel-optimization`
with `embedded_inplace`. AWK commits through `790fdfa`/`d4487d5` fixed the
blocking integration bugs from session `20260620-153001`. A subsequent real
optimization run (`20260620-173848`, `exp_dir=mmq-real-opt-1`) proved the
embedded gate at scale (2 iterations × 2 candidates, all compile+correct) but
exposed workflow/solver/profiling gaps below.

---

## Fixed (please close / reference in changelog)

### F1. Integration mis-routing + worktree hard-fail (pre-790fdfa) — **FIXED**

- **Symptom**: `integration method=standalone` for `.cuh` + `project_root` +
  build/test commands; `parallel[0] failed: Cannot create agent worktree: not
  in a git repository`; `attempts=[]`, no build/test/bench.
- **Fix**: Trust boundary (classify → mechanical `integration_strategist.py`);
  `.cuh` heuristic forces `can_standalone=no`; `useWorktree` only for
  `standalone`; embedded serial eval + restore safety net.
- **Verified**: SMOKE `smoke-generalist-mmq-retry` + real run `mmq-real-opt-1`.

### F2. InPlacePatch `classifyProposalResult` TypeError — **FIXED** (`de52cd0`)

- **Symptom**: `TypeError: undefined is not an object (evaluating 'prop.value.applied')`
  at ClassifyProposalResult (session `20260620-145823`).
- **Fix**: Separate propose grounding from measured-numeric grounding.
- **Status**: Code merged; dedicated InPlacePatch SMOKE not re-run post-fix.

---

## Open — Workflow / solver

### O1. Generalist Profile phase ignores `profiling-strategist` on embedded path (High) — **FIXED** (`9220923`)

**Symptom**: Session `20260620-173848` selected `native_profiler/ncu` at Setup
(`prof_cache.json`), but per-iteration Profile uses hardcoded
`ncu_command || benchmark_command` prompt — not `PROFILING_DECISION`.

**Impact**: When `ncu_command` is present but NCU hw counters are blocked
(`RmProfilingAdminOnly=1`), agents ad-hoc fall back to nsys+benchmark estimates.
This is non-deterministic and bypasses the substrate ladder
(`native → perf_heuristic → static`).

**Suggested fix**:
- Branch Profile/Diagnose on `PROFILING_DECISION.method` for **all** integration
  modes (not only `USE_DRIVER_STANDALONE`).
- If native probe fails (binary present but counters blocked), abstain to
  `perf_heuristic` with stamped `confidence=inferred`.

**Evidence**: `mmq-real-opt-1/genome.jsonl` Profile notes;
`20260620-173848/run-1/analysis.md` ins-4.

---

### O2. nsys on no-ncu hosts: A/B design decision, gated on GPU nsys output (Medium — needs design + GPU)

**Symptom**: `_substrate/backends/cuda/manifest.json` declares `profiler.name=ncu`
only. `nsys` is `optional_tools` but not in the ladder. Hosts with nsys but no
working ncu fall through to `perf_heuristic` — same as having no profiler.

**Why "just add nsys to the ladder" is NOT enough**: the strategist's native gate
(`_substrate/profiling/profiling_strategist.py:104-112`) requires `latency_ms` AND
≥1 bottleneck axis (`dram_pct`/`sm_pct`); a latency-only native is forced to
`perf_heuristic` (unit test S3 proves this for Metal counters). nsys typically
emits kernel timeline + duration and does **not** directly give `dram_pct`/`sm_pct`
without extra ncu runs or heavy post-estimation. So adding nsys to the ladder
without changing the gate / manifest capabilities still fails the native gate and
lands at `perf_heuristic` — identical to the mmq run's agent-improvised
"nsys + benchmark 反推 dram" path, just non-deterministic instead of routed.

**Impact**: Cannot structurally use nsys when ncu is absent/admin-blocked; agents
improvise (mmq `20260620-173848` genome Profile: nsys+benchmark estimates,
`confidence=inferred`, not strategist-routed).

**Two real paths — a design decision that needs GPU nsys output to pick**:

| Path | Meaning | To validate on GPU |
|------|---------|--------------------|
| **A. nsys = native_profiler** | new `nsys_to_evidence.py`; manifest declares `latency_ms + dram_pct/sm_pct` (or occupancy); possibly a relaxed/new gate kind (e.g. `timeline_native` guarantees latency + kernel attribution only) | run nsys on the mmq harness, check the normalizer can **stably produce** `{latency_ms, dram_pct|sm_pct}` consumable by `diagnose.py`; compare ncu/perf_heuristic for fewer mis-routes |
| **B. nsys = perf_heuristic input** | ladder unchanged; nsys only helps parse perf / pick the hot kernel; evidence stays `profile_heuristic` + `inferred` | run nsys, verify "nsys timeline + test-backend-ops perf" points at batch-vs-decode more accurately than pure perf stdout; do **not** claim `confidence=measured` |

The mmq experiment was a **wild version of B**: agent used nsys for timeline +
benchmark for roofline, but bypassed the strategist's deterministic routing → not
reproducible.

**Minimal GPU experiment to decide A vs B** (do before any code):
1. Collect: alongside `test-backend-ops perf -o MUL_MAT ...`, run
   `nsys profile --stats=true -o mmq.nsys-rep -- ...` (like KerSor
   `profile-tools-smoke`, but bound to the mmq harness).
2. Inspect what nsys can export: kernel name, duration, and whether the memory
   throughput column is enough to compute `dram_pct`/`sm_pct`, or only latency.
3. Decide: stably derivable bottleneck axis → **A** (write normalizer + explicit
   capabilities); timeline only → **B** (strategist stays perf_heuristic; Generalist
   Profile uses nsys as optional enrich, does NOT raise confidence).

**Status**: problem GPU-confirmed; fix unimplemented and **must not be coded before
the A/B decision** — changing the gate/manifest/normalizer blind is a high-risk
unvalidated substrate change.

**Evidence**: `_substrate/profiling/profiling_strategist.py:104-112` (native gate);
`_substrate/profiling/tests/test_strategist.sh` S3 (latency-only → perf_heuristic);
mmq `20260620-173848` genome Profile (agent-improvised nsys path).

---

### O3. `anti_cheat.py` zeros `recorded_speedup` for valid compile+correct ~1.0x runs (Medium) — **FIXED** (`9220923`)

**Symptom**: SMOKE and real runs show `genome.jsonl` speedup ~1.0–1.02x but
`output.json attempts[].speedup=0`, `reward=0`. Beam keeps baseline only when
aggregate geomean does not beat eager baseline with `reward>=2`.

**Impact**: Orchestrator/analysis consumers misread `attempts=0` as failure;
SMOKE success looks like failure in raw workflow return.

**Suggested fix**:
- Add `measured_speedup` separate from `recorded_speedup`, or a
  `correctness_only` mode for SMOKE/target-not-met-but-valid attempts.
- Generalist Report should surface genome-measured sub-path speedups when beam
  reverts.

**Evidence**: `smoke-generalist-mmq-retry/run-1/cand-1/metrics.json` speedup=1.0 vs
`output.json attempts[0].speedup=0`; real run iter-2-cand-1 genome 1.020x on
q4K/q6K.

---

### O4. Solver optimizes wrong quant sub-path when note focuses q8_0 (Medium — solver behavior)

**Symptom**: Real run (`mmq-real-opt-1`) note required q8_0 focus + 1.05x geomean.
All four candidates improved q4K/q6K decode (1.017–1.020x) via y-tile load
width/cp.async; q8_0 unchanged (~1.000x). Workflow reverted to baseline.

**Impact**: Integration works; optimization goal not met. Likely combination of
(1) aggregate benchmark geomean dilutes sub-path wins, (2) profiling mis-targeted
decode load-width vs q8_0 batch occupancy, (3) no harness filter for q8_0-only gate.

**Suggested fix** (AWK + harness contract):
- Support `benchmark_filter` / per-dtype perf gate in workflow args.
- Plan phase: require explicit sub-path alignment when `op_description` names a dtype.
- Method gate: penalize techniques that only move non-focus dtypes.

**Evidence**: `mmq-real-opt-1/genome.jsonl` Evaluate lines; `20260620-173848/USER_FEEDBACK.md`.

---

### O5. `integration-strategist` fully wired only in Generalist (Medium — scope)

**Symptom**: Only `Generalist/generalist-kernel-optimization.js` calls
`integration_strategist.py` with eval branching. Other ~27 workflows got
profiling-strategist INJECT but not integration wiring per
`_substrate/integration/README.md`.

**Impact**: llama.cpp embedded kernels still break on standalone-default workflows.

**Suggested fix**: Extract shared `integration_eval` inline block (like
`embedded_eval.js` patch) and roll out to priority workflows (CUDAAgent, AKO4X).

---

### O6. Integration decision still agent-`cat` mediated (Low — hardening)

**Symptom**: Mechanical agent runs python + `cat integration_decision.json`; workflow
sets `INTEGRATION_DECISION` from agent return JSON, not direct file read.

**Suggested fix**: Workflow runtime reads `integ_cache.json` / `integration_decision.json`
directly after mechanical runner (mirrors KerSor validate-run file-first discipline).

---

### O7. InPlacePatch SMOKE not re-validated post-fix (Low)

**Symptom**: Session `20260620-152944` dispatched but no `output.json`; prior
`20260620-145823` failed on propose gate before `de52cd0`.

**Suggested fix**: Re-run InPlacePatch SMOKE on mmq.cuh with current AWK main.

---

## Sessions / artifacts

| Session | Purpose | Result |
|---------|---------|--------|
| `20260620-153001` | Generalist SMOKE (pre-fix) | Failed — standalone mis-route |
| `20260620-161544` / `smoke-generalist-mmq-retry` | Generalist SMOKE (790fdfa) | Pass — embedded_inplace gate |
| `20260620-173848` / `mmq-real-opt-1` | Generalist real opt (1.05x q8_0) | Failed target — q4K/q6K wins only |
| `20260620-145823` / `20260620-152944` | InPlacePatch | Failed / incomplete |

Exp dir (real opt): `LlamaCpp-Exp/.kersor/mmq-real-opt-1/`  
Analysis: `LlamaCpp-Exp/.kersor/20260620-173848/run-1/analysis.json`

## Acceptance criteria (for closing this meta-issue)

- [x] Generalist embedded Profile honors `profiling-strategist` decision end-to-end — **FIXED** (`9220923`; code, GPU-behavior confirm pending)
- [ ] No-ncu / nsys-only hosts degrade deterministically to perf_heuristic (no ad-hoc nsys) — **blocked on O2 A/B decision** (needs GPU nsys output)
- [x] `attempts[]` reporting distinguishes valid+measured from beam-recorded speedup — **FIXED** (`9220923`; `anti_cheat.measured_speedup`, GPU-behavior confirm pending)
- [ ] Document or implement dtype/subtest-focused benchmark gate for embedded GEMM — O4, needs GPU 对照
- [ ] InPlacePatch SMOKE PASS on mmq.cuh — O7, needs GPU
