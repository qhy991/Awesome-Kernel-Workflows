# Issue draft: implement generic `triton/run.sh` (`assert_close` + `do_bench`)

> **Status:** draft — not yet filed as a GitHub issue  
> **Suggested title:** `feat(triton): implement generic run.sh — assert_close + do_bench (decouple from sol-execbench)`  
> **Labels:** `substrate`, `triton`, `backend-driver`, `enhancement`  
> **Related:** AWK #25 (`flags.yaml` SSOT), `BACKEND-DRIVER-SDK.md` §4.5, `REGISTRY.md` triton launcher contract

---

## Background

`_substrate/backends/triton/run.sh` is currently a **stub**. Even when CUDA and Triton are available, the embedded Python path prints `"real triton run not yet implemented"` and exits 2. Any workflow that routes evaluation through the Triton substrate driver (e.g. `adaexplore-kernel-optimization` with `backend=triton`) therefore gets `correct=false` and `speedup=0` for every candidate — poisoned MCTS rewards with no real GPU evidence.

Observed in production (KerSor FI `024_rmsnorm_h2048`, session `20260629-050304`):

| Round | Workflow | Evaluator path | Result |
|-------|----------|----------------|--------|
| 1 | `kernelagent-triton-synthesis` | Direct `sol-execbench` CLI | 1.0314×, 4/4 correct |
| 2 | `adaexplore-kernel-optimization` | `_substrate/backends/triton/run.sh` | Infrastructure failure, 0/2 correct |

`profile.sh` already implements the Proton path and documents a launcher ABI. `run.sh` was never brought to parity. `manifest.json` still lists `status: experimental`.

---

## Goals

1. Implement a full GPU-tier `triton/run.sh`:
   - Correctness: `torch.testing.assert_close`
   - Latency: `triton.testing.do_bench` (device-side timing, not wall clock)
   - Output: JSON envelope with **exactly** the keys `anti_cheat.py` reads (same shape as `cuda/run.sh`)
2. **Generic substrate contract only:** depend on a substrate-owned `problem.json` + Python launcher ABI. Do **not** import, shell out to, or special-case `sol-execbench` CLI or directory layouts.
3. **Align with `profile.sh`:** reuse the same launcher / problem conventions so `build → run → profile` share one problem description.
4. Preserve macOS / no-GPU degradation (deferred GPU tier; existing L0 CI tests stay green).

---

## Non-goals

- Do **not** recognize `sol-execbench` problem directories (`definition.json`, `workload.jsonl`, `reference.py`, etc.) inside `run.sh`.
- Do **not** embed FlashInfer-Bench / Contest L2–specific logic in the substrate.
- Do **not** implement the CUDA `run.sh` GPU path in this issue.
- Do **not** use NCU for run-stage timing (run uses `do_bench`; profile continues to use Proton).

**Benchmark integration pattern:** a workflow or standalone adapter **materializes** an external benchmark into `{launcher.py, problem.json}` before calling the substrate. The driver stays language×vendor neutral per `BACKEND-DRIVER-SDK.md`.

---

## Current state and gaps

| Component | State | Gap |
|-----------|-------|-----|
| `run.sh` | Stub (L69–70) | No import / compare / bench |
| `profile.sh` | Implemented | Launcher ABI exists (`make_inputs`, `forward`) |
| `build.sh` | Partial | Warmup may not force a real launch; cache can be empty (TODO L53) |
| `flags.yaml` | Present | `run.sh` lacks `--source` (profile.sh has it) |
| Preflight | `[ -f "$PROBLEM" ]` | Accepts **files only** — incompatible with sol-execbench **directories** by design |

Passing a sol-execbench benchmark directory to `--problem` correctly fails preflight (exit 3). Fixing that belongs in a workflow adapter, not by teaching `run.sh` about benchmark directory trees.

---

## Proposed design

### 1. CLI extension (`flags.yaml` + `_gen_flag_parser.py`)

Add `--source` to match `profile.sh`:

```bash
run.sh \
  --artifact <cache_dir> \      # TRITON_CACHE_DIR (build.sh --out)
  --source <launcher.py> \      # candidate kernel / launcher module (new; required on GPU tier)
  --problem <problem.json> \    # generic problem descriptor (must be a regular file)
  --out <result.json> \
  [--reps N] [--rtol f] [--atol f] [--baseline eager|compile|both]
```

- `--artifact`: directory (build output / Triton cache); keep `[ -e "$ARTIFACT" ]`.
- `--source`: Python launcher module (same contract as profile).
- `--problem`: JSON file only.

### 2. Launcher ABI (same as `profile.sh` / `REGISTRY.md`)

The `--source` module must expose:

```python
def make_inputs() -> tuple: ...
# or
def make_inputs_from_problem(problem: dict) -> tuple: ...

def forward(*inputs): ...   # or launch(*inputs)
```

**Correctness reference** — pick one approach and document it (recommend **A**):

**A (recommended):** `problem.json` names a reference launcher:

```json
{
  "reference": {
    "module": "path/to/reference_launcher.py",
    "entry": "forward"
  }
}
```

**B:** The candidate launcher also exports `reference_forward(*inputs)`; `problem.json` may omit `reference`.

### 3. Minimal generic `problem.json` schema (substrate-owned, v1)

```json
{
  "schema_version": 1,
  "op": "rmsnorm",
  "inputs": {},
  "reference": {
    "module": "reference_launcher.py",
    "entry": "forward"
  },
  "bench": {
    "reps": 50,
    "warmup": 25
  },
  "tolerance": {
    "rtol": 0.001,
    "atol": 0.001
  },
  "baseline": "both",
  "bytes": null,
  "flops": null
}
```

- CLI flags `--rtol`, `--atol`, `--reps`, `--baseline` override problem fields when present.
- `bytes` / `flops` are for profile roofline annotation; `run.sh` may ignore them.
- No benchmark-specific fields (`definition.json`, `workload.jsonl`, etc.).

### 4. `run.sh` GPU execution flow

```
1. Preflight (args, files, python3, CUDA, triton)
2. TRITON_CACHE_DIR = --artifact
3. import --source launcher
4. problem = json.load(--problem)
5. inputs = make_inputs[_from_problem](problem)
6. candidate_out = forward(*inputs)
7. reference_out = reference_forward(*inputs)   # per schema A or B
8. torch.testing.assert_close(candidate, reference, rtol, atol)
   → correct true/false; record max_abs_err / max_rel_err
9. candidate_latency_ms = triton.testing.do_bench(lambda: forward(*inputs), rep=...)
10. If --baseline eager|both: eager_latency_ms = do_bench(eager reference)
11. If --baseline compile|both: compile_latency_ms = do_bench(torch.compile(ref))
12. claimed_speedup = eager_latency_ms / candidate_latency_ms
    (honesty: correct=false ⇒ claimed_speedup ≤ 1.0)
13. Write stdout + --out; exit 0 (incorrectness is data, not exit 2; crashes → exit 2)
```

Per spec §4.5:

- `compile_latency_ms` = **torch.compile baseline latency**, not build time.
- `build_latency_ms` lives only in `build.sh`.

### 5. `build.sh` follow-up (same PR or separate)

Warmup must run at least one real `forward(*inputs)` so `TRITON_CACHE_DIR` is non-empty. Can share `tests/fixtures/proton/launcher_add.py`.

### 6. Workflow-side adapter (out of scope for `run.sh`, required for acceptance)

Workflows using the substrate driver should:

1. `build.sh --source <candidate.py> --out <cache_dir>`
2. `run.sh --artifact <cache_dir> --source <candidate.py> --problem <problem.json> --out <result.json>`
3. Generate `problem.json` from `args.problem_path` when the upstream benchmark is not already a JSON file (e.g. materialize from a sol-execbench directory in the workflow layer).

Legacy path (`language=triton` without `backend_dir`) continues to invoke `benchmark_command` directly and does not require this driver.

---

## Test plan

| Tier | Coverage |
|------|----------|
| L0 (no GPU) | Missing args → exit 3; full envelope keys; `claimed_speedup ≤ 1.0` when `correct:false` |
| L1 (fake) | Stub launcher with wrong output → `correct:false`; import/syntax errors → exit 2 |
| GPU tier | `@unittest.skipUnless(cuda+triton)`: `tests/fixtures/proton/launcher_add.py` + minimal `problem.json` → `correct:true`, `candidate_latency_ms > 0` |
| Contract | `anti_cheat.evaluate()` accepts result; `evidence_schema` honesty rules pass |
| Regression | Existing `*-triton-dryrun.test.js` stay green; add `test_run_end_to_end_on_gpu` |

---

## Acceptance criteria

- [ ] `triton/run.sh` on CUDA+triton no longer returns `"real triton run not yet implemented"`.
- [ ] Evaluation uses **only** `problem.json` + launcher; no sol-execbench package or directory layout inside substrate.
- [ ] Correctness via `torch.testing.assert_close`; latency via `triton.testing.do_bench`.
- [ ] stdout / `--out` JSON keys match `anti_cheat.py` exactly.
- [ ] `correct:false` ⇒ `claimed_speedup ≤ 1.0`.
- [ ] `--baseline both` fills `eager_latency_ms` and `compile_latency_ms` (or `compile_latency_ms: null` when no compile equivalent).
- [ ] `flags.yaml` gains `--source`; parser regenerated via `_gen_flag_parser.py`.
- [ ] `REGISTRY.md` documents run.sh launcher + `problem.json` contract.
- [ ] Optional: bump `manifest.json` `status` to `stable` after GPU tests pass.
- [ ] Integration check: KerSor `adaexplore + backend=triton` gets real rewards when given a valid materialized `problem.json` (validation lives outside substrate).

---

## Repro

```bash
# Current (fails)
_substrate/backends/triton/run.sh \
  --artifact /tmp/cache --problem /path/to/problem.json --out /tmp/r.json
# → {"error":"real triton run not yet implemented", ...} exit 2

# Expected (generic fixture)
_substrate/backends/triton/run.sh \
  --artifact /tmp/cache \
  --source _substrate/tests/fixtures/proton/launcher_add.py \
  --problem /tmp/problem.json \
  --out /tmp/r.json
# → {"ok":true,"compiled":true,"correct":true,"candidate_latency_ms":...,"claimed_speedup":...}
```

---

## Open questions (resolve before implementation)

1. **Reference scheme:** problem.json `reference.module` (A) vs launcher `reference_forward` (B)?
2. **`--artifact` semantics:** cache dir only, or also allow a `.py` path (align with build output)?
3. **Multi-workload:** support `cases: [...]` in v1, or single-case only (recommend single-case v1; adapter loops `run.sh`)?
4. **`build.sh` in same PR:** without real warmup, cold JIT skews `do_bench` noise.

---

## Reference files

- `_substrate/backends/triton/run.sh` (stub)
- `_substrate/backends/triton/profile.sh` (launcher contract reference)
- `_substrate/tests/fixtures/proton/launcher_add.py`
- `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md` §4.5
- `_substrate/backends/REGISTRY.md` — triton profile launcher contract
- `_substrate/tests/test_driver_scripts.py` — `TestTritonScripts`
