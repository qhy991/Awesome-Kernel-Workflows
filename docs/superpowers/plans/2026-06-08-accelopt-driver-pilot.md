# AccelOpt Driver Pilot (P4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrofit the AccelOpt workflow onto the backend-driver contract — the first workflow to consume a driver — while PROVING the cuda path is byte-identical to today, via a new node `--print-prompts` harness; real triton/GPU execution is deferred.

**Architecture:** A reusable node harness (`_meta/tools/print-workflow-prompts.js`) sandboxes a Workflow .js with stubbed runtime globals (agent/phase/parallel/pipeline/log) and a schema-to-stub generator, capturing every rendered prompt deterministically. AccelOpt's 21 NCU/CUDA seams become `backend_dir`-gated (no backend_dir → verbatim legacy prompts; backend_dir → driver-call + idioms injection + diagnose). The cuda byte-identity gate diffs the retrofit's rendered prompts against a golden captured from today's AccelOpt; the guard is unit-tested in pure JS; a triton dry-run asserts no CUDA tokens leak. Real triton compile/profile is GPU-deferred.

**Tech Stack:** node v24 (vm, node:test, node:assert — no package.json/npm), the P1-P3 python substrate + cuda/triton drivers, the Claude Workflow .js. No GPU here.

> **Critical correctness fact (drives the whole plan):** AccelOpt's main loop *short-circuits* when every `agent()` return is an empty/degenerate stub. Specifically: `read-baseline.kernel_code===''` and `impl.code===''` are falsy (L542 `if (impl && impl.code)` → `allVariants=[]` → no Evaluate call), and `is_correct`/`is_compilable` stubbing to `false` (L639) → `dedupedResults=[]` → no Learn call. The generate-seed (L161) only fires when `kernel_path` is absent. **Therefore the golden capture AND the byte-identity gate MUST drive a deterministic `agentReturns` map (committed fixture) that unlocks the full loop, so every CUDA-laden seam renders and is locked by the diff.** A second fixture pair runs in *generate mode* so the generate-seed seam (S7) also renders. Coverage is asserted by an explicitly enumerated *label set*, never a magic `=== 8`.

---

## File Structure

| Path | Disposition | Purpose |
|---|---|---|
| `_meta/tools/lib/schema-stub.js` | **Create** (Part 1) | Pure `schemaStub(schema)` → minimal valid stub |
| `_meta/tools/lib/run-workflow.js` | **Create** (Part 1) | Strip-export + vm async-wrap sandbox + stub agent/parallel/pipeline; `capturePrompts` core |
| `_meta/tools/print-workflow-prompts.js` | **Create** (Part 1) | CLI + exported `capturePrompts({ workflowPath, args, agentReturns })` → stable-key JSON; CLI accepts `--agent-returns` |
| `_meta/tools/fixtures/accelopt-cuda-args.json` | **Create** (Part 1) | Fixed cuda legacy-path args (no `backend`/`language`/`backend_dir`, `kernel_path` set); feeds BOTH golden capture and the byte-identity gate |
| `_meta/tools/fixtures/accelopt-cuda-agent-returns.json` | **Create** (Part 1) | Deterministic `label`→return map that unlocks the full loop (read/impl/eval/learn) for the OPTIMIZE-mode golden + gate |
| `_meta/tools/fixtures/accelopt-generate-args.json` | **Create** (Part 1) | Fixed GENERATE-mode args (no `kernel_path`; `problem_path` + test/benchmark commands) so the generate-seed seam renders |
| `_meta/tools/fixtures/accelopt-generate-agent-returns.json` | **Create** (Part 1) | `label`→return map for generate mode (adds `generate-initial-kernel`) |
| `_meta/tools/fixtures/accelopt-today.golden.json` | **Create** (Part 1, pre-retrofit) | Frozen golden capture of TODAY's AccelOpt prompts (OPTIMIZE mode) |
| `_meta/tools/fixtures/accelopt-today-generate.golden.json` | **Create** (Part 1, pre-retrofit) | Frozen golden capture of TODAY's AccelOpt prompts (GENERATE mode) |
| `_meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt` | **Create** (Part 1) | The pre-retrofit commit SHA the goldens were captured at; Stage C references it |
| `_meta/tools/fixtures/accelopt-triton-args.json` | **Create** (Part 3) | Triton driver-path args (`backend_dir` set) |
| `_meta/tools/fixtures/triton/kernel.py` | **Create** (Part 3) | Tiny fake triton kernel so a `kernel_path` exists |
| `_meta/tools/test/print-workflow-prompts.test.js` | **Create** (Part 1) | Harness determinism + coverage unit tests |
| `_meta/tools/test/accelopt-cuda-byte-identity.test.js` | **Create** (Part 3) | The headline cuda byte-identity gate vs golden (both modes) |
| `_meta/tools/test/accelopt-guard.test.js` | **Create** (Part 3) | §6.4 guard units via harness |
| `_meta/tools/test/accelopt-triton-dryrun.test.js` | **Create** (Part 3) | Triton prompt-wiring dry-run (no GPU) |
| `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md` | **Create** (Part 3) | GPU/CI-tier checklist (not executable here) |
| `AccelOpt/accelopt-kernel-optimization.js` | **Modify** (Part 2 only) | The 21-seam `backend_dir`-gated retrofit |
| `_substrate/backends/triton/{manifest,idioms}.json`, `build/run/profile.sh`, `to_evidence.py` | **Reuse** (P3) | The triton driver consumed by the dry-run |

**Resolved cross-part conventions (single source of truth):**
- **Harness module:** `_meta/tools/print-workflow-prompts.js`, exporting async `capturePrompts({ workflowPath, args, agentReturns })` → `Array<{ seq, label, phase, prompt }>` in call order. CLI form: `node _meta/tools/print-workflow-prompts.js --workflow <wf> --args <args.json> [--agent-returns <returns.json>] [--out <golden.json>]`.
- **Stub-agent contract:** the stub `agent(prompt, opts)` records `{ seq, label: opts.label, phase: opts.phase, prompt }` and RETURNS a value: `agentReturns[opts.label]` if present, else `schemaStub(opts.schema)`. Body `throw`s propagate (the harness rejects its promise — never catch-and-ignore).
- **Args fixtures:** `accelopt-cuda-args.json` (optimize mode) + `accelopt-generate-args.json` (generate mode). Each feeds both the corresponding golden capture and the byte-identity gate.
- **Agent-returns fixtures:** `accelopt-cuda-agent-returns.json` + `accelopt-generate-agent-returns.json`. **The SAME agent-returns map feeds the golden capture (Stage A) and the byte-identity gate (Stage C).** This is what unlocks the eval/learn/seed seams so all CUDA-laden prompts render and are locked.
- **Golden fixtures:** `accelopt-today.golden.json` + `accelopt-today-generate.golden.json` (captured from PRE-retrofit AccelOpt; never regenerated from the retrofitted tree).
- **Tests live in** `_meta/tools/test/`; run from the repo root with `node --test _meta/tools/test/*.test.js` (NEVER `node --test <dir>` — node v24 treats a bare directory arg as a test file and emits a spurious failing line).

## Testability note (node-renderable vs GPU-deferred)

| Concern | How it is proven here (no GPU) | Why it works |
|---|---|---|
| cuda legacy path unchanged | `capturePrompts` renders the retrofit with cuda args + the SAME agent-returns map used for the golden; byte-diff vs golden | AccelOpt is **deterministic** — no `Date`/`Math.random`/`performance.now` (`sampleWithoutReplacement` L206-220 is a deterministic index-shuffle). Fixed args + fixed `agentReturns` ⇒ identical prompt sequence every run, with the full loop (eval/learn) unlocked. |
| §6.4 guard (resolve/throw) | Run body via harness with crafted `args`; assert resolved `BACKEND` via `caps.some(c=>/CUDA/.test(c.prompt))` or `assert.rejects` on the thrown guard error | Guard is sync, runs before the first `agent()`; the body `throw` propagates through the harness. |
| triton prompt wiring | Run with `backend_dir` + real P3 driver + an `agentReturns` map that returns the load-driver shape AND non-empty impl/eval (so Evaluate/Learn render); assert no CUDA tokens across the FULL rendered set, python fence, triton ABI | The body threads `idioms.json` fields into later prompts; tests the **wiring**, never an execution. |
| **GPU-deferred (NOT proven here):** real triton JIT compile, `run.sh` correctness, `ncu`-attributed CSV, live `diagnose.py` class, `evidence_schema.py validate` | Checklist only (`DEFERRED-GPU-VERIFICATION.md`) | macOS has no `nvcc`/`ncu`/`triton`; AccelOpt only runs in the Workflow runtime. Belongs to a self-hosted NVIDIA-runner CI tier (spec §9.3). |

> **Why degenerate stub returns do not break the gate (harness limitation, surfaced):** the agent returns we feed are deterministic but *not realistic* — they exist only to drive the loop down the same control-flow branches every time. Because the golden AND the retrofit are captured with the **identical** stub + `agentReturns` map, any branch divergence is symmetric and cancels in the diff. **The gate proves RELATIVE byte-identity (today's prompts == retrofit's cuda prompts), NOT that the rendered prompts reflect a realistic GPU run.** The realistic run is the deferred GPU tier.

All Part-1/Part-3 tests run under `node --test _meta/tools/test/*.test.js` with `node:test`+`node:assert` (no `package.json`, no npm). All commits end:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

# Stage A — Harness + golden (Part 1)

> Branch `dev/solver-substrate`. macOS node v24; no GPU/Workflow-runtime/package.json. CommonJS; the workflow is read as a **string**, a lone `export` is stripped, top-level `await`/`return` are handled by textual async-wrapping (see Task 2), and it is **never imported**. AccelOpt is deterministic, so fixed args + a fixed agent-returns map give byte-identical prompts; the byte-diff today vs Stage C (same args + same agent-returns, no `backend_dir`) proves the cuda path unchanged.
>
> **The goldens MUST be captured from TODAY's (pre-retrofit) AccelOpt and committed in this Stage, BEFORE Stage B edits the file.** This is the hard prerequisite for everything downstream. The pre-retrofit commit SHA is emitted into `_meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt` (Task 4 Step 5) so Stage C's re-baseline procedure is self-contained.

## Task 1 — `schema-stub.js` (pure stub generator)

### Files
- **Create** `_meta/tools/lib/schema-stub.js`
- **Create/extend** `_meta/tools/test/print-workflow-prompts.test.js` (schemaStub cases)

### Steps
- [ ] **Step 1 — RED.** Write a failing test asserting `schemaStub` behavior: `enum` ⇒ first element; `string` ⇒ `''`; `number` ⇒ `0`; **`integer` ⇒ `0`**; `boolean` ⇒ `false`; `array` ⇒ `[]`; `object` ⇒ recurse over **all** `properties` (and any `required` not in properties). `additionalProperties:true`-only object ⇒ `{}`. Add an explicit case asserting `schemaStub({type:'number'}) === 0` with a comment: `// LOAD-BEARING: AccelOpt L406/L442 do bestLatency.toFixed(3) and baselineLatency/bestLatency; a null/''/undefined number stub would throw TypeError on .toFixed and crash the capture before final-report.` Note that `integer→0` is included because a retrofit schema could introduce `integer`.
- [ ] **Step 2 — GREEN.** Implement pure `schemaStub(schema)` to satisfy the above. No I/O, no globals. Both `number` and `integer` map to `0`.
- [ ] **Step 3 — Commit** (`P4 harness: pure schemaStub(schema) minimal-valid generator (number/integer→0)`).

## Task 2 — `run-workflow.js` (sandbox + stub runtime, textual async-wrap)

### Files
- **Create** `_meta/tools/lib/run-workflow.js`
- **Extend** `_meta/tools/test/print-workflow-prompts.test.js`

### Steps
- [ ] **Step 1 — RED.** Test that a fixture workflow string with a lone top-level `export`, a genuine top-level `await`, and a top-level `return {meta, ...}` runs under the sandbox and yields the recorded `agent()` calls in order. Add a SECOND RED test that pins the `parallel`/`pipeline` signatures: a fixture that calls `await parallel([()=>agent('a',{label:'a'}), ()=>agent('b',{label:'b'})])` (1-arg array-of-thunks) AND `await pipeline([1,2], (n)=>agent('p'+n,{label:'p'+n}))` (2-arg items+mapper) must record labels `a,b,p1,p2` in that order. This locks AccelOpt's real call shapes: `pipeline(validPlans, fn)` 2-arg at L497 and `parallel([...thunks])` 1-arg at L467/L499/L560/L739.
- [ ] **Step 2 — GREEN.** Read the workflow as a string, strip the lone leading `export` token, then build the executable source as a STRING that wraps the stripped body in an async IIFE that *returns* `{meta, calls}`, and pass THAT string to `vm.runInNewContext` (or `new Function`). This is the load-bearing detail: `vm` rejects the file's genuine top-level `await` (L292) AND top-level `return` (L856) unless the body is textually wrapped in an async function **before** compilation. The exact wrapper:
  ```js
  // strippedBody = source with the lone leading `export ` removed (the body still ends in `return {...}`)
  const wrapped =
    '(async function(){\n' +
    strippedBody +
    '\n})()'
  // wrapped is what vm compiles. The body's own `return {...}` becomes the IIFE's resolved value,
  // and its top-level `await`s are now legal because they're inside an async function.
  ```
  Run it under `vm.runInNewContext(wrapped, sandbox)` where `sandbox` provides:
  - `agent(prompt, opts)` — records `{ seq: nextSeq(), label: opts && opts.label, phase: currentPhase, prompt }`; returns `agentReturns[opts.label]` if supplied, else `schemaStub(opts && opts.schema)`.
  - `phase(title)` — sets `currentPhase = title` for subsequent calls.
  - `async function parallel(thunks){ const o=[]; for(const t of thunks) o.push(await t()); return o }` — EXACT shape (1-arg, array of zero-arg thunks; sequential ⇒ deterministic).
  - `async function pipeline(items, fn){ const o=[]; for(const it of items) o.push(await fn(it)); return o }` — EXACT shape (2-arg: items array + mapper fn; sequential ⇒ deterministic).
  - `log`, `budget` — no-op stubs.
  - The captured `calls` array is closed over by `agent`; the IIFE's returned object exposes the body's `meta` (read as the body's normal `const meta`) so the runner can return `{ meta, calls }`.
  - Propagate body `throw`: do NOT wrap the `await wrapped-IIFE` in a try/catch that swallows — let the rejection propagate so callers can `assert.rejects` on the §6.4 guard throw.
  > **No sentinel-abort meta probe.** `meta` is an ordinary `const` inside the wrapped body; the IIFE returns it directly alongside `calls`. There is no need to run-then-abort.
- [ ] **Step 3 — Commit** (`P4 harness: vm async-wrap sandbox + stub agent/phase/parallel(1-arg)/pipeline(2-arg); export-strip`).

## Task 3 — `print-workflow-prompts.js` (CLI + `capturePrompts`)

### Files
- **Create** `_meta/tools/print-workflow-prompts.js`
- **Extend** `_meta/tools/test/print-workflow-prompts.test.js`

### Steps
- [ ] **Step 1 — RED.** Test that `capturePrompts({ workflowPath, args, agentReturns })` resolves to `Array<{ seq, label, phase, prompt }>`, that an `agentReturns` map is consulted before the schema generator, and that `--out` writes stable-key JSON (sorted keys, deterministic ordering).
- [ ] **Step 2 — GREEN.** Export async `capturePrompts({ workflowPath, args, agentReturns })` (wrapping `run-workflow.js`; returns the `calls` array). Add a CLI parsing `--workflow`, `--args`, **`--agent-returns <json>`**, `--out`. The CLI reads the args JSON and (if present) the agent-returns JSON, passes both to `capturePrompts`, then prints/writes stable-key JSON. `agentReturns` (a `label`→value map) is consulted **first**, the schema generator **second**.
- [ ] **Step 3 — Commit** (`P4 harness: print-workflow-prompts CLI (--agent-returns) + exported capturePrompts → stable-key JSON`).

## Task 4 — cuda + generate fixtures, agent-returns maps, golden captures, determinism/coverage test

### Files
- **Create** `_meta/tools/fixtures/accelopt-cuda-args.json`
- **Create** `_meta/tools/fixtures/accelopt-cuda-agent-returns.json`
- **Create** `_meta/tools/fixtures/accelopt-generate-args.json`
- **Create** `_meta/tools/fixtures/accelopt-generate-agent-returns.json`
- **Create** `_meta/tools/fixtures/accelopt-today.golden.json` (captured from PRE-retrofit AccelOpt, optimize mode)
- **Create** `_meta/tools/fixtures/accelopt-today-generate.golden.json` (captured from PRE-retrofit AccelOpt, generate mode)
- **Create** `_meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt`
- **Extend** `_meta/tools/test/print-workflow-prompts.test.js` (determinism + coverage)

### Steps

- [ ] **Step 1 — Write the fixed OPTIMIZE-mode cuda args fixture** (cuda legacy path: no `backend`, no `language`, no `backend_dir`; `kernel_path` set ⇒ `INPUT_MODE='optimize_existing'`). Keys must match what today's AccelOpt actually reads. `_meta/tools/fixtures/accelopt-cuda-args.json`:
```json
{
  "kernel_path": "/tmp/fixture/kernel.cu",
  "harness_path": "/tmp/fixture/harness.cu",
  "problem_path": "/tmp/fixture/problem.json",
  "iterations": 1,
  "breadth": 1,
  "samples_per_plan": 1,
  "substrate_command_prefix": "python3",
  "ncu_binary": "ncu",
  "harness_build_cmd": "nvcc -O3 -lineinfo",
  "kernel_name_regex": "my_kernel"
}
```

- [ ] **Step 2 — Write the OPTIMIZE-mode agent-returns map** that unlocks the full loop. Read the EXACT fields AccelOpt destructures (verified against the source): `read-baseline` is read for `kernel_code` (must be non-empty — feeds `baselineKernel`, and `impl.code===''` short-circuit is at L542 so `impl` must also be non-empty), `op_type`, `key_functions`; `ncu-baseline` for `latency_ms` (`.toFixed` at L406) plus the profile fields; the `impl-*` call for `code` (non-empty ⇒ passes L542); the `eval-*` call for `is_correct`/`is_compilable` (both must be `true` to pass L639) and `estimated_speedup` (`> MAX_THRESHOLD` 1.05 so ≥1 Learn pair fires at L702). **Keying:** the harness `agentReturns` is keyed by the `opts.label`. The `plan`/`impl`/`eval`/`learn` labels are *dynamic* (`plan-${iter}-${i}`, `impl-${iter}-${title}-v${sampleIdx}`, `eval-${variant.id}`, `learn-${plan_title}`). With `iterations:1, breadth:1, samples_per_plan:1` and `read-baseline.op_type='gemm'` the concrete labels are deterministic: `plan-0-0`, the impl label `impl-0-<plan.title[0:15]>-v0`, `eval-plan_0_sample_0`, `learn-<plan.title[0:20]>`. To avoid depending on the stubbed plan `title`, give the `plan-0-0` return an explicit short `title:"t"`, which fixes the impl label to `impl-0-t-v0`, the eval label to `eval-plan_0_sample_0`, and the learn label to `learn-t`. Write `_meta/tools/fixtures/accelopt-cuda-agent-returns.json`:
```json
{
  "read-baseline": {
    "kernel_code": "__global__ void k(){}",
    "op_type": "gemm",
    "key_functions": ["k"],
    "current_approach": "x"
  },
  "ncu-baseline": {
    "latency_ms": 1.0,
    "sm_throughput_pct": 50,
    "dram_throughput_pct": 50,
    "achieved_occupancy_pct": 50,
    "theoretical_occupancy_pct": 60,
    "waves_per_sm": 2,
    "registers_per_thread": 32,
    "top_stall_reason": "long_scoreboard",
    "top_stall_pct": 45,
    "sectors_per_request": 8,
    "l1_hit_rate_pct": 50,
    "l2_hit_rate_pct": 50,
    "ncu_rule_suggestions": ["s"],
    "bottleneck_diagnosis": "memory-latency-bound",
    "profile_summary": "summary",
    "ncu_available": true
  },
  "plan-0-0": {
    "title": "t",
    "focus_area": "memory",
    "ncu_evidence": "long_scoreboard 45%",
    "analysis": "a",
    "plan": "p",
    "expected_impact": "2x",
    "risk": "low"
  },
  "impl-0-t-v0": {
    "code": "__global__ void k_opt(){}",
    "implementation_notes": "n"
  },
  "eval-plan_0_sample_0": {
    "is_correct": true,
    "is_compilable": true,
    "estimated_latency_ms": 0.94,
    "estimated_speedup": 1.06,
    "correctness_issues": [],
    "ncu_comparison": "c",
    "bottleneck_addressed": true,
    "new_bottleneck": "none",
    "performance_analysis": "pa"
  },
  "learn-t": {
    "title": "rule",
    "ncu_trigger": "long_scoreboard high",
    "rule": "add ILP",
    "original_snippet": "a",
    "optimized_snippet": "b",
    "why": "hides latency",
    "is_antipattern": false
  }
}
```
  > **Reachable LABEL set under this map (OPTIMIZE mode):** `read-baseline`, `ncu-baseline`, `plan-0-0`, `impl-0-t-v0`, `eval-plan_0_sample_0`, `learn-t`, `final-report`. That is **7 labels** — and the count is *only* 7 because the map forces eval (non-empty impl.code + true correctness) and learn (speedup 1.06 > MAX_THRESHOLD). `generate-initial-kernel` is NOT in this set (kernel_path ⇒ optimize_existing). Do NOT assert a magic number; assert this exact label set (see Step 8).

- [ ] **Step 3 — Write the GENERATE-mode args fixture** (no `kernel_path` ⇒ `INPUT_MODE='generate_then_optimize'`; `problem_path` + test/benchmark commands so the seed branch's `verified` guard is satisfiable). `_meta/tools/fixtures/accelopt-generate-args.json`:
```json
{
  "problem_path": "/tmp/fixture/problem.json",
  "test_command": "python3 test.py",
  "benchmark_command": "python3 bench.py",
  "iterations": 1,
  "breadth": 1,
  "samples_per_plan": 1,
  "substrate_command_prefix": "python3",
  "ncu_binary": "ncu",
  "harness_build_cmd": "nvcc -O3 -lineinfo"
}
```

- [ ] **Step 4 — Write the GENERATE-mode agent-returns map.** Identical to the optimize map BUT adds `generate-initial-kernel`. The seed branch reads `generated_kernel_path` (non-empty, becomes `KERNEL_PATH`), `initial_candidates`, `initial_generation_result.verified` (must be `true` because `TEST_CMD||BENCH_CMD` is set → L199 throws if `verified===false`). After the seed, `read-baseline` still fires on the generated path. `_meta/tools/fixtures/accelopt-generate-agent-returns.json`:
```json
{
  "generate-initial-kernel": {
    "generated_kernel_path": "/tmp/fixture/generated/best.cu",
    "initial_candidates": [{ "id": "c0" }],
    "initial_generation_result": { "verified": true }
  },
  "read-baseline": {
    "kernel_code": "__global__ void k(){}",
    "op_type": "gemm",
    "key_functions": ["k"],
    "current_approach": "x"
  },
  "ncu-baseline": {
    "latency_ms": 1.0,
    "sm_throughput_pct": 50,
    "dram_throughput_pct": 50,
    "achieved_occupancy_pct": 50,
    "theoretical_occupancy_pct": 60,
    "waves_per_sm": 2,
    "registers_per_thread": 32,
    "top_stall_reason": "long_scoreboard",
    "top_stall_pct": 45,
    "sectors_per_request": 8,
    "l1_hit_rate_pct": 50,
    "l2_hit_rate_pct": 50,
    "ncu_rule_suggestions": ["s"],
    "bottleneck_diagnosis": "memory-latency-bound",
    "profile_summary": "summary",
    "ncu_available": true
  },
  "plan-0-0": {
    "title": "t",
    "focus_area": "memory",
    "ncu_evidence": "long_scoreboard 45%",
    "analysis": "a",
    "plan": "p",
    "expected_impact": "2x",
    "risk": "low"
  },
  "impl-0-t-v0": {
    "code": "__global__ void k_opt(){}",
    "implementation_notes": "n"
  },
  "eval-plan_0_sample_0": {
    "is_correct": true,
    "is_compilable": true,
    "estimated_latency_ms": 0.94,
    "estimated_speedup": 1.06,
    "correctness_issues": [],
    "ncu_comparison": "c",
    "bottleneck_addressed": true,
    "new_bottleneck": "none",
    "performance_analysis": "pa"
  },
  "learn-t": {
    "title": "rule",
    "ncu_trigger": "long_scoreboard high",
    "rule": "add ILP",
    "original_snippet": "a",
    "optimized_snippet": "b",
    "why": "hides latency",
    "is_antipattern": false
  }
}
```
  > **Reachable LABEL set under this map (GENERATE mode):** `generate-initial-kernel`, `read-baseline`, `ncu-baseline`, `plan-0-0`, `impl-0-t-v0`, `eval-plan_0_sample_0`, `learn-t`, `final-report` — **8 labels** (the seed seam S7 now renders).

- [ ] **Step 5 — Capture BOTH goldens from TODAY's AccelOpt (before any Stage B edit), from the repo root.** Run:
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node _meta/tools/print-workflow-prompts.js \
  --workflow AccelOpt/accelopt-kernel-optimization.js \
  --args _meta/tools/fixtures/accelopt-cuda-args.json \
  --agent-returns _meta/tools/fixtures/accelopt-cuda-agent-returns.json \
  --out _meta/tools/fixtures/accelopt-today.golden.json && \
node _meta/tools/print-workflow-prompts.js \
  --workflow AccelOpt/accelopt-kernel-optimization.js \
  --args _meta/tools/fixtures/accelopt-generate-args.json \
  --agent-returns _meta/tools/fixtures/accelopt-generate-agent-returns.json \
  --out _meta/tools/fixtures/accelopt-today-generate.golden.json
```
  Read each golden and CONFIRM the captured label sequence equals the enumerated set from Steps 2/4 (7 labels optimize; 8 labels generate) BEFORE freezing. If a label is missing, the agent-returns map failed to unlock that branch — fix the map, do not edit the expected count.

- [ ] **Step 6 — Emit the pre-retrofit baseline SHA.** Stage the goldens/fixtures, then write the *current* (pre-retrofit) HEAD SHA into a committed file so Stage C's re-baseline is self-contained:
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
git rev-parse HEAD > _meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt
```
  > This SHA is the tree the goldens were captured against. Stage C Task 12 greps for it.

- [ ] **Step 7 — Determinism + coverage test.** In `_meta/tools/test/print-workflow-prompts.test.js`:
  - **Determinism:** assert two consecutive `capturePrompts({ workflowPath, args: CUDA_ARGS, agentReturns: CUDA_RETURNS })` runs are deep-equal.
  - **Coverage (optimize):** assert the captured `label` set equals the explicit array `['read-baseline','ncu-baseline','plan-0-0','impl-0-t-v0','eval-plan_0_sample_0','learn-t','final-report']` (order-checked), and that the `phase` for each matches `Setup/Setup/Plan/Execute/Evaluate/Learn/Iterate`. Add a comment: `// The count (7) is a CONSEQUENCE of the agentReturns map unlocking eval (non-empty impl.code + is_correct/is_compilable true) and learn (estimated_speedup 1.06 > MAX_THRESHOLD 1.05). With empty stubs only 5 render. Do NOT hardcode 8.`
  - **Coverage (generate):** assert the generate golden's label set equals `['generate-initial-kernel','read-baseline','ncu-baseline','plan-0-0','impl-0-t-v0','eval-plan_0_sample_0','learn-t','final-report']`.

- [ ] **Step 8 — Run + commit.** Run from the repo root (file glob, NOT a bare directory):
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node --test _meta/tools/test/*.test.js
```
  Confirm green, then commit. **This is the re-baseline anchor commit; its SHA must equal the one already written to `GOLDEN-BASELINE-SHA.txt` in Step 6** (commit the SHA file alongside, then verify `git rev-parse HEAD` after committing matches — if `git rev-parse HEAD` differs because the commit moved HEAD, re-run Step 6 with the parent SHA the goldens were captured at; the goldens were captured against the working tree of the pre-retrofit commit, so use the commit that contains them but NO AccelOpt edits). Commit message: `P4 harness: cuda+generate args/agent-returns fixtures + frozen pre-retrofit goldens + GOLDEN-BASELINE-SHA + determinism/coverage tests`.

> **GATE for Stage B:** Stage B may not edit `AccelOpt/accelopt-kernel-optimization.js` until both goldens exist, `GOLDEN-BASELINE-SHA.txt` is committed, and the determinism/coverage test is green.

---

# Stage B — AccelOpt 21-Seam Retrofit (`backend_dir`-gated, cuda path byte-identical) (Part 2)

> **This Stage changes exactly one file:** `AccelOpt/accelopt-kernel-optimization.js`. No substrate `.py`, no new tools, no other workflow. The harness and its byte-identity tests are Stage C — this Stage produces the code that Stage C diffs.
>
> **THE IRON RULE (enforced by Stage C):** with **no `args.backend_dir`**, every rendered `agent()` prompt string, every schema object, every field read, and every return key must be **VERBATIM today's**. The retrofit is purely *additive*: a `USE_DRIVER` gate wrapping each NCU/CUDA seam as `USE_DRIVER ? <driver branch> : <legacy verbatim branch>`, where the legacy branch is the *exact* current code and `IDIOMS` defaults are seeded from today's literals so the legacy path reads identically.
>
> **Col-0 template literals — DO NOT inline-wrap (load-bearing).** The source prompts at L292-337 (ncu-baseline), L378-397 (baselineNcuProfile), L501-520 (execute), L562-602 (evaluate), L741-780 (learn), L820-851 (final-report) are template literals whose lines start at **column 0** — that leading whitespace is part of the string. Wrapping them in an `if/else` block or a ternary invites re-indentation, which changes the rendered bytes. **For each such legacy prompt, extract it VERBATIM into a named `const` at column 0** (e.g. `const LEGACY_NCU_PROMPT = \`...\``) BEFORE introducing the gate, confirm byte-identity, then reference it as the else branch: `USE_DRIVER ? driverPrompt : LEGACY_NCU_PROMPT`. The driver branch may be a normally-indented expression; only the legacy literal must stay at col 0.
>
> Because AccelOpt's body uses **no `Date`/`Math.random`/`performance.now`** (verified — `sampleWithoutReplacement` at L206-220 is a deterministic index-shuffle), fixed `args` + a fixed agent-returns map render the same prompt sequence every run. That is what makes the byte-identity diff (Stage C) meaningful.

### TDD discipline note for this Stage

Stage B is a **pure source refactor of a Workflow `.js` that cannot execute on this macOS host** (no node Workflow runtime, nvcc/ncu/triton absent). The executable red→green loop lives in **Stage C** (the harness + `node --test` byte-identity suite). Therefore each task here is gated on a Stage-C/Stage-A test:

- **Before editing any seam-group**, the Stage-A goldens must already exist (Stage A Task 4). That capture is a hard prerequisite.
- **Each task below ends** with: run the byte-identity check for the touched phase (created in Stage C Task 1; if running Stage B first, use the raw capture-vs-golden diff command below) → it must stay **green** (legacy branch unchanged). The "red" you avoid is *regressing* the golden. Commit only when green.

> **Note on ordering:** Stage C Task 1 creates the byte-identity test file. If you execute strictly Stage A → Stage B → Stage C, run the regression gate after each Stage-B task with the equivalent inline diff (run from the repo root, passing the SAME agent-returns map so the full loop renders):
> ```bash
> cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
> node _meta/tools/print-workflow-prompts.js --workflow AccelOpt/accelopt-kernel-optimization.js \
>   --args _meta/tools/fixtures/accelopt-cuda-args.json \
>   --agent-returns _meta/tools/fixtures/accelopt-cuda-agent-returns.json --out /tmp/retrofit.json && \
> diff _meta/tools/fixtures/accelopt-today.golden.json /tmp/retrofit.json && echo "BYTE-IDENTICAL (optimize)" && \
> node _meta/tools/print-workflow-prompts.js --workflow AccelOpt/accelopt-kernel-optimization.js \
>   --args _meta/tools/fixtures/accelopt-generate-args.json \
>   --agent-returns _meta/tools/fixtures/accelopt-generate-agent-returns.json --out /tmp/retrofit-gen.json && \
> diff _meta/tools/fixtures/accelopt-today-generate.golden.json /tmp/retrofit-gen.json && echo "BYTE-IDENTICAL (generate)"
> ```
> Either form is the regression gate.

---

## Task 5 — Path helpers, guard flip, new args, `USE_DRIVER` gate (S1, S2, S6 args)

Adds the §6.1 path helpers and the §6.4 guard split. **No prompt strings change yet** — this task only adds new top-level constants/functions and rewires the suitability guard. With no `args.backend`/`args.language`/`backend_dir`, `BACKEND` resolves to `'cuda'`, `USE_DRIVER` is `false`, and the guard passes exactly as the old whitelist did.

> **Banner correction (finding 15):** `IDIOMS.plan_angles` is a **byte-exact** copy of L421-425 and `IDIOMS.read_metric_guide` is a **byte-exact** copy of L447-457 (`.join('\n')`). `IDIOMS.impl_requirements` is a **driver-path-only paraphrase** of L513 (it drops the leading `1. ` numbering) — the legacy Execute prompt keeps the L513 literal verbatim, so impl_requirements is NEVER read on the cuda path. The L585 `#include` list (`cuda_runtime.h, cuda_fp16.h, ...`) is **NOT** captured into any IDIOMS constant; the legacy Evaluate prompt keeps it verbatim and the driver Evaluate prompt drops it in favor of `IDIOMS.impl_requirements`. Do not claim IDIOMS equals L513/L585.

### Files
- **Modify** `AccelOpt/accelopt-kernel-optimization.js`
  - Replace `WORKFLOW_SUITABILITY` (L15–20) with the §6.4 split.
  - Replace `assertWorkflowSuitability()` (L45–70) + its bare call (L72) with the §6.4 `resolveBackend()`/`assertWorkflowSuitability()` returning `BACKEND`.
  - Keep `normalizeSuitabilityValue` (L22–39) and `supportsSuitabilityValue` (L41–43) **byte-identical**.
  - Insert path helpers + new-arg consts + `USE_DRIVER` after L147 (after the existing param block).

### Steps

- [ ] **Step 1 — Confirm the Stage-A goldens exist and the determinism test is green.**
  Prerequisite gate: `node --test _meta/tools/test/*.test.js` (run from the repo root) passes against the unmodified AccelOpt, and both goldens + `GOLDEN-BASELINE-SHA.txt` are present. If absent, STOP and finish Stage A Task 4 first. Do not edit AccelOpt before the goldens capture today's prompts.

- [ ] **Step 2 — Replace `WORKFLOW_SUITABILITY` (S1).**

  BEFORE (L15–20):
  ```js
  const WORKFLOW_SUITABILITY = {
    supported_languages: ['cuda'],
    supported_problem_types: ['cuda-kernel-optimization', 'cuda-kernel-generation'],
    problem_types: ['existing CUDA kernel optimization', 'CUDA generation from problem_definition with benchmark contract'],
    reason: 'AccelOpt workflow is built around CUDA kernels, NCU metrics, CUDA harnesses, and CUDA-specific optimization patterns.',
  }
  ```

  AFTER:
  ```js
  const WORKFLOW_SUITABILITY = {
    // §6.4 split: method-support (JS-static) vs driver-presence+capability (agent-side, checked in Setup).
    method_supported_backends: ['cuda', 'triton'],      // vendor_locked to NCU-class profilers
    default_backend: 'cuda',                            // backward-compatible default
    requires_capability: { bottleneck_classes: [], metrics: ['dram_pct', 'sm_pct'] },
    // UNCHANGED keys (problem-type whitelist + reason) — normalizeSuitabilityValue still applies.
    supported_problem_types: ['cuda-kernel-optimization', 'cuda-kernel-generation'],
    problem_types: ['existing CUDA kernel optimization', 'CUDA generation from problem_definition with benchmark contract'],
    reason: 'AccelOpt is intrinsic to NCU-class profiling (dram_pct/sm_pct); runs on any NVIDIA-vendor backend (cuda, triton) with a present driver.',
  }
  ```

- [ ] **Step 3 — Replace `assertWorkflowSuitability()` + its call (S1 cont.).**

  BEFORE (L45–72): the whole `function assertWorkflowSuitability() { ... }` body (the `requestedLanguage`/`requestedProblemType` version) **plus** the bare `assertWorkflowSuitability()` call on L72.

  AFTER:
  ```js
  function resolveBackend() {
    const b = args.backend ? normalizeSuitabilityValue(args.backend) : null
    const l = args.language ? normalizeSuitabilityValue(args.language) : null
    if (b && l && b !== l) {
      throw new Error(`Conflicting args: backend="${b}" vs language="${l}". Pass only one.`)
    }
    if (b) return b
    if (l) return l                                   // legacy alias
    const ms = WORKFLOW_SUITABILITY.method_supported_backends
    if (Array.isArray(ms) && ms.length === 1) return normalizeSuitabilityValue(ms[0])
    return WORKFLOW_SUITABILITY.default_backend        // 'cuda'
  }

  function assertWorkflowSuitability() {               // SYNC — returns a string, no file/agent I/O
    const backend = resolveBackend()

    // (a) METHOD-support check (JS-only)
    const ms = WORKFLOW_SUITABILITY.method_supported_backends
    if (ms !== 'any' && !ms.map(normalizeSuitabilityValue).includes(backend)) {
      throw new Error(
        `${meta.name}'s method does not support backend="${backend}". ` +
        `Method-supported: ${ms.join(', ')}. Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }

    // (b) problem_type check — unchanged suffix-match + 'auto' sentinel.
    const requestedProblemType = normalizeSuitabilityValue(args.problem_type)
    if (requestedProblemType && requestedProblemType !== 'auto') {
      const supportedProblemTypes = (WORKFLOW_SUITABILITY.supported_problem_types || []).map(normalizeSuitabilityValue)
      if (supportedProblemTypes.length && !supportsSuitabilityValue(supportedProblemTypes, requestedProblemType)) {
        throw new Error(
          `${meta.name} is not suitable for problem_type="${args.problem_type}". ` +
          `Supported problem types: ${WORKFLOW_SUITABILITY.supported_problem_types.join(', ')}. ` +
          `Typical use cases: ${WORKFLOW_SUITABILITY.problem_types.join('; ')}. ` +
          `Reason: ${WORKFLOW_SUITABILITY.reason}`
        )
      }
    }

    // (c) driver presence + capability floor: deferred to the Setup 'load-driver' agent (only when USE_DRIVER).
    return backend
  }

  const BACKEND = assertWorkflowSuitability()          // sync; the ONLY BACKEND assignment
  ```

  > **Byte-identity note:** old `assertWorkflowSuitability()` *threw* on an unsupported `args.language`; the new path resolves `language` as a backend alias. With no `args.language` (the byte-identity test args), both old and new reach `BACKEND='cuda'` and the problem_type branch is untouched, so behavior is identical. The new conflicting-`backend`+`language` throw is a *new* arg combination today's callers never pass.

- [ ] **Step 4 — Add path helpers, new args, and `USE_DRIVER` (after L147).**
  Insert immediately after the AccelOpt-aligned-parameters block (after `const TOPK_LEARN = args.topk_learn || 5`, L147) and before the `// State` comment (L149):

  ```js
  // =============================================================================
  // Backend-driver wiring (§6.1) — additive; gated on backend_dir.
  // With NO backend_dir, USE_DRIVER === false and every seam takes the legacy path.
  // =============================================================================
  const SUBSTRATE = args.substrate_dir || '_substrate'
  const PY = args.substrate_command_prefix || ''       // python interpreter for .py only
  const SH = args.driver_shell_prefix || ''            // optional; '' → rely on shebang
  const BACKEND_DIR = args.backend_dir || ''           // '' → legacy inline-prompt path
  const DRIVER_DIR = BACKEND_DIR || `${SUBSTRATE}/backends/${BACKEND}`
  const USE_DRIVER = !!args.backend_dir                 // the single gate for every seam

  function substrateInstruction(script, cliArgs) {     // .py substrate scripts
    const p = `${SUBSTRATE}/${script}`
    return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
              : `No substrate_command_prefix for ${p} ${cliArgs}; do not invent an interpreter.`
  }
  function driverPy(script, cliArgs) {                  // to_evidence.py — python prefix
    const p = `${DRIVER_DIR}/${script}`
    return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
              : `No substrate_command_prefix for ${p}; do not invent an interpreter.`
  }
  function driverSh(script, cliArgs) {                  // build/run/profile — shebang, NO python
    return `Run exactly: \`${SH ? SH + ' ' : ''}${DRIVER_DIR}/${script} ${cliArgs}\`.`
  }
  const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }
  const DRIVER_EXT = '.json'                            // real P3 drivers ship manifest.json/idioms.json

  // IDIOMS defaults: plan_angles == today's L421-425 BYTE-EXACT; read_metric_guide == today's
  // L447-457 BYTE-EXACT. impl_requirements is a DRIVER-PATH-ONLY paraphrase of L513 (drops "1. ");
  // the legacy Execute prompt keeps the L513 literal verbatim, so impl_requirements is never read
  // on the cuda path. On the driver path the Setup load-driver agent (Task 6) overwrites IDIOMS
  // with idioms.json content.
  let IDIOMS = {
    lang_fence: 'cuda',
    impl_requirements:
      'Output a COMPLETE .cu file: all #includes, struct definitions, __global__ kernel(s), forward() wrapper, PYBIND11_MODULE',
    // S12 — planAngles (current L421-425 verbatim)
    plan_angles: [
      'memory latency hiding: address long_scoreboard stalls via ILP, prefetching, async copies, or software pipelining',
      'memory coalescing and vectorization: fix uncoalesced accesses (sectors/request > 4), use float4/int4 loads',
      'occupancy and parallelism: address SM idle time, tail effects, or low achieved occupancy',
      'compute restructuring: tensor core usage, warp-level reductions, reduced synchronization',
      'data layout and tiling: shared memory staging, bank-conflict-free layouts, double-buffering',
    ],
    // S13 — "How to read NCU data for planning" body (current L447-457 verbatim, lines only)
    read_metric_guide: [
      '- If top stall is "long_scoreboard" (>40%): kernel is MEMORY-LATENCY-BOUND. Add ILP, async loads, or data reuse.',
      '- If top stall is "short_scoreboard" (>30%): heavy shared-mem or dep chains. Shorten chains, add ILP.',
      '- If top stall is "barrier" (>20%): too much __syncthreads. Use warp-level primitives.',
      '- If top stall is "math_pipe_throttle": actually compute-bound — good! Look elsewhere.',
      '- If DRAM throughput > 80%: bandwidth-bound. Reduce bytes read (compression, shared-mem reuse).',
      '- If DRAM throughput < 10% AND long_scoreboard high: latency-bound on L1, not DRAM.',
      '- If sectors/request > 5: uncoalesced access — big optimization opportunity.',
      '- If achieved occupancy << theoretical: stalls prevent filling SM, fix stall source first.',
      '- If waves/SM < 1: grid too small, parallelize more or use persistent kernel.',
      '- If registers/thread > 128: likely register spill — add __launch_bounds__.',
      '- NCU rule suggestions with "Est. Speedup: X%" are surprisingly accurate — prioritize them.',
    ].join('\n'),
    unsupported_methods: [],
  }
  ```

  > **S6 (args) note:** the existing `HARNESS_*`/`KERNEL_NAME_REGEX`/`NCU_BINARY` consts (L120–124) are **unchanged**; per spec §8.2 they become the cuda driver's `--build-cmd`/`--ncu` inputs and flow through unchanged on the driver path (wired in Task 7). No edit to L120–124.

  > **S2 note:** `const LANGUAGE = args.language || 'cuda'` (L135) and `OP_DESC` (L114) are **left untouched** in this Stage. `LANGUAGE` is still read by the legacy generate-seed prompt (S7), which is byte-identical on the no-driver path. The driver path uses `BACKEND`/`IDIOMS.lang_fence` instead; the two coexist.

- [ ] **Step 5 — Regression gate.** Run the byte-identity diff (no-`backend_dir`, BOTH modes). MUST stay green (no prompt string changed in this task — only constants/guard added). Confirm before committing.

- [ ] **Step 6 — Commit.**
  ```
  git add AccelOpt/accelopt-kernel-optimization.js
  git commit -m "$(cat <<'EOF'
  AccelOpt P4: add backend-driver path helpers + §6.4 guard split (S1/S2/S6)

  Adds resolveBackend()/assertWorkflowSuitability() returning BACKEND,
  DRIVER_DIR/driverSh/driverPy/substrateInstruction helpers, USE_DRIVER gate,
  and IDIOMS defaults seeded from today's L421-425/L447-457 literals (plan_angles +
  read_metric_guide byte-exact; impl_requirements is a driver-only paraphrase of L513).
  No prompt strings change; no-backend_dir path is byte-identical (Stage-C test green).

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 6 — Setup load-driver agent + meta/phase/doc cosmetics (S3, S4, S5, Setup driver load)

Adds the §6.2 load-driver agent **only when `USE_DRIVER`**, and makes the meta/phase/header docs backend-neutral *without changing the legacy-rendered prompt strings*. S3/S4/S5 are `meta`/comments/`phase()` titles — they are **not** captured as `agent()` prompts by the harness, so they do not affect byte-identity. They are updated to neutral wording for honesty, but the *legacy* `phase('Setup')` etc. titles stay identical (the harness records `phase` on each captured call).

### Files
- **Modify** `AccelOpt/accelopt-kernel-optimization.js`
  - `meta.description`/`whenToUse` (L3–4) and `meta.phases[]` details (L6–11): keep `phase()` **call** titles identical; only soften the `detail` strings (cosmetic, not harness-captured).
  - Insert the load-driver agent after `phase('Setup')` (L252) and before the generate-seed branch (L254), gated on `USE_DRIVER`.

### Steps

- [ ] **Step 1 — S3/S4/S5 cosmetic neutralization (non-prompt text).**
  These are metadata/comments, never passed to `agent()`. Edit for honesty; the byte-identity test is unaffected because the harness captures only `agent()` prompts + their `{phase}`.

  - S3 (L3): change `description` to `'Self-improving kernel optimization loop with profiler-driven evidence (AccelOpt methodology; NCU on the cuda backend)'` and `whenToUse` (L4) to `'When you need to iteratively optimize a GPU kernel through plan-execute-profile-learn cycles. Uses the backend driver's profiler (e.g. Nsight Compute on cuda) for evidence-based bottleneck classification rather than guessing.'`
  - S4 (L6–11): soften each `detail` (e.g. `'Read target kernel, build via driver, profile baseline'`). **Do not** touch the `phase('Setup'|'Plan'|'Execute'|'Evaluate'|'Learn'|'Iterate')` *calls* later in the body — those titles are harness-recorded and must stay exactly as-is.
  - S5 (L90–96): header usage comment — add `backend`, `backend_dir`, `driver_shell_prefix`, `substrate_command_prefix`, `substrate_dir` to the documented args; keep `ncu_binary`/`harness_*` listed (still valid cuda inputs). Comment-only.

- [ ] **Step 2 — Insert the load-driver agent (gated on USE_DRIVER).**

  BEFORE (L252–256):
  ```js
  phase('Setup')

  if (INPUT_MODE === 'generate_then_optimize') {
    KERNEL_PATH = await resolveInitialKernelFromProblem()
  }
  ```

  AFTER:
  ```js
  phase('Setup')

  // §6.2 — load the backend driver ONLY on the driver path. On the legacy path this whole
  // block is skipped, so no extra agent() call is rendered and byte-identity holds.
  if (USE_DRIVER) {
    const driver = await agent(
      `Load the backend driver for backend="${BACKEND}".\n` +
      `1. Run exactly: \`cat ${DRIVER_DIR}/manifest${DRIVER_EXT}\` and parse JSON.\n` +
      `2. Run exactly: \`cat ${DRIVER_DIR}/idioms${DRIVER_EXT}\` and parse JSON.\n` +
      `If either is missing, return {present:false, reason:"no driver for backend ${BACKEND}"}.\n` +
      `Also compare manifest.capabilities against the required capability floor ` +
      `${JSON.stringify(WORKFLOW_SUITABILITY.requires_capability)};\n` +
      `if a required metric/class is missing return {present:true, capability_ok:false, missing:[...]}.\n` +
      `Return {present, capability_ok, missing, backend_id, source_ext, lang_fence, hw_vendor,\n` +
      `  profiler_name|null, profiler_format, capability_metrics, supported_classes, problem_types,\n` +
      `  requires_tools, impl_requirements, read_metric_guide,\n` +
      `  plan_angles:[...], unsupported_methods:[...],\n` +
      `  idioms:{<method>:{idiom,prompt_guidance}}}.`,
      { label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH })

    if (!driver.present) {
      throw new Error(`No backend driver present for backend="${BACKEND}". Provide ${DRIVER_DIR}/ or pick a supported backend.`)
    }
    if (driver.capability_ok === false) {
      throw new Error(`backend="${BACKEND}" lacks required capability: ${(driver.missing || []).join(', ')}.`)
    }
    // Overwrite IDIOMS defaults with driver-supplied idioms (lang_fence, impl_requirements,
    // plan_angles, read_metric_guide, unsupported_methods, per-method idiom/prompt_guidance).
    IDIOMS = {
      lang_fence: driver.lang_fence || IDIOMS.lang_fence,
      impl_requirements: driver.impl_requirements || IDIOMS.impl_requirements,
      plan_angles: (driver.plan_angles && driver.plan_angles.length) ? driver.plan_angles : IDIOMS.plan_angles,
      read_metric_guide: driver.read_metric_guide || IDIOMS.read_metric_guide,
      unsupported_methods: driver.unsupported_methods || [],
      profiler_name: driver.profiler_name || null,
      profiler_format: driver.profiler_format || '',
      source_ext: driver.source_ext || '.cu',
      ...(driver.idioms || {}),
    }
    log(`Driver loaded: ${BACKEND} (fence=${IDIOMS.lang_fence}, profiler=${IDIOMS.profiler_name || 'none'})`)
  }

  if (INPUT_MODE === 'generate_then_optimize') {
    KERNEL_PATH = await resolveInitialKernelFromProblem()
  }
  ```

  > **Driver-dir extension note (finding 19):** the real P3 triton driver ships `manifest.json` + `idioms.json` (verified — NOT `.yaml`). The example above already cats the `.json` form via `DRIVER_EXT`, so an executor copy-pasting it satisfies the Task 14 dry-run assertions (`/manifest\.json/`, `/idioms\.json/`). The byte-identity test never exercises this branch (it is `USE_DRIVER`-gated off), so the extension choice does not affect cuda byte-identity.

  > **Tricky byte-identity point:** the load-driver agent is an **extra `agent()` call** that the harness records. It is rendered **only when `USE_DRIVER`**. On the no-`backend_dir` run it is skipped entirely, so the captured prompt sequence is unchanged. Stage C's byte-identity diff runs with no `backend_dir`, so this branch is dead code under that test — exactly as required. The triton dry-run (Task 14) asserts the load-driver step is **absent** when no `backend_dir` is passed (gating check).

- [ ] **Step 3 — Regression gate.** Byte-identity diff (no-`backend_dir`, both modes) green (the new agent is gated off; S3/S4/S5 are not harness-captured).

- [ ] **Step 4 — Commit** (`AccelOpt P4: add Setup load-driver agent (cats manifest.json/idioms.json) + neutral meta/doc (S3/S4/S5)`; same Co-Authored-By trailer).

---

## Task 7 — Core NCU swap: profile.sh→to_evidence.py→diagnose.py, baseline schema + profile builder (S9, S10, S11)

The heart of the retrofit. On the **legacy path** the current `ncuSetup` agent (L292–337), its NCU schema (L343–361), and `baselineNcuProfile` (L378–397) are **verbatim**. On the **driver path** the prompt becomes a `driverSh('profile.sh')` + `driverPy('to_evidence.py')` call returning canonical metrics, the schema collapses to `JSON_PASSTHROUGH`, the profile string renders present keys only, and **new:** the metrics feed `diagnose.py` so AccelOpt gains `bottleneck_class`.

> **Col-0 extraction (finding 7):** extract the legacy ncu-baseline prompt VERBATIM into `const LEGACY_NCU_BASELINE_PROMPT` at column 0, and the legacy baseline-profile string into `const LEGACY_BASELINE_PROFILE` (a function of `ncuSetup`, since it interpolates). Confirm byte-identity, THEN gate.

### Files
- **Modify** `AccelOpt/accelopt-kernel-optimization.js`
  - Wrap `ncuSetup` agent (L292–362) in `USE_DRIVER ? <driver> : <legacy named const>`.
  - Add a `diagnose.py` agent + `let bottleneckClass` on the driver path.
  - Wrap `baselineNcuProfile` builder (L378–397) in the gate.

### Steps

- [ ] **Step 1 — Add a module-level `bottleneckClass` state var.**
  In the `// State` block (after L156 `candidateBeam = []`), add:
  ```js
  let bottleneckClass = 'unknown'   // populated by diagnose.py on the driver path; 'unknown' on legacy
  ```

- [ ] **Step 2 — Extract the legacy prompt to a col-0 named const, then gate the baseline-profile agent + schema (S9/S10).**

  First, hoist the legacy prompt VERBATIM (the entire current L292–337 string, every backtick and `${baselineKernel.substring(0, 4000)}`, ending with `Return a structured profile result.`) into a column-0 const ABOVE the gate:
  ```js
  const LEGACY_NCU_BASELINE_PROMPT = `You are a CUDA profiling expert using Nsight Compute (ncu). ... Return a structured profile result.`
  ```
  Confirm a byte-identity diff is still green after only the extraction (the agent now references `LEGACY_NCU_BASELINE_PROMPT`). THEN add the gate:
  ```js
  // Profile the baseline. Legacy path = today's hand-rolled NCU prompt (verbatim named const).
  // Driver path = profile.sh → to_evidence.py (canonical metrics) → diagnose.py (bottleneck_class).
  let ncuSetup
  if (USE_DRIVER) {
    const profileResult = await agent(
      IDIOMS.profiler_name
        ? `Profile the baseline kernel via the backend driver and normalize to canonical metrics.\n` +
          `Kernel: ${KERNEL_PATH}. Experiment dir: ${EXP_DIR}/baseline.\n` +
          `1. ` + driverSh('build.sh', `--source ${KERNEL_PATH} --out ${EXP_DIR}/baseline/artifact ${HARNESS_BUILD_CMD ? `--build-cmd "${HARNESS_BUILD_CMD}"` : ''}`) + `\n` +
          `2. ` + driverSh('profile.sh', `--artifact ${EXP_DIR}/baseline/artifact --problem ${PROBLEM_PATH || PROBLEM_DEFINITION || KERNEL_PATH} --out ${EXP_DIR}/baseline/prof.native`) + `\n` +
          `3. ` + driverPy('to_evidence.py', `--native ${EXP_DIR}/baseline/prof.native --format ${IDIOMS.profiler_format}`) + `\n` +
          `Return its stdout JSON verbatim: {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy,...}, coverage:[...], source_backend}. ` +
          `If the profiler exits 4 (unavailable), return {ok:true, metrics:{latency_ms:null,dram_pct:null,sm_pct:null,occupancy:null}, coverage:[], profiler_available:false}.`
        : `Backend "${BACKEND}" declares no profiler. Do not invent one. ` +
          `Build via ` + driverSh('build.sh', `--source ${KERNEL_PATH} --out ${EXP_DIR}/baseline/artifact`) + ` then ` +
          driverSh('run.sh', `--artifact ${EXP_DIR}/baseline/artifact --problem ${PROBLEM_PATH || KERNEL_PATH} --out ${EXP_DIR}/baseline/result.json`) + ` for latency only. ` +
          `Return {ok:true, metrics:{latency_ms:<from run.sh>,dram_pct:null,sm_pct:null,occupancy:null}, coverage:["latency_ms"], profiler_available:false}.`,
      { label: 'ncu-baseline', phase: 'Setup', schema: JSON_PASSTHROUGH })

    const metrics = profileResult.metrics || {}
    // Feed canonical metrics to diagnose.py → AccelOpt gains a bottleneck_class.
    const diag = await agent(
      `Write these metrics to ${EXP_DIR}/baseline/metrics.json:\n${JSON.stringify(metrics)}\n` +
      `${substrateInstruction('diagnose.py', `--metrics ${EXP_DIR}/baseline/metrics.json`)} Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { label: 'diagnose-baseline', phase: 'Setup', schema: JSON_PASSTHROUGH })
    bottleneckClass = diag.bottleneck_class || 'unknown'

    // Map canonical metrics onto the legacy ncuSetup field names so the rest of the loop is unchanged.
    ncuSetup = {
      latency_ms: metrics.latency_ms,
      sm_throughput_pct: metrics.sm_pct,
      dram_throughput_pct: metrics.dram_pct,
      achieved_occupancy_pct: (metrics.occupancy != null) ? metrics.occupancy * 100 : undefined,
      bottleneck_diagnosis: `${bottleneckClass}: ${(diag.evidence || []).join('; ')}`,
      profile_summary: profileResult.profiler_available === false
        ? `profiler unavailable; static analysis only (class=${bottleneckClass})`
        : `class=${bottleneckClass}, metrics=${JSON.stringify(metrics)}`,
      profile_evidence: diag.evidence || [],
      profiler_available: profileResult.profiler_available !== false,
      _metrics: metrics,
      _coverage: profileResult.coverage || [],
    }
  } else {
    ncuSetup = await agent(LEGACY_NCU_BASELINE_PROMPT, {
      label: 'ncu-baseline',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          latency_ms: { type: 'number' },
          sm_throughput_pct: { type: 'number' },
          dram_throughput_pct: { type: 'number' },
          achieved_occupancy_pct: { type: 'number' },
          theoretical_occupancy_pct: { type: 'number' },
          waves_per_sm: { type: 'number' },
          registers_per_thread: { type: 'number' },
          top_stall_reason: { type: 'string' },
          top_stall_pct: { type: 'number' },
          sectors_per_request: { type: 'number' },
          l1_hit_rate_pct: { type: 'number' },
          l2_hit_rate_pct: { type: 'number' },
          ncu_rule_suggestions: { type: 'array', items: { type: 'string' } },
          bottleneck_diagnosis: { type: 'string' },
          profile_summary: { type: 'string' },
          ncu_available: { type: 'boolean' },
        },
        required: ['latency_ms', 'bottleneck_diagnosis', 'profile_summary'],
      },
    })
  }
  ```

  > **CRITICAL byte-identity instruction:** `LEGACY_NCU_BASELINE_PROMPT` MUST be the **exact current L292–337 string** — every backtick, the `${baselineKernel.substring(0, 4000)}` interpolation, the `## Step 1..4` headings, the NCU metric list, all of it, at column 0. The schema object is the current L343–361 verbatim. **This is the single hardest seam to keep identical (S9 prompt vs driver call)** — the col-0 named-const extraction is what guarantees it.

- [ ] **Step 3 — Extract the legacy `baselineNcuProfile` builder to a col-0 const, then gate (S11).**

  First hoist the legacy template (current L378–397, leading newline + unindented `## NCU Profile Results (Baseline)` are PART of the string) verbatim into a col-0 helper:
  ```js
  function legacyBaselineProfile(ncuSetup) {
    return `
## NCU Profile Results (Baseline)
- Latency: ${ncuSetup.latency_ms} ms
- SM Throughput: ${ncuSetup.sm_throughput_pct || 'N/A'}% of peak
- DRAM Throughput: ${ncuSetup.dram_throughput_pct || 'N/A'}% of peak
- Achieved Occupancy: ${ncuSetup.achieved_occupancy_pct || 'N/A'}%
- Theoretical Occupancy: ${ncuSetup.theoretical_occupancy_pct || 'N/A'}%
- Waves/SM: ${ncuSetup.waves_per_sm || 'N/A'}
- Registers/Thread: ${ncuSetup.registers_per_thread || 'N/A'}
- Top Stall Reason: ${ncuSetup.top_stall_reason || 'N/A'} (${ncuSetup.top_stall_pct || 'N/A'}% of samples)
- Sectors/Request (global LD): ${ncuSetup.sectors_per_request || 'N/A'} (ideal=4)
- L1 Hit Rate: ${ncuSetup.l1_hit_rate_pct || 'N/A'}%
- L2 Hit Rate: ${ncuSetup.l2_hit_rate_pct || 'N/A'}%

## Bottleneck Diagnosis:
${ncuSetup.bottleneck_diagnosis}

## NCU Rule Suggestions:
${(ncuSetup.ncu_rule_suggestions || []).map(s => `- ${s}`).join('\n') || 'N/A'}
`
  }
  ```
  (The lines inside the template literal are at column 0 — preserve exactly; the indentation in this doc shows the function wrapper only, not the string.) Confirm byte-identity after the extraction (the assignment is now `baselineNcuProfile = legacyBaselineProfile(ncuSetup)`). THEN gate:
  ```js
  // Build the profile string. Legacy = today's NCU-shaped block (verbatim const). Driver = present-keys-only.
  if (USE_DRIVER) {
    const m = ncuSetup._metrics || {}
    const lines = []
    if (m.latency_ms != null) lines.push(`- Latency: ${m.latency_ms} ms`)
    if (m.sm_pct != null) lines.push(`- SM Throughput: ${m.sm_pct}% of peak`)
    if (m.dram_pct != null) lines.push(`- DRAM Throughput: ${m.dram_pct}% of peak`)
    if (m.occupancy != null) lines.push(`- Achieved Occupancy: ${(m.occupancy * 100).toFixed(1)}%`)
    baselineNcuProfile = `
## Profile Results (Baseline, backend=${BACKEND}, class=${bottleneckClass})
${lines.join('\n')}

## Bottleneck Diagnosis:
${ncuSetup.bottleneck_diagnosis}

## Evidence:
${(ncuSetup.profile_evidence || []).map(s => `- ${s}`).join('\n') || 'N/A'}
`
  } else {
    baselineNcuProfile = legacyBaselineProfile(ncuSetup)
  }
  ```

- [ ] **Step 4 — Regression gate.** Run the byte-identity diff (both modes). The no-`backend_dir` test renders the legacy `ncu-baseline` prompt + the legacy `baselineNcuProfile` (now flowing into the plan/eval/learn/final prompts the agent-returns map unlocks) and must be byte-identical. Confirm green.

- [ ] **Step 5 — Commit** (`AccelOpt P4: gate core NCU swap profile→to_evidence→diagnose (S9/S10/S11); legacy prompt extracted to col-0 const`).

---

## Task 8 — planAngles + how-to-read-NCU + plan schema (S12, S13, S14)

The plan-prompt's focus areas (L421–425), the NCU-reading causal block (L447–457), and the plan schema (L472–484). Legacy reads from `IDIOMS.plan_angles`/`IDIOMS.read_metric_guide` — **which default to today's literals** — so the rendered string is verbatim. The schema gains a neutral `profile_evidence` alias on the driver path only.

### Files
- **Modify** `AccelOpt/accelopt-kernel-optimization.js`
  - `const planAngles = [...]` (L420–426) → `const planAngles = IDIOMS.plan_angles`.
  - The `# How to read NCU data for planning:` block inside `planPromptBase` (L446–457) → interpolate `IDIOMS.read_metric_guide`.
  - The plan schema (L472–484) → gated; driver path adds `profile_evidence`.

### Steps

- [ ] **Step 1 — S12: replace the planAngles literal with the IDIOMS-sourced array.**

  BEFORE (L419–426):
  ```js
    // NCU-informed focus areas
    const planAngles = [
      'memory latency hiding: ...',
      ... (5 entries) ...
    ]
  ```
  AFTER:
  ```js
    // NCU-informed focus areas (legacy defaults == IDIOMS.plan_angles; driver overwrites IDIOMS in Setup)
    const planAngles = IDIOMS.plan_angles
  ```
  > Byte-identity holds because `IDIOMS.plan_angles` is initialized to the exact 5 strings from L421–425. `planAngles[i % planAngles.length]` (L469) is unchanged.

- [ ] **Step 2 — S13: interpolate `read_metric_guide` into `planPromptBase`.**

  BEFORE (L446–457): the literal 12-line block (`# How to read NCU data for planning:` header + 11 guide lines).
  AFTER (inside the `planPromptBase` template literal, replace those 12 lines with):
  ```
  # How to read NCU data for planning:
  ${IDIOMS.read_metric_guide}
  ```
  > `IDIOMS.read_metric_guide` defaults to the exact 11 lines `.join('\n')` (Task 5 Step 4). The header line `# How to read NCU data for planning:` + the following newline + the 11 lines reproduce L446–457 character-for-character. **Verify the interpolation introduces no extra blank line** — the guide string has no trailing newline, matching the original where line 457 is immediately followed by the blank line before `# Optimization Plan Requirements:` (keep that blank line in the template).

- [ ] **Step 3 — S14: gate the plan schema (add `profile_evidence` alias on driver path).**

  BEFORE (L472–484): the inline `schema: { type:'object', properties: { title, focus_area, ncu_evidence, analysis, plan, expected_impact, risk }, required: ['title','ncu_evidence','plan','expected_impact'] }`.

  AFTER — hoist a const just above the `parallel(...)` call (before L467) and reference it:
  ```js
    const planSchema = USE_DRIVER
      ? {
          type: 'object',
          properties: {
            title: { type: 'string' },
            focus_area: { type: 'string' },
            profile_evidence: { type: 'string' },   // neutral name (driver path)
            ncu_evidence: { type: 'string' },        // kept as optional alias
            analysis: { type: 'string' },
            plan: { type: 'string' },
            expected_impact: { type: 'string' },
            risk: { type: 'string' },
          },
          required: ['title', 'plan', 'expected_impact'],
        }
      : {
          type: 'object',
          properties: {
            title: { type: 'string' },
            focus_area: { type: 'string' },
            ncu_evidence: { type: 'string' },
            analysis: { type: 'string' },
            plan: { type: 'string' },
            expected_impact: { type: 'string' },
            risk: { type: 'string' },
          },
          required: ['title', 'ncu_evidence', 'plan', 'expected_impact'],
        }
  ```
  and change the agent call's `schema:` (L472) to `schema: planSchema`.

  > The `else` object is the **exact current L473–484** (same property order, same `required` array). Stage C compares schema objects structurally; the legacy branch is identical.

- [ ] **Step 4 — Add the neutral evidence read where plans are consumed.**
  Wherever `plan.ncu_evidence` is read downstream (L490 log, L509 execute prompt, L565 evaluate prompt, L718/L732 learn pairs), introduce a single helper just after `const validPlans = plans.filter(Boolean)` (L489):
  ```js
    const planEvidence = (p) => (p.profile_evidence ?? p.ncu_evidence)   // §8.2 neutral alias
  ```
  On the legacy path `profile_evidence` is `undefined`, so `?? p.ncu_evidence` returns exactly today's value — byte-identical. The execute/evaluate/learn prompt edits in Tasks 9–10 use `planEvidence(plan)` only on the driver path (legacy keeps the literal `${plan.ncu_evidence}` / `${variant.plan.ncu_evidence}`), so legacy strings are untouched.

- [ ] **Step 5 — Regression gate** (byte-identity diff, both modes, no-`backend_dir` green) and **commit** (`AccelOpt P4: gate planAngles/read-guide/plan-schema (S12/S13/S14)`).

---

## Task 9 — Fences + execute prompt + beam fence (S15, S16)

`buildBeamSection` fence (L246), the execute prompt's CUDA-specific wording (L501–520) including `lang_fence`, `impl_requirements`, and `-lineinfo`. Legacy verbatim; driver path uses `IDIOMS.lang_fence`/`IDIOMS.impl_requirements`.

> **Col-0 extraction (finding 7):** extract the legacy execute prompt VERBATIM into a col-0 builder function `legacyExecutePrompt(...)` (it interpolates `bestKernelCode`, `plan`, `sampleIdx`), confirm byte-identity, then reference as the else branch.

### Files
- **Modify** `AccelOpt/accelopt-kernel-optimization.js`
  - `buildBeamSection` (L244–247): gate the fence.
  - Execute prompt (L500–531): gate prompt + schema-field read.

### Steps

- [ ] **Step 1 — S15: gate `buildBeamSection`'s code fence.**

  BEFORE (L244–247):
  ```js
  function buildBeamSection(candidateBeam) {
    if (candidateBeam.length <= 1) return ''
    return `\n\n# Candidate Beam (top-${candidateBeam.length} kernels from previous iterations)\n${candidateBeam.map((c, i) => `## Candidate ${i + 1}: "${c.planTitle}" — ${c.speedup.toFixed(2)}x, ${c.latency.toFixed(3)}ms\nNCU: ${c.ncuSummary || 'N/A'}\n\`\`\`cuda\n${c.code.substring(0, 1500)}\n\`\`\``).join('\n\n')}`
  }
  ```
  AFTER — `buildBeamSection` is defined at L244 (hoisted) but only *called* at L417 (inside the loop, after Setup), where `IDIOMS.lang_fence` is in scope. Parameterize:
  ```js
  function buildBeamSection(candidateBeam, fence) {
    if (candidateBeam.length <= 1) return ''
    return `\n\n# Candidate Beam (top-${candidateBeam.length} kernels from previous iterations)\n${candidateBeam.map((c, i) => `## Candidate ${i + 1}: "${c.planTitle}" — ${c.speedup.toFixed(2)}x, ${c.latency.toFixed(3)}ms\nNCU: ${c.ncuSummary || 'N/A'}\n\`\`\`${fence}\n${c.code.substring(0, 1500)}\n\`\`\``).join('\n\n')}`
  }
  ```
  and at the call site (L417) change `const beamSection = buildBeamSection(candidateBeam)` → `const beamSection = buildBeamSection(candidateBeam, IDIOMS.lang_fence)`.
  > Legacy `IDIOMS.lang_fence === 'cuda'`, so `\`\`\`${fence}` renders ` ```cuda ` — identical. (With `breadth:1`, the beam has length 1 in iteration 0 so `buildBeamSection` returns `''` under the test fixtures; the parameterization is still exercised for byte-identity safety.)

- [ ] **Step 2 — S16: extract the legacy execute prompt to a col-0 builder, then gate.**

  Hoist the legacy prompt (exact current L501–520: CUDA developer, ` ```cuda `, the 6 numbered requirements including `5. MUST compile with -lineinfo`, `${plan.ncu_evidence}`) into a col-0 builder:
  ```js
  function legacyExecutePrompt(bestKernelCode, plan, sampleIdx, SAMPLES_PER_PLAN) {
    return `You are an expert CUDA kernel developer. ... Return the complete CUDA code.`
  }
  ```
  (The template literal lines stay at column 0.) Confirm byte-identity, THEN gate at the call site:
  ```js
        agent(USE_DRIVER
          ? `You are an expert ${BACKEND} kernel developer. Implement this profiler-informed optimization plan as a complete, compilable kernel.

  # Original Kernel:
  \`\`\`${IDIOMS.lang_fence}
  ${bestKernelCode.substring(0, 4000)}
  \`\`\`

  # Optimization Plan: "${plan.title}"
  Profiler Evidence: ${planEvidence(plan)}
  Plan: ${plan.plan}

  # Requirements:
  1. ${IDIOMS.impl_requirements}
  2. Must be FUNCTIONALLY CORRECT (same output as baseline within FP tolerance)
  3. Apply the plan faithfully — the plan is based on real profiler data, so the optimization targets a real bottleneck
  4. Keep the entrypoint signature unchanged
  5. This is variant ${sampleIdx + 1}/${SAMPLES_PER_PLAN}

  Return the complete ${BACKEND} code.`
          : legacyExecutePrompt(bestKernelCode, plan, sampleIdx, SAMPLES_PER_PLAN), {
            label: `impl-${iter}-${plan.title.substring(0, 15)}-v${sampleIdx}`,
            phase: 'Execute',
            schema: {
              type: 'object',
              properties: {
                code: { type: 'string' },
                implementation_notes: { type: 'string' },
              },
              required: ['code'],
            },
          })
  ```
  > The execute schema (L523–530) is shared (no NCU fields), so it stays a single object — unchanged. The legacy branch is now a named col-0 builder, removing the re-indentation temptation.

- [ ] **Step 3 — Regression gate** (byte-identity diff, both modes, no-`backend_dir` green) and **commit** (`AccelOpt P4: gate beam fence + execute prompt (S15/S16); legacy execute extracted to col-0 builder`).

---

## Task 10 — Evaluate + Learn prompts/schemas + generate-seed + setup-read (S7, S8, S17, S18, S19, S20) + in-loop NCU builder (S11b)

The remaining prompt/schema seams. All legacy branches verbatim (extracted to col-0 named consts/builders); driver branches swap fence/vocabulary and add neutral `profile_*` aliases (S18 `profile_comparison`, S20 `profile_trigger`). **Also gates the second, in-loop NCU profile builder at L673–681 (finding 14).**

### Files
- **Modify** `AccelOpt/accelopt-kernel-optimization.js`
  - Generate-seed prompt (L161–193, S7) — gate `language`→`BACKEND` + fence.
  - Setup read prompt (L258–285, S8) — gate `CUDA kernel`/`__global__`.
  - Evaluate prompt + schema (L562–620, S17/S18).
  - In-loop `baselineNcuProfile` reassignment (L673–681, S11b).
  - Learn prompt + schema (L741–796, S19/S20) + experience-format string (L803).

### Steps

- [ ] **Step 1 — S7: extract legacy generate-seed prompt to a col-0 const, then gate.**
  Hoist the exact current L161–181 prompt (uses `${LANGUAGE}`, "CUDA kernel candidates") into a col-0 const `LEGACY_GENERATE_SEED_PROMPT` (it interpolates `PROBLEM_DEFINITION/PROBLEM_PATH/OP_DESC/LANGUAGE/TARGET_GPU/SEED_CANDIDATES/TEST_CMD/BENCH_CMD/EXP_DIR`). Confirm byte-identity, then gate: `agent(USE_DRIVER ? <driver> : LEGACY_GENERATE_SEED_PROMPT, {...})`. Driver branch replaces `language: ${LANGUAGE}` → `backend: ${BACKEND}`, "CUDA kernel candidates" → `${BACKEND} kernel candidates`, and appends `Materialize complete kernels honoring: ${IDIOMS.impl_requirements}`. The schema (L184–192) is backend-neutral already (`generated_kernel_path`/`initial_candidates`/`initial_generation_result`) — leave it shared/unchanged. **This seam is exercised by the GENERATE-mode golden + gate.**

- [ ] **Step 2 — S8: extract legacy setup-read prompt to a col-0 const, then gate.**
  Hoist the exact current L258–268 (mentions "CUDA kernel", "`__global__` kernels") into `LEGACY_SETUP_READ_PROMPT`. Confirm byte-identity, then gate. Driver branch: "Read the CUDA kernel" → `Read the ${BACKEND} kernel source`, "`__global__` kernels" → "entry kernels (the backend's launch entrypoints)". Schema (L272–284) is neutral — shared/unchanged.

- [ ] **Step 3 — S17/S18: extract legacy evaluate prompt to a col-0 builder, then gate prompt + schema.**
  Hoist the exact current L562–602 prompt (Nsight Compute, `cuda_runtime.h`/`cuda_fp16.h` (L585), `__shared__`, sectors/request, `${ncuSetup.top_stall_reason}`, `${variant.plan.ncu_evidence}`) into a col-0 builder `legacyEvaluatePrompt(variant, bestLatency, ncuSetup)`. Confirm byte-identity, then gate. Driver branch swaps to `${BACKEND}` evaluator, ` ```${IDIOMS.lang_fence} `, generic correctness/compilability checks (drop the L585 CUDA-include list, instead reference `${IDIOMS.impl_requirements}`), `Bottleneck class: ${bottleneckClass}`, and `${planEvidence(variant.plan)}`.

  S18 schema (L605–619) — hoist `const evalSchema = USE_DRIVER ? {...with profile_comparison + ncu_comparison alias...} : {...current L606-619 verbatim...}` and set `schema: evalSchema`. Driver `required` keeps `['is_correct','is_compilable','estimated_speedup']`. Add a downstream read alias where `ncu_comparison` is consumed (L656, L672, L678, L719, L732): a helper `const evalProfile = (e) => (e.profile_comparison ?? e.ncu_comparison)` near L625, used **only** on the driver path; legacy keeps literal `r.evaluation.ncu_comparison`.

- [ ] **Step 4 — S11b: gate the in-loop NCU profile reassignment (L673–681, finding 14).**
  The second NCU-literal builder inside the Evaluate loop (`## NCU Profile Results (After Iteration ...)`, `- Bottleneck addressed`, `- Comparison: ${...ncu_comparison}`) emits vendor wording on the driver path. Extract the legacy literal VERBATIM into a col-0 builder `legacyIterProfile(iter, candidateBeam, bottleneckResult, bestLatency, baselineLatency, baselineNcuProfile)` and gate it:
  ```js
  if (bestResult && bestResult.evaluation.ncu_comparison) {
    baselineNcuProfile = USE_DRIVER
      ? `
## Profile Results (After Iteration ${iter + 1}, class=${bottleneckClass} — Best: "${candidateBeam[0].planTitle}")
- Latency: ${bestLatency}ms (${(baselineLatency / bestLatency).toFixed(2)}x speedup vs original)
- Bottleneck addressed: ${bestResult.evaluation.bottleneck_addressed ? 'YES' : 'NO'}
- New bottleneck: ${bestResult.evaluation.new_bottleneck || 'unknown'}
- Comparison: ${evalProfile(bestResult.evaluation)}

Previous profile data for reference:
${baselineNcuProfile}`
      : legacyIterProfile(iter, candidateBeam, bestResult, bestLatency, baselineLatency, baselineNcuProfile)
  }
  ```
  The legacy builder is the exact current L673–681 string (`## NCU Profile Results (After Iteration ...)` at col 0). Add `'NCU Profile Results'` to the triton dry-run banned-token list (Task 14) — now reachable because eval/iterate render under the agent-returns map.
  > Under `iterations:1, breadth:1` the test fixtures DO drive `bestResult.evaluation.ncu_comparison` non-empty (eval returns `ncu_comparison:'c'`) and `estimated_speedup 1.06` makes `candidateBeam[0].latency` improve, so this reassignment **renders** and its driver/legacy split is covered by both the byte-identity gate (legacy) and the triton dry-run (driver).

- [ ] **Step 5 — S19/S20: extract legacy learn prompt to a col-0 builder, then gate prompt + schema + experience-format.**
  Hoist the exact current L741–780 prompt (` ```cuda ` fences, the `**{Short title}** / NCU trigger: / Rule: / Original code: / Optimized code: / Why:` format) into a col-0 builder `legacyLearnPrompt(pair)`. Confirm byte-identity, then gate. Driver branch: `${BACKEND} optimization expert with profiler expertise`, ` ```${IDIOMS.lang_fence} `, `Profiler trigger:` instead of `NCU trigger:`, and reads `${pair.profile_evidence ?? pair.ncu_evidence}` / `${pair.profile_comparison ?? pair.ncu_comparison}`.

  S20 schema (L783–794) — hoist `const learnSchema = USE_DRIVER ? {...profile_trigger + ncu_trigger alias...} : {...current L784-794 verbatim with required ['title','ncu_trigger','rule','original_snippet','optimized_snippet','why']...}`.

  The experience-format string (L803) `**${s.title}**\nNCU trigger: ${s.ncu_trigger}\n...` — gate the literal `NCU trigger:`/read: on the driver path use `Profiler trigger: ${s.profile_trigger ?? s.ncu_trigger}`; legacy keeps `NCU trigger: ${s.ncu_trigger}` verbatim. The `pairsToSummarize` builders (L712–736) read `r.variant.plan.ncu_evidence`/`r.evaluation.ncu_comparison` — on the driver path use `planEvidence(...)`/`evalProfile(...)`; legacy unchanged.

- [ ] **Step 6 — Regression gate** (byte-identity diff, BOTH modes, no-`backend_dir` green across Setup/Plan/Execute/Evaluate/Learn/Iterate prompts — the agent-returns maps now render all of them) and **commit** (`AccelOpt P4: gate seed/read/evaluate/in-loop-profile/learn prompts+schemas (S7/S8/S11b/S17/S18/S19/S20)`).

---

## Task 11 — Final report + return: `baseline_profile` alias + Layer-A evidence envelope (S21)

Keep the existing `ncu_baseline_profile` return key (legacy unchanged). On the driver path: render a backend-neutral final-report prompt, add a `baseline_profile` alias, and an **assembly step** that builds a Layer-A INSIGHT envelope from `evidence_schema.py`'s template and validates it, so AccelOpt becomes L0/L2 substrate-conformant.

> **Col-0 extraction (finding 7):** extract the legacy final-report prompt VERBATIM into a col-0 builder `legacyFinalReportPrompt(...)`, confirm byte-identity, then gate.

### Files
- **Modify** `AccelOpt/accelopt-kernel-optimization.js`
  - Final-report agent (L820–854, S21): gate the prompt.
  - Assembly step (new, driver-only) before the `return`.
  - Return object (L856–877): add driver-only neutral aliases + `evidence`.

### Steps

- [ ] **Step 1 — S21a: extract legacy final-report prompt to a col-0 builder, then gate.**
  Hoist the exact current L820–851 prompt ("AccelOpt + NCU", the 6 NCU-themed report sections, ` ```cuda ` at L841) into a col-0 builder `legacyFinalReportPrompt(...)`. Confirm byte-identity, then gate `const finalReport = await agent(USE_DRIVER ? <driver> : legacyFinalReportPrompt(...), { label:'final-report', phase:'Iterate' })`. Driver branch: "AccelOpt + NCU" → `AccelOpt (${BACKEND} backend, profiler=${IDIOMS.profiler_name || 'none'})`, ` ```cuda ` (L841) → ` ```${IDIOMS.lang_fence} `, and the 6 report-section bullets reworded from "NCU metrics" to "profiler metrics / bottleneck_class". The agent opts (L852–853, `label:'final-report', phase:'Iterate'`, no schema) are unchanged.

- [ ] **Step 2 — S21b: driver-only Layer-A assembly + validate (new, before `return`).**
  Insert between L854 (`finalReport` close) and L856 (`return {`). **The insight items MUST match `_substrate/evidence_schema.py` `_validate_item` (finding 13):** each item needs `kind` ∈ {bottleneck,...}, `directive` ∈ {explore,avoid,constrain,reuse,gate}, `evidence` ∈ {ncu, profile_heuristic, ...}, `confidence` ∈ {measured, inferred, hypothesized}, and a non-empty **`claim`** (NOT `text`). Map `experienceMemory` entries to `claim`:
  ```js
  let evidenceEnvelope = null
  if (USE_DRIVER) {
    // §4.10 ASSEMBLY: build a Layer-A INSIGHT envelope from evidence_schema.py's template, then validate.
    const insightItems = experienceMemory.slice(0, TOPK_LEARN).map(e => ({
      kind: 'bottleneck',
      directive: 'explore',
      evidence: IDIOMS.profiler_name ? 'ncu' : 'profile_heuristic',
      confidence: 'inferred',
      claim: e,
    }))
    const built = await agent(
      `Build a Layer-A evidence envelope for this AccelOpt run, then validate it.\n` +
      `1. ${substrateInstruction('evidence_schema.py', 'template')} to get the envelope shape.\n` +
      `2. Fill it: attempt_id="accelopt-${BACKEND}", backend="${BACKEND}", ` +
      `compiled=true, correct=true, speedup=${(baselineLatency / bestLatency)}, ` +
      `metrics=${JSON.stringify(ncuSetup._metrics || {})}, ` +
      `bottleneck_class="${bottleneckClass}", ` +
      `insights=${JSON.stringify(insightItems)}.\n` +
      `Each insight item already has {kind, directive, evidence, confidence, claim} as required by ` +
      `evidence_schema.py _validate_item.\n` +
      `3. Write it to ${EXP_DIR}/evidence.json.\n` +
      `4. ${substrateInstruction('evidence_schema.py', `validate ${EXP_DIR}/evidence.json`)}\n` +
      `Return {valid, normalized} (the validator stdout JSON verbatim).`,
      { label: 'assemble-evidence', phase: 'Iterate', schema: JSON_PASSTHROUGH })
    evidenceEnvelope = built.valid ? (built.normalized || null) : null
    if (!built.valid) log(`WARN: Layer-A envelope failed evidence_schema validation`)
  }
  ```
  > `evidence_schema.py validate` takes a **positional path** (§4.10), not `--metrics` — note the positional `validate ${EXP_DIR}/evidence.json`. The insight item shape `{kind, directive, evidence, confidence, claim}` is what `_validate_item` requires; using `text` instead of `claim` would make every envelope fail validation (`evidenceEnvelope` always null) and defeat the L0/L2-conformance DoD claim.

- [ ] **Step 3 — S21c: return aliases (driver-only).**

  BEFORE (L856–877): the current `return { ..., ncu_baseline_profile: baselineNcuProfile, report: finalReport }`.

  AFTER — keep every current key **unchanged**; append driver-only fields via spread:
  ```js
  return {
    input_mode: INPUT_MODE,
    problem_definition: PROBLEM_DEFINITION,
    problem_path: PROBLEM_PATH,
    generated_kernel_path: generatedKernelPath,
    initial_candidates: initialCandidates,
    initial_generation_result: initialGenerationResult,
    baseline_latency_ms: baselineLatency,
    best_latency_ms: bestLatency,
    overall_speedup: baselineLatency / bestLatency,
    iterations_completed: ITERATIONS,
    candidate_beam: candidateBeam.map(c => ({
      plan_title: c.planTitle,
      latency_ms: c.latency,
      speedup: c.speedup,
    })),
    experience_patterns_count: experienceMemory.length,
    experience_patterns: experienceMemory,
    best_kernel_code: bestKernelCode,
    ncu_baseline_profile: baselineNcuProfile,          // UNCHANGED legacy key
    report: finalReport,
    ...(USE_DRIVER ? {
      backend: BACKEND,
      baseline_profile: baselineNcuProfile,            // §8.2 neutral alias
      bottleneck_class: bottleneckClass,
      evidence: evidenceEnvelope,                      // Layer-A envelope (null if validation failed)
    } : {}),
  }
  ```
  > On the no-`backend_dir` path `USE_DRIVER` is `false`, so the spread is `{}` — the returned object has the **exact current key set**. Stage C's return-key assertion (source-grep) passes.

- [ ] **Step 4 — Regression gate** (byte-identity diff, both modes, no-`backend_dir` final-report prompt + return keys byte-identical) and **commit** (`AccelOpt P4: gate final report + Layer-A return envelope with claim-shaped insights (S21)`).

---

## Closing invariant check (end of Stage B)

After Task 11, run the full Stage-C suite once more from the repo root: `node --test _meta/tools/test/*.test.js`. With **no `backend_dir`**, all captured `agent()` prompts (the enumerated 7-label optimize set and 8-label generate set), all schemas (source-grep), the beam evolution on the recorded fixtures, and the return-key set must equal the golden fixtures from Stage A — **byte-identical**. Only then is the retrofit complete. Every driver-path branch (`USE_DRIVER === true`) is exercised by Stage C's triton/driver test (Task 14), not by the byte-identity test.

---

# Stage C — Verification (no GPU): cuda byte-identity, guard units, triton dry-run (Part 3)

> **Branch:** `dev/solver-substrate`
> **Depends on:** Stage A (the harness `_meta/tools/print-workflow-prompts.js` + the captured goldens of *today's* AccelOpt for both modes + the committed agent-returns maps + `GOLDEN-BASELINE-SHA.txt`) and Stage B (the retrofit landed on `AccelOpt/accelopt-kernel-optimization.js`: §6.4 guard, `args.backend`/`args.backend_dir`/`args.language`/`args.driver_shell_prefix`, `IDIOMS` defaults where **plan_angles == L421-425 byte-exact** and **read_metric_guide == L447-457 byte-exact** (impl_requirements is a driver-only paraphrase of L513; L585 stays a legacy literal), `ncu_*`→`profile_*` schema aliases, `ncu_baseline_profile` return key + `baseline_profile` alias).
> **Goal of this Stage:** prove — on macOS with no GPU — that (1) the cuda legacy path renders **byte-identical** prompts to today's AccelOpt (both optimize and generate modes), (2) the guard resolves/throws per §6.4, and (3) the triton path renders the right *vendor-neutral* prompt wiring across the FULL rendered set (including Evaluate/Learn). Everything here runs under `node --test _meta/tools/test/*.test.js`.

### Environment facts this Stage is honest about

- macOS, no `nvcc`/`ncu`/`triton`; `node v24` + `python3` present. AccelOpt is a Workflow `.js` that runs only in the Workflow runtime — it **cannot execute here**. Every test verifies **rendered prompt strings**, never a real kernel run.
- AccelOpt's body is **deterministic** (no time/random calls). Given fixed `args` + a fixed `agentReturns` map, prompt strings render identically every run — and the agent-returns map is what unlocks the eval/learn/seed seams so they render at all.
- The triton driver under `_substrate/backends/triton/` ships as **`manifest.json` + `idioms.json`** (JSON, not YAML — verified). The Setup load-driver prompt and the triton dry-run assertions reflect `.json`, not `.yaml`.
- AccelOpt has **8 `agent()` call SITES** in source (lines 161, 258, 292, 469, 501, 562, 741, 820), but the number that RENDER depends on mode + agent-returns: **7 labels** in optimize mode (`read-baseline, ncu-baseline, plan-0-0, impl-0-t-v0, eval-plan_0_sample_0, learn-t, final-report` — no generate-seed) and **8 labels** in generate mode (add `generate-initial-kernel`). Plus runtime globals `phase`, `parallel`, `pipeline`, `log` (and `budget`, stubbed). The Stage-A harness stubs all of these. **Never assert `=== 8`; assert the explicit label set per mode.**

---

## Task 12 — CUDA byte-identity gate (the headline check)

With **no `backend_dir`**, the retrofitted AccelOpt must render prompts **byte-for-byte equal** to the Stage-A goldens captured from *today's* AccelOpt — for BOTH the optimize-mode and generate-mode fixtures, each driven by its committed agent-returns map.

### Files
- **Create** `_meta/tools/test/accelopt-cuda-byte-identity.test.js`
- **Reuse** `_meta/tools/print-workflow-prompts.js` (exports `capturePrompts`)
- **Reuse** `_meta/tools/fixtures/accelopt-cuda-args.json` + `accelopt-cuda-agent-returns.json` (Stage A)
- **Reuse** `_meta/tools/fixtures/accelopt-generate-args.json` + `accelopt-generate-agent-returns.json` (Stage A)
- **Reuse** `_meta/tools/fixtures/accelopt-today.golden.json` + `accelopt-today-generate.golden.json` (Stage A goldens — captured BEFORE the retrofit)
- **Reuse** `_meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt` (the pre-retrofit SHA)

> **Ordering dependency:** the goldens were captured from *today's* (pre-retrofit) AccelOpt and committed in Stage A, before Stage B edited the file. By the time this Stage runs, the working tree holds the **retrofitted** AccelOpt — you cannot re-derive "today" from `HEAD`. The pre-retrofit commit SHA lives in `GOLDEN-BASELINE-SHA.txt`. To re-baseline a legitimately-intended prompt change, see Step 5.

- [ ] **Step 1 — Confirm the harness contract.** Confirm `_meta/tools/print-workflow-prompts.js` exports async `capturePrompts({ workflowPath, args, agentReturns })` returning `{ seq, label, phase, prompt }[]` in call order. Run from the repo root:
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node -e "const {capturePrompts}=require('./_meta/tools/print-workflow-prompts.js'); console.log(typeof capturePrompts)"
```
Expected stdout: `function`. If `undefined`, fix the import path / export name before proceeding.

- [ ] **Step 2 — Confirm the fixed cuda args + agent-returns fixtures.** `accelopt-cuda-args.json` and `accelopt-cuda-agent-returns.json` (created in Stage A) are the SAME inputs fed to capture the optimize golden; likewise the `*-generate-*` pair for the generate golden. No `backend_dir`/`backend`/`language`, so the retrofit takes the legacy CUDA path. Confirm contents match Stage A Task 4.

- [ ] **Step 3 — Write the failing test.** Create `_meta/tools/test/accelopt-cuda-byte-identity.test.js`:
```js
'use strict'
// RE-BASELINE: the goldens (accelopt-today.golden.json, accelopt-today-generate.golden.json)
// were captured from PRE-RETROFIT AccelOpt at the commit recorded in
// _meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt. Do NOT regenerate them from the current
// (retrofitted) tree — that would make this gate tautological. To intentionally change a
// cuda prompt: (1) git worktree the SHA in GOLDEN-BASELINE-SHA.txt, (2) apply the SAME
// logical edit there, (3) re-run the capture with --agent-returns into the golden, (4) commit
// the new golden alone, explaining the intent.
// <RECORD Stage-A Task 4 commit SHA here — must equal GOLDEN-BASELINE-SHA.txt; Step 6 greps for it>
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))
const WORKFLOW = path.join(ROOT, 'AccelOpt/accelopt-kernel-optimization.js')

function load(f) { return JSON.parse(fs.readFileSync(path.join(ROOT, '_meta/tools/fixtures', f), 'utf8')) }

const CASES = [
  {
    name: 'optimize',
    args: load('accelopt-cuda-args.json'),
    agentReturns: load('accelopt-cuda-agent-returns.json'),
    golden: load('accelopt-today.golden.json'),
    expectedLabels: ['read-baseline', 'ncu-baseline', 'plan-0-0', 'impl-0-t-v0',
                     'eval-plan_0_sample_0', 'learn-t', 'final-report'],
  },
  {
    name: 'generate',
    args: load('accelopt-generate-args.json'),
    agentReturns: load('accelopt-generate-agent-returns.json'),
    golden: load('accelopt-today-generate.golden.json'),
    expectedLabels: ['generate-initial-kernel', 'read-baseline', 'ncu-baseline', 'plan-0-0',
                     'impl-0-t-v0', 'eval-plan_0_sample_0', 'learn-t', 'final-report'],
  },
]

for (const C of CASES) {
  test(`cuda legacy path (${C.name}): rendered label set is exactly the unlocked seams`, async () => {
    const captured = await capturePrompts({ workflowPath: WORKFLOW, args: C.args, agentReturns: C.agentReturns })
    assert.deepEqual(captured.map(c => c.label), C.expectedLabels,
      `label set drifted — the agentReturns map must unlock eval/learn(+seed). ` +
      `NOTE: count is NOT a magic 8; it is the enumerated set for ${C.name} mode.`)
  })

  test(`cuda legacy path (${C.name}): prompt sequence is byte-identical to today's AccelOpt`, async () => {
    const captured = await capturePrompts({ workflowPath: WORKFLOW, args: C.args, agentReturns: C.agentReturns })
    assert.equal(captured.length, C.golden.length,
      `agent() call count changed: today=${C.golden.length} retrofit=${captured.length}. If intended, re-baseline per Step 5.`)
    for (let i = 0; i < C.golden.length; i++) {
      const g = C.golden[i], c = captured[i]
      assert.equal(c.label, g.label, `seq ${i}: label drift ${g.label} -> ${c.label}`)
      assert.equal(c.phase, g.phase, `seq ${i} (${g.label}): phase drift ${g.phase} -> ${c.phase}`)
      if (c.prompt !== g.prompt) {
        let k = 0
        while (k < g.prompt.length && k < c.prompt.length && g.prompt[k] === c.prompt[k]) k++
        assert.fail(
          `seq ${i} (${g.label}/${g.phase}): prompt NOT byte-identical at offset ${k}.\n` +
          `  golden : ${JSON.stringify(g.prompt.slice(Math.max(0, k - 30), k + 30))}\n` +
          `  cuda   : ${JSON.stringify(c.prompt.slice(Math.max(0, k - 30), k + 30))}\n` +
          `If this change is intended, re-baseline per Step 5.`)
      }
    }
  })
}

test('cuda legacy path: ncu_* schema aliases still present in baseline schema', async () => {
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  for (const k of ['sm_throughput_pct', 'dram_throughput_pct', 'sectors_per_request', 'ncu_available']) {
    assert.match(src, new RegExp(`\\b${k}\\b`), `legacy ncu baseline-schema key "${k}" missing from retrofit source`)
  }
  for (const k of ['ncu_evidence', 'ncu_comparison', 'ncu_trigger']) {
    assert.match(src, new RegExp(`\\b${k}\\b`), `legacy schema alias "${k}" must be retained (optional) on the cuda path`)
  }
})

test('cuda legacy path: ncu_baseline_profile return key retained', async () => {
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  assert.match(src, /\bncu_baseline_profile\b/, 'return key ncu_baseline_profile must be preserved for back-compat')
  assert.match(src, /\bbaseline_profile\b/, 'neutral alias baseline_profile must be added alongside ncu_baseline_profile')
})

test('re-baseline SHA placeholder is filled (no unresolved <RECORD ...> marker)', () => {
  const me = fs.readFileSync(__filename, 'utf8')
  assert.ok(!me.includes('<RECORD Stage-A Task 4 commit SHA here'),
    'fill the <RECORD ...> placeholder with the SHA from GOLDEN-BASELINE-SHA.txt before committing')
  const sha = fs.readFileSync(path.join(ROOT, '_meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt'), 'utf8').trim()
  assert.ok(me.includes(sha), 'the test comment must cite the GOLDEN-BASELINE-SHA.txt commit SHA')
})
```

- [ ] **Step 4 — Run, see it fail for the RIGHT reason, then pass.** Run from the repo root:
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node --test _meta/tools/test/accelopt-cuda-byte-identity.test.js
```
TDD interpretation:
  - If Stage B's retrofit is correct, the byte-identity tests **pass immediately** (the cuda path is verbatim). That is the intended steady state — this gate is a *regression lock*, not a red-then-green feature.
  - To prove the gate actually bites, temporarily change one legacy cuda prompt literal in `AccelOpt/...js` (e.g. `Nsight Compute` → `Nsight Compute.`), re-run, and confirm the test reports the exact offset and `seq/label` for the eval/learn/iterate seam too (those now render because the agent-returns map unlocks them). Revert and confirm green. Record that you did this in the commit body.
  - The grep-based tests fail until Stage B retains the aliases — if they fail, the bug is in Stage B's S10/S14/S18/S20/S21 work, not here.

- [ ] **Step 5 — Fill + verify the re-baseline SHA.** Read `_meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt`, then replace the `<RECORD Stage-A Task 4 commit SHA here ...>` placeholder in the test's top comment with that exact SHA. The `re-baseline SHA placeholder is filled` test (Step 3) fails CI if the placeholder remains or the SHA is absent — so the SHA cannot be left unfilled.

- [ ] **Step 6 — Commit.**
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
git add _meta/tools/test/accelopt-cuda-byte-identity.test.js && \
git commit -m "$(cat <<'EOF'
test(accelopt): cuda byte-identity gate vs pre-retrofit goldens (optimize+generate) + alias/return-key checks

Locks the legacy (no backend_dir) cuda path for BOTH modes: rendered prompt sequence must
equal the Stage-A goldens byte-for-byte (driven by the committed agentReturns maps that unlock
eval/learn/seed); ncu_* schema aliases and ncu_baseline_profile return key must be retained;
the re-baseline SHA from GOLDEN-BASELINE-SHA.txt must be filled in. Verified the gate bites.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13 — Guard unit tests (`resolveBackend` / `assertWorkflowSuitability`)

Test the §6.4 guard directly: default `'cuda'`, the `language` alias, `triton` allowed, `metal` rejected (naming the supported set), and conflicting `backend`+`language` → throw.

### Files
- **Create** `_meta/tools/test/accelopt-guard.test.js`

> **Factoring decision (justified):** the guard (`resolveBackend`/`assertWorkflowSuitability`/`normalizeSuitabilityValue`/`WORKFLOW_SUITABILITY`) lives **inline** in the workflow body and references the runtime global `args` and the module-local `meta`. The body has **top-level `await`** and a **top-level `return`**, so the file cannot be `require()`d directly by node. Two options:
> - **(A) Exercise the guard *through the harness*** (run the body via the Stage-A sandbox with crafted `args`, assert it throws or that `BACKEND` resolves). Zero source change → **cannot break cuda byte-identity**.
> - **(B) Factor the guard into a tiny pure module** — but the body cannot `import`/`require` (runtime forbids it), so the functions would have to be **duplicated/inlined** anyway — net negative, and any divergence risks the byte-identity gate.
>
> **We choose (A).** It needs no source edit, so the cuda path stays provably byte-identical, and it tests the guard *as actually wired*.

- [ ] **Step 1 — Confirm the harness can surface guard outcome.** We need (a) the thrown `Error` when the guard rejects (so the body never reaches the first `agent()`), and (b) the resolved backend on success (inferred from the rendered prompts' vocabulary). Run from the repo root:
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node -e "const m=require('./_meta/tools/print-workflow-prompts.js'); console.log(Object.keys(m))"
```
The body's top-level `throw` propagates through the wrapping async IIFE (Task 2), so `capturePrompts` rejects on guard failure — assert via `assert.rejects`. For the success path, infer the backend from the prompt set using `caps.some(c => /CUDA/.test(c.prompt))` (NOT `caps[0]`, which couples to prompt ordering/wording).

- [ ] **Step 2 — Write the failing test.** Create `_meta/tools/test/accelopt-guard.test.js`:
```js
'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'AccelOpt/accelopt-kernel-optimization.js')

// Minimal args that let the body proceed far enough to resolve BACKEND without needing a real
// kernel. The harness stub-agent satisfies all schemas; we only care whether the guard throws
// BEFORE the first agent() call. (No backend_dir => USE_DRIVER false => legacy path renders.)
const baseArgs = {
  kernel_path: '/tmp/fixture/kernel.cu',
  problem_path: '/tmp/fixture/problem.json',
  iterations: 1, breadth: 1, samples_per_plan: 1,
  substrate_command_prefix: 'python3',
}

async function run(extra) {
  return capturePrompts({ workflowPath: WORKFLOW, args: { ...baseArgs, ...extra } })
}

test('no backend/language -> defaults to cuda (CUDA appears in some prompt)', async () => {
  const caps = await run({})
  assert.ok(caps.length > 0, 'expected at least one rendered prompt')
  assert.ok(caps.some(c => /CUDA/.test(c.prompt)), 'default cuda path should render CUDA vocabulary')
})

test('language:"cuda" alias resolves like backend:"cuda"', async () => {
  const caps = await run({ language: 'cuda' })
  assert.ok(caps.some(c => /CUDA/.test(c.prompt)))
})

test('backend:"triton" is method-supported (does NOT throw at the guard)', async () => {
  // method_supported_backends includes 'triton'. With NO backend_dir the body takes the legacy
  // prompt path but the guard must NOT reject triton. So the promise resolves rather than throwing.
  await assert.doesNotReject(run({ backend: 'triton' }), /does not support backend/)
})

test('backend:"metal" -> throws naming the supported set', async () => {
  await assert.rejects(run({ backend: 'metal' }), (err) => {
    assert.match(err.message, /backend="metal"/)
    assert.match(err.message, /cuda/)
    assert.match(err.message, /triton/)
    return true
  })
})

test('conflicting backend:"triton" + language:"cuda" -> throws', async () => {
  await assert.rejects(run({ backend: 'triton', language: 'cuda' }), /Conflicting args/)
})

test('hip alias normalizes (backend:"hip" -> rocm, rejected, names rocm not hip)', async () => {
  await assert.rejects(run({ backend: 'hip' }),
    (err) => { assert.match(err.message, /backend="rocm"/); return true })
})
```

- [ ] **Step 3 — Run and resolve.** Run from the repo root:
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node --test _meta/tools/test/accelopt-guard.test.js
```
  - `metal` / conflicting / `hip` cases should throw with the §6.4 messages — if not, the bug is in Stage B's guard (`resolveBackend`/`assertWorkflowSuitability`); fix there.
  - If `assert.rejects` does not fire because the harness **swallows** body throws (resolves with an empty capture instead of rejecting), that is a harness gap: amend the Stage-A harness to **propagate** body exceptions (do not catch-and-ignore around the wrapping async IIFE). Document the one-line harness fix in this Stage's commit.
  - The `triton`-does-not-throw and `CUDA`-appears assertions confirm the success path.

- [ ] **Step 4 — Commit.**
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
git add _meta/tools/test/accelopt-guard.test.js && \
git commit -m "$(cat <<'EOF'
test(accelopt): guard units via harness — default cuda, language alias, triton ok, metal/conflict/hip throw

Exercises resolveBackend/assertWorkflowSuitability as-wired (option A: through the print-prompts
harness, no source edit) so the cuda byte-identity gate stays provable. Backend inferred via
caps.some(c=>/CUDA/.test(c.prompt)), not caps[0], to decouple from prompt ordering.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14 — Triton dry-run (prompt-content check, no GPU)

Run the harness with `backend:'triton'` + the **real P3 driver dir** and an `agentReturns` map that returns the load-driver shape AND non-empty impl/eval so the **Evaluate and Learn prompts also render**, then assert the rendered prompts are vendor-neutral-for-triton: no CUDA tokens (across the FULL set incl. Evaluate/Learn/Iterate), the `python` lang fence, `@triton.jit`, and a Setup load-driver step. Also assert the load-driver step is **gated** (absent on no-backend_dir). This checks **PROMPT WIRING only**, not execution.

### Files
- **Create** `_meta/tools/fixtures/accelopt-triton-args.json`
- **Create** `_meta/tools/fixtures/triton/kernel.py` (tiny fake kernel so a `kernel_path` exists)
- **Create** `_meta/tools/test/accelopt-triton-dryrun.test.js`
- **Reuse** `_substrate/backends/triton/` (P3 driver — `manifest.json` + `idioms.json`, verified present)

> **Agent-returns note:** the harness consults `agentReturns[label]` first. To render the triton path with the python fence and idioms AND to unlock Evaluate/Learn, the map must return: `load-driver` → the §6.2 driver-load shape (`present:true, capability_ok:true, source_ext:'.py', lang_fence:'python', impl_requirements:'...@triton.jit...No PYBIND11...', idioms:{...}`); `read-baseline` → non-empty `kernel_code` (triton source, no CUDA tokens); `diagnose-baseline` → `{bottleneck_class, evidence}`; the ncu/profile call → `{metrics:{...}, ...}`; the impl label → `{code:'@triton.jit ...'}` (non-empty ⇒ passes L542); the eval label → `{is_correct:true,is_compilable:true,estimated_speedup:1.06,...}` (so Evaluate renders and ≥1 Learn pair fires). This tests that the body *threads driver fields into prompts*, not that a GPU ran anything.

- [ ] **Step 1 — Create the triton args + fake kernel.** Write `_meta/tools/fixtures/accelopt-triton-args.json`:
```json
{
  "backend": "triton",
  "backend_dir": "_substrate/backends/triton",
  "kernel_path": "_meta/tools/fixtures/triton/kernel.py",
  "problem_path": "/tmp/fixture/problem.json",
  "iterations": 1,
  "breadth": 1,
  "samples_per_plan": 1,
  "substrate_command_prefix": "python3"
}
```
And `_meta/tools/fixtures/triton/kernel.py`:
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)
```

- [ ] **Step 2 — Confirm the harness accepts an agentReturns map.** Run from the repo root:
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node -e "const {capturePrompts}=require('./_meta/tools/print-workflow-prompts.js'); console.log(capturePrompts.length)"
```
Use `{ workflowPath, args, agentReturns }` (the label→value map from the Stage-A contract). The override is consulted first, the schema generator second.

- [ ] **Step 3 — Write the failing test.** Create `_meta/tools/test/accelopt-triton-dryrun.test.js`:
```js
'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))
const WORKFLOW = path.join(ROOT, 'AccelOpt/accelopt-kernel-optimization.js')
const ARGS = JSON.parse(fs.readFileSync(path.join(ROOT, '_meta/tools/fixtures/accelopt-triton-args.json'), 'utf8'))

const triton = JSON.parse(fs.readFileSync(path.join(ROOT, '_substrate/backends/triton/idioms.json'), 'utf8'))
const tritonManifest = JSON.parse(fs.readFileSync(path.join(ROOT, '_substrate/backends/triton/manifest.json'), 'utf8'))

// agentReturns: load-driver shape + non-empty impl/eval so Evaluate/Learn ALSO render.
const agentReturns = {
  'load-driver': {
    present: true, capability_ok: true, missing: [],
    backend_id: 'triton',
    source_ext: tritonManifest.source_ext,          // ".py"
    lang_fence: triton.lang_fence,                   // "python"
    hw_vendor: tritonManifest.hw_vendor,
    profiler_name: tritonManifest.profiler.name,     // "ncu"
    profiler_format: tritonManifest.profiler.format,
    capability_metrics: tritonManifest.capabilities.metrics,
    supported_classes: tritonManifest.capabilities.bottleneck_classes,
    problem_types: tritonManifest.capabilities.problem_types,
    requires_tools: tritonManifest.requires_tools,
    impl_requirements: triton.impl_requirements,     // "...@triton.jit...No PYBIND11..."
    read_metric_guide: triton.read_metric_guide,
    idioms: triton.methods,
    unsupported_methods: triton.unsupported_methods || [],
  },
  'read-baseline': { kernel_code: '@triton.jit\ndef k():\n    pass', op_type: 'gemm', key_functions: ['k'], current_approach: 'x' },
  'ncu-baseline': { ok: true, metrics: { latency_ms: 1.0, dram_pct: 50, sm_pct: 50, occupancy: 0.5 }, coverage: ['latency_ms', 'dram_pct', 'sm_pct', 'occupancy'], profiler_available: true },
  'diagnose-baseline': { bottleneck_class: 'memory_bound', evidence: ['dram 50% high'] },
  'plan-0-0': { title: 't', focus_area: 'memory', profile_evidence: 'dram high', plan: 'p', expected_impact: '2x' },
  'impl-0-t-v0': { code: '@triton.jit\ndef k_opt():\n    pass', implementation_notes: 'n' },
  'eval-plan_0_sample_0': { is_correct: true, is_compilable: true, estimated_latency_ms: 0.94, estimated_speedup: 1.06, profile_comparison: 'c', bottleneck_addressed: true, new_bottleneck: 'none', performance_analysis: 'pa' },
  'learn-t': { title: 'rule', profile_trigger: 'dram high', rule: 'tile', original_snippet: 'a', optimized_snippet: 'b', why: 'reuse' },
  'assemble-evidence': { valid: true, normalized: {} },
}

async function tritonPrompts() {
  return capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns })
}

test('triton dry-run: a Setup load-driver step is present and cats the .json driver files', async () => {
  const caps = await tritonPrompts()
  const loader = caps.find(c => c.label === 'load-driver')
  assert.ok(loader, 'expected a load-driver agent call in Setup')
  assert.equal(loader.phase, 'Setup')
  assert.match(loader.prompt, /backends\/triton\/manifest\.json/, 'load-driver must cat the triton manifest.json (not .yaml)')
  assert.match(loader.prompt, /idioms\.json/, 'load-driver must cat idioms.json')
})

test('triton dry-run: Evaluate and Learn prompts DO render (agentReturns unlocked the loop)', async () => {
  const caps = await tritonPrompts()
  const labels = caps.map(c => c.label)
  assert.ok(labels.includes('eval-plan_0_sample_0'), 'Evaluate prompt must render')
  assert.ok(labels.includes('learn-t'), 'Learn prompt must render')
})

test('triton dry-run: NO CUDA-only tokens leak into ANY prompt (incl. Evaluate/Learn/Iterate)', async () => {
  const caps = await tritonPrompts()
  const all = caps.map(c => c.prompt).join('\n----\n')
  for (const banned of ['__global__', 'PYBIND11_MODULE', 'NCU Profile Results', 'cuda_fp16.h', 'cuda_runtime.h']) {
    assert.ok(!all.includes(banned), `triton prompts must not contain "${banned}"`)
  }
  assert.ok(!/\.cu\b/.test(all), 'triton prompts must not reference .cu sources')
  assert.ok(!/```cuda/.test(all), 'triton prompts must not open a ```cuda fence')
})

test('triton dry-run: uses the python lang fence and triton ABI requirements', async () => {
  const caps = await tritonPrompts()
  const all = caps.map(c => c.prompt).join('\n----\n')
  assert.match(all, /```python/, 'triton path must use the python code fence')
  assert.match(all, /@triton\.jit/, 'executor prompt must carry triton impl requirements')
})

test('triton dry-run: load-driver is GATED — absent when no backend_dir is passed', async () => {
  const { backend_dir, ...noDir } = ARGS
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: noDir, agentReturns })
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'load-driver must NOT render without backend_dir (USE_DRIVER gate)')
})

test('triton dry-run: this checks PROMPT WIRING, not execution (documented)', () => {
  // The stub-agent returned the driver-load + impl/eval shapes; no build.sh/run.sh/ncu ran.
  // Real end-to-end triton verification is the deferred GPU tier (DEFERRED-GPU-VERIFICATION.md).
  assert.ok(true)
})
```

- [ ] **Step 4 — Run and resolve.** Run from the repo root:
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node --test _meta/tools/test/accelopt-triton-dryrun.test.js
```
  - Failures on the "NO CUDA tokens" / "python fence" assertions over the FULL set (now incl. Evaluate/Learn/Iterate, since the agent-returns map unlocks them) point at Stage B seam work that did not gate on `backend_dir`/`lang_fence` (S7/S8/S11b/S15/S16/S17/S19/S9/S13) — fix in the retrofit, not here.
  - If `@triton.jit` is absent because the executor prompt does not inject `driver.impl_requirements`, that is the S16 swap — fix in Stage B.
  - The `manifest.json`/`idioms.json` assertions catch the `.yaml` slip; the gating test catches a load-driver that fires without `backend_dir`.

- [ ] **Step 5 — Commit.**
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
git add _meta/tools/fixtures/accelopt-triton-args.json \
        _meta/tools/fixtures/triton/kernel.py \
        _meta/tools/test/accelopt-triton-dryrun.test.js && \
git commit -m "$(cat <<'EOF'
test(accelopt): triton dry-run prompt-wiring check across the FULL rendered set (no GPU)

With the real P3 triton driver dir + a fake kernel + an agentReturns map that unlocks
Evaluate/Learn: asserts no __global__/.cu/PYBIND11/NCU-Profile-Results/cuda-include leak,
the python lang fence, @triton.jit impl_requirements, a Setup load-driver step that cats
manifest.json/idioms.json, AND that load-driver is gated (absent without backend_dir).
Checks prompt wiring only, not execution.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Definition of Done

- [ ] **Step 1 — Run the full no-GPU suite green (file glob, NEVER a bare directory).**
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
node --test _meta/tools/test/*.test.js
```
Expected: `print-workflow-prompts`, `accelopt-cuda-byte-identity`, `accelopt-guard`, and `accelopt-triton-dryrun` all pass; `0` failures. (Do NOT run `node --test _meta/tools/test/` — node v24 treats the bare directory as a test file and emits a spurious failing line.)

- [ ] **Step 2 — Confirm the DoD mapping to spec §8.**

| Spec ref | Claim | Proven by |
|---|---|---|
| §8.3a | cuda legacy path renders **byte-identical** prompts to today's AccelOpt (optimize + generate modes) | Task 12 byte-identity gate vs Stage-A goldens (full loop unlocked by agentReturns) |
| §8.2 (S10/S14/S18/S20/S21) | `ncu_*` schema aliases + `ncu_baseline_profile` return key retained | Task 12 alias/return-key tests |
| §6.4 | guard: default `cuda`; `language` alias; `triton` allowed; `metal`/conflict/`hip` throw with the supported-set message | Task 13 guard units |
| §8.3b (prompt half) | triton path: no `__global__`/`.cu`/`PYBIND11`/`NCU Profile Results`, python fence, triton ABI, gated Setup load-driver — across Evaluate/Learn too | Task 14 triton dry-run |
| §8.2 (S2 `bottleneck_class`) | `diagnose.py` invocation is WIRED into the driver-path prompt (Task 14); the resolved bottleneck_class value is GPU-tier | Task 14 wiring; live class deferred |
| §8.3b (execution half) | build/run/ncu/diagnose/Layer-A end-to-end | Deferred to NVIDIA box (below) |

- [ ] **Step 3 — State the headline invariant.** The cuda path is **provably unchanged** (byte-identical, regression-locked by Task 12 across both modes), while the SAME workflow file now: (a) honors `args.backend`/`args.backend_dir` dispatch (Tasks 13/14), (b) renders vendor-neutral prompts for triton driven entirely by `idioms.json` (Task 14, full loop), and (c) wires `diagnose.py` for a `bottleneck_class` on the driver path (Task 14 wiring; live class in the GPU tier). No GPU was required to prove the retrofit's correctness for the cuda contract and the triton prompt wiring.

- [ ] **Step 4 — Final commit.**
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
git commit --allow-empty -m "$(cat <<'EOF'
chore(accelopt): P4 verification DoD — cuda byte-identical (both modes), triton wiring proven (no GPU)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Deferred verification (GPU/CI tier)

Everything above is no-GPU prompt verification. The following **require a real NVIDIA box** and are out of scope for this macOS environment — they belong to a self-hosted-runner CI tier (spec §9.3 "real-hardware tier opt-in").

- [ ] **Step 1 — Write the deferred checklist.** Create `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md`:
```markdown
# Deferred GPU verification — AccelOpt triton end-to-end (spec §8.3b, §9.3 real-hardware tier)

These steps CANNOT run on macOS (no nvcc/ncu/triton). They require a self-hosted
NVIDIA runner. The no-GPU tests prove prompt WIRING; this tier proves EXECUTION.

## Requires a real NVIDIA box with:
- NVIDIA GPU + driver
- `python3` + `triton` installed (JIT warmup materializes PTX into TRITON_CACHE_DIR)
- `ncu` (Nsight Compute) on PATH, with permission to profile (perf-counter access)
- the substrate scripts runnable: `diagnose.py`, `method_gate.py`, `evidence_schema.py`

## End-to-end triton run (the §8.3b "Pass" conditions)
Invoke AccelOpt with:
`backend:'triton', backend_dir:'_substrate/backends/triton',
 kernel_path:<a real small .py triton kernel>, problem_path:<real problem.json>,
 substrate_command_prefix:'python3', iterations:1, breadth:1`

Assert (all on real hardware):
1. No `__global__`/`.cu` in any rendered prompt (already covered no-GPU; re-confirm live).
2. `build.sh` (JIT warmup) returns `{ ok:true, compiled:true }`.
3. `run.sh` returns `{ compiled:true, correct:true }` (torch.allclose + do_bench).
4. `profile.sh` + `ncu` attributed a **non-empty CSV** to the mangled
   `triton_<fn>_<hash>` kernel — proving kernel-name auto-discovery from
   TRITON_CACHE_DIR, not merely that to_evidence.py parsed a file (§5.1 caveat).
5. `to_evidence.py` emits canonical metrics with occupancy ∈ [0,1] (the ÷100 rule).
6. `diagnose.py --metrics <file>` returns a class in the diagnose.py taxonomy
   (memory_bound|compute_bound|latency_occupancy|overhead_bound, or unknown when no
   dominant signal) — AccelOpt gains a `bottleneck_class` it has none of today.
7. The Layer-A assembly step builds an INSIGHT envelope (each insight shaped
   {kind, directive, evidence, confidence, claim}) that `evidence_schema.py validate <path>`
   accepts (AccelOpt becomes L0/L2 conformant).

## Promotion gate
Only after 1-7 pass on real hardware: promote the triton driver manifest
`status: experimental -> stable`, and record AccelOpt as the first
matrix-eligible (partial) workflow emitting a Layer-A envelope (spec §9.3 matrix
smoke test seed).

## What this tier does NOT cover (still deferred beyond it)
- Metal / non-NVIDIA (needs P6: Apple threshold calibration + MPS fallback patterns).
- Performance-comparable cross-backend scoring (§10 open).
```

- [ ] **Step 2 — Commit.**
```bash
cd /Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows && \
git add _meta/tools/test/DEFERRED-GPU-VERIFICATION.md && \
git commit -m "$(cat <<'EOF'
docs(accelopt): deferred GPU-tier verification checklist for triton end-to-end

Lists the §8.3b execution checks that need a real NVIDIA box (build/run compiled+correct,
ncu-attributed CSV proving kernel-name discovery, diagnose.py 4-class taxonomy + unknown,
Layer-A envelope with {kind,directive,evidence,confidence,claim} insights) plus the driver
status-promotion gate. Out of scope on macOS.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
