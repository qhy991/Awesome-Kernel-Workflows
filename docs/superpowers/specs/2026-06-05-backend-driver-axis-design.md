# Backend Driver Axis — Design Spec

> Status: design approved (brainstorming), pre-implementation. Produced via a survey →
> design → adversarial-review workflow; this revision folds in 9 blocker + 11 major
> review findings (every code claim below is verified against the live `_substrate/`
> scripts and workflow files). Branch context: `dev/solver-substrate`.

---

## 1. Summary

Today every kernel-optimization workflow in this repo has its **backend welded into its
body**: `AccelOpt/accelopt-kernel-optimization.js` names `nvcc`, `ncu`, `__global__`,
`sm_throughput_pct`, and `.cu` directly in prompts and schemas; `KernelAgent/...js`
hardcodes `@triton.jit`/`tl.load`; `CutlassGEMM` bakes in `GemmUniversal`/`StreamK`. The
optimization **method** (beam, MCTS, evolutionary, multi-agent) is tangled with the
**backend**, so retargeting a proven method to a new chip means rewriting the workflow.

The thesis: **a backend is not one axis but a product of two —
`backend = (source language) × (hardware / profiler vendor)` — and that product is
orthogonal to the optimization method.** The goal is a **pluggable Backend Driver** that
adapts native backend tooling to the universal metric/method vocabulary, so the repo can
run *any clean method × any backend* as a matrix: swapping a backend becomes setting
`args.backend`, not re-typing three shell strings and re-authoring a dozen prompts.

The driver is a **translation layer** that sits between a workflow and the shared
substrate. **It is NOT true that the substrate stays byte-frozen** — the honest invariant
(see §3.2, §5.3, §9.1) is: **the substrate is frozen *except three scoped, default-`nvidia`,
golden-tested edits*** — two hunks in `diagnose.py` (a vendor threshold profile *and*
null-metric handling) and one new optional argument on `anti_cheat.py`. `method_gate.py`,
`evidence_schema.py`, `memory_store.py`, and `verify_insight.py` stay byte-identical. Any
spec text that earlier implied "zero substrate change" is wrong and is corrected here.

---

## 2. Background & current state

### 2.1 The three axes

A kernel workflow already decomposes along three orthogonal axes; this spec makes the
third explicit.

1. **Universal substrate** — six Python scripts under `_substrate/`, invoked *by an agent
   via Bash* (never imported by the `.js` body): `evidence_schema.py` (Layer A evidence
   envelope), `anti_cheat.py` (Layer B reward/honesty), `diagnose.py` (Layer C bottleneck
   classification), `memory_store.py` (Layer D memory), `method_gate.py` (Layer E method
   gating), `verify_insight.py`.
2. **Exploration topology** — the method's search shape (iterative / search / tree /
   pipeline), realized from a template and *authored by an LLM agent* in the generator
   (see §9.2 — the generator is not a substitution engine).
3. **Backend** — the `(language) × (vendor)` product. The silent, welded axis this spec
   extracts.

### 2.2 The existing seams we extend (not replace)

The backend is already *almost* a seam, in five places (all verified):

- **`args.language` polymorphism.** Several workflows already select the backend at
  invocation: `const LANGUAGE = args.language || 'cuda'` (`AccelOpt/...js:135`). The `.js`
  body is language-agnostic; only embedded prompt vocabulary is not.
- **`WORKFLOW_SUITABILITY` + `assertWorkflowSuitability()`** (lines ~15–72 of each
  workflow): a constant declaring `supported_languages`/`supported_problem_types`/
  `problem_types`/`reason` (four keys, verified in KDA/CUDAAgent), guarded by a throw.
  `normalizeSuitabilityValue()` already maps vendor aliases (`hip→rocm`, `intel-xpu→xpu`,
  `c++→cpp`, `cute→cute-dsl`) and applies `.replace(/_/g,'-')`. `supportsSuitabilityValue()`
  does suffix-matching (`cuda-kernel-optimization` matches `kernel-optimization`) and an
  `auto` sentinel short-circuits. We reuse all of this.
- **Substrate-as-scripts.** Substrate ops run through the `substrateInstruction()`
  Bash-from-agent pattern (`Generalist/...js:142`): the body builds a command *string*, an
  agent runs it via Bash and returns stdout JSON, and the body threads the parsed result on.
- **Caller-provided command discipline.** Workflows accept
  `compile_command`/`test_command`/`benchmark_command`/`ncu_command`/`profile_command`
  templates and never invent a fallback compiler (`AccelOpt/...js:315-319`). The generator
  *agent prompt* tells the model not to hardcode such commands (`generate-workflow.js:282`
  lists `ncu_command` as a **permitted user-supplied arg**, not a forbidden token); the
  template comment (`iterative-loop.js:70-75`) says "do not hardcode evaluator/compiler/
  profiler commands" without naming `nvcc`/`ncu`.
- **Two vendor-locked substrate scripts.** `diagnose.py` keys on
  `metrics.{dram_pct, sm_pct, occupancy}` (NCU counter projections — *vendor*-level, not
  *language*-level) with NVIDIA-tuned literal thresholds; `method_gate.py`'s `TABLE` lists
  NVIDIA SM idioms (`tensor_core_mma`, `async_copy_pipeline`, `shared_memory_tiling`);
  `anti_cheat.py`'s `FALLBACK_PATTERNS` (lines 19–26) only match `cublas`/`cudnn`/
  `torch.matmul`/`F.linear`/`torch.nn.functional`/`at::matmul`. The other three scripts are
  already neutral.

The Backend Driver is therefore an **extension of an existing caller-provides-commands
discipline**, formalized into a versioned, per-backend, conformance-tested directory.

> **Why "vendor", not "language", is the deep sub-axis.** `dram_pct`/`sm_pct`/`occupancy`,
> `tensor_core_mma`, and the `cublas`/`cudnn` patterns are all *vendor*-level concepts with
> nothing to do with whether the source was `.cu` or `.py`. This is why CUDA and Triton
> collapse onto one vendor for the **classifier-counter layer** (§5.1).

### 2.3 Tooling-tree note (pre-existing duplication)

`generate-workflow.js` and `validate-workflow.js` exist in **both** `_tools/` and
`_meta/tools/`; the four templates exist in **both** `_templates/` and `_meta/templates/`.
`SOLVER-SDK.md:117` names `_meta/tools/validate-workflow.js` as the conformance checker, so
**`_meta/` is canonical.** All new tooling (`validate_backend.py`, `_schema/`) and all edits
to the generator/validator/templates/manifest-schema land in the `_meta/` tree. The plan
**must not deepen** the `_tools/`↔`_meta/tools/` divergence; deduping the two trees is
out of scope here but flagged.

---

## 3. Architecture

### 3.1 Layering

```
                     ┌─────────────────────────────────────────────┐
   args.backend ───► │   method .js (topology: loop/search/tree)   │   ← optimization method
                     │   - no fs / no import / no time / no random  │     (orthogonal to backend)
                     │   - builds STRING paths from args.backend    │
                     └──────────────────┬──────────────────────────┘
                                        │ agent Bash (cat / run)
                     ┌──────────────────▼──────────────────────────┐
                     │   BACKEND DRIVER  _substrate/backends/<id>/  │   ← (language) × (vendor)
                     │   manifest.yaml  build.sh  run.sh            │     TRANSLATION LAYER
                     │   profile.sh  to_evidence.py  idioms.yaml    │
                     └──────────────────┬──────────────────────────┘
                                        │ canonical metrics dict + abstract method names
                     ┌──────────────────▼──────────────────────────┐
                     │   SUBSTRATE (frozen except 3 scoped edits)   │   ← shared by all backends
                     │   evidence_schema · anti_cheat* · diagnose** │   * +1 optional arg
                     │   method_gate · memory_store · verify_insight│   ** +2 golden-tested hunks
                     └─────────────────────────────────────────────┘
```

Per-attempt pipeline (only the first four files are backend-specific):

```
build.sh → run.sh → profile.sh → to_evidence.py → diagnose.py → method_gate.py → idiom-translate → (assemble) → evidence_schema.py validate
```

### 3.2 The driver-as-translation-layer principle

The driver is the adapter between native backend tools and the universal metric/method
vocabulary. All backend variation that *can* live in the driver lives in **two driver
files**: `to_evidence.py` (native profiler output → canonical `metrics` dict) and
`idioms.yaml` (abstract `method_gate.TABLE` names → concrete backend idioms, applied
downstream in the prompt). `method_gate.TABLE` is **never** parameterized by `--backend`
inside the script — the abstract list flows out unchanged and idiom translation happens in
the planner prompt.

But two backend concerns **cannot** be pushed entirely into the driver and force scoped
substrate edits (enumerated in §5.3, golden-tested in §9.1):

- **`diagnose.py` — null-vs-zero + vendor thresholds.** `diagnose.py:24` is `d = dram or 0.0`,
  which coerces a `None` (unmeasured) DRAM% into a real `0.0`, then routes
  `dram<40 and sm<40 → overhead_bound`. So a null-metric backend is given a *confident wrong
  label*, not `unknown` (see §5.2/§5.3). And the thresholds are NVIDIA-calibrated.
- **`anti_cheat.py` — per-vendor cheat patterns (BOTH lists).** Two module-level constants are
  Python/CUDA-flavored: `FALLBACK_PATTERNS` (lines 19–26: `cublas`/`cudnn`/`torch.matmul`/… —
  Metal's `MPSMatrixMultiplication`/etc. are invisible) **and `SKIP_PATTERNS`** (lines 28–33:
  `return\s+input`, `#\s*TODO`, `raise\s+NotImplementedError`, `^\s*pass\s*$` — none of which
  appear in Metal C++ `.metal`/`.mm`, where the equivalents are `// TODO`, `return;`, no
  `pass`/`NotImplementedError`). So a Metal kernel that secretly delegates to MPS **or** stubs
  out compute passes the L1 "honest" gate. Both lists need the per-vendor mechanism (§5.3.3).

These are real, named, bounded substrate changes — not "zero change". `method_gate.py`,
`evidence_schema.py`, `memory_store.py`, `verify_insight.py` stay byte-identical — **with one
caveat:** `method_gate.gate()` has a *dormant* NVIDIA-tuned cutoff (`method_gate.py:32` drops
`occupancy_increase` when `occupancy >= 0.8`). It is inert only because §6.3 invokes
`method_gate.py --class <bclass>` **without `--metrics`**. Workflows MUST keep calling the gate
without `--metrics` (so it stays neutral); if any migrated workflow needs metrics-aware gating,
that `0.8` must be folded into the vendor threshold profile (§5.3.1), making it a fourth scoped
edit. Until then, `method_gate.py` is byte-identical and neutral.

---

## 4. The Backend Driver contract

### 4.1 Directory layout

A driver is a directory under `_substrate/backends/<backend_id>/`
(`backend_id ∈ {cuda, triton, rocm, metal, …}`, the post-`normalizeSuitabilityValue`
canonical form). No `backends/` directory exists today (verified) — net-new scaffolding.

```
_substrate/
├── evidence_schema.py        # FROZEN
├── anti_cheat.py             # FROZEN body + ONE new optional arg (§5.3.3)
├── diagnose.py               # FROZEN classifier + TWO golden-tested hunks (§5.3.1/.2)
├── method_gate.py            # FROZEN
├── memory_store.py           # FROZEN
├── verify_insight.py         # FROZEN
└── backends/
    ├── REGISTRY.md
    ├── _schema/
    │   ├── manifest.schema.json
    │   └── idioms.schema.json
    ├── cuda/   { manifest.yaml build.sh run.sh profile.sh to_evidence.py idioms.yaml }
    ├── triton/ { … to_evidence.py is the nvidia mapping shared with cuda … }
    └── metal/  { … to_evidence.py is a SEPARATE file … }
```

`validate_backend.py` (in `_substrate/backends/`) validates each driver against the contract.

### 4.2 Role and IO of each file

| File | Role | Read by | Determinism |
|---|---|---|---|
| `manifest.yaml` | Static identity, extensions, toolchain, **capabilities**. The single source of truth a workflow reads to know "does this backend exist & what can it do". | agent `cat` (YAML → prompt) + offline validators | static |
| `build.sh` | Compile/JIT source → artifact. **Executable shell script** (shebang), invoked with NO Python prefix. | agent Bash | deterministic given inputs |
| `run.sh` | Artifact → correctness + latencies, in `anti_cheat.py` key shape (§4.5). Executable shell. | agent Bash | timings noisy; JSON shape deterministic |
| `profile.sh` | Artifact → **native** profiler output; emits a *pointer* to the raw file. Executable shell. | agent Bash | run-to-run noisy; opaque to substrate |
| `to_evidence.py` | **Neutral interface.** Native profiler output → canonical `metrics` dict. **Python**, invoked WITH the python prefix. | agent Bash | pure function |
| `idioms.yaml` | **Method translation table.** Abstract `method_gate.TABLE` name → concrete backend idiom + prompt guidance. | agent `cat` → prompt | static |

> **Invocation prefix split (blocker fix).** `args.substrate_command_prefix` is the *Python
> interpreter* (e.g. `python3`) for the `.py` substrate scripts **and** for `to_evidence.py`.
> It must **never** be prepended to a `.sh` driver. `build.sh`/`run.sh`/`profile.sh` are
> executable (shebang `#!/usr/bin/env bash`) and invoked as `${DRIVER_DIR}/build.sh …` with
> no interpreter, or via an optional separate `args.driver_shell_prefix`. See §6.1.

`manifest.compiler.invoke`/`runner.invoke`/`profiler.invoke` are **informational** (they
document the entrypoint filename); the workflow body uses the fixed filenames
`build.sh`/`run.sh`/`profile.sh`/`to_evidence.py` directly. `validate_backend.py` asserts the
declared `invoke` values equal those fixed names (no divergence allowed).

### 4.3 Universal envelope rule (every executable)

- **stdout** is a **single JSON object** on success; logs/diagnostics go to **stderr only**.
- **Exit code** is the in-band signal; a `"ok"` field mirrors it.
- Reserved codes: `0` success · `2` operation error (well-formed JSON still printed) · `3`
  bad args / missing tool (preflight) · `4` **profiler unavailable** (degrade to
  `bottleneck_class: unknown` deterministically) · never `1` (reserve for uncaught).

### 4.4 `manifest.yaml` schema

```yaml
schema_version: 1
backend_id: cuda                 # REQUIRED. == normalizeSuitabilityValue(id) == dir name. dispatch key.
display_name: "NVIDIA CUDA"      # REQUIRED.
source_ext: ".cu"                # REQUIRED. primary kernel source extension.
aux_ext: [".cuh", ".h"]          # OPTIONAL.
artifact_ext: ".so"              # REQUIRED.
hw_vendor: nvidia                # REQUIRED. enum: nvidia | amd | intel | apple | cpu | generic.
threshold_profile: nvidia        # REQUIRED. diagnose.py profile key (default nvidia → CUDA/Triton unchanged).

compiler: { name: nvcc, invoke: "build.sh" }       # REQUIRED.
runner:   { invoke: "run.sh" }                     # REQUIRED.
profiler:                                          # OPTIONAL. omit → no native profiler → unknown.
  name: ncu                                        # ncu | torch.profiler | rocprof | metal-counters
  invoke: "profile.sh"
  format: "ncu-csv"                                # to_evidence --format selector
  to_evidence: "to_evidence.py"

capabilities:                                      # REQUIRED.
  metrics: { dram_pct: true, sm_pct: true, occupancy: true, latency_ms: true }  # which canonical keys honestly populated
  bottleneck_classes: [memory_bound, compute_bound, latency_occupancy, overhead_bound]  # MEANINGFUL classes (subset of the 4 non-unknown; unknown always implicitly allowed)
  problem_types: [cuda-kernel-optimization, cuda-kernel-generation]
  precisions: [fp32, fp16, bf16, tf32]             # OPTIONAL.

requires_tools: [nvcc, python3]                    # REQUIRED. preflight: must resolve.
optional_tools: [ncu]                              # OPTIONAL. absence degrades gracefully.
vendor_patterns_file: "anti_cheat_patterns.txt"    # OPTIONAL. per-vendor [fallback]+[skip] regexes (§5.3.3); absent → CUDA defaults.
idioms: "idioms.yaml"                              # REQUIRED.
status: stable                                     # REQUIRED. enum: stable | experimental | stub.
```

> **Enum note:** `diagnose.py`'s `CLASSES` is a **5-class enum including `unknown`**.
> `capabilities.bottleneck_classes` lists only the **4 meaningful** classes; `unknown` is
> always implicitly allowed. `validate_backend.py`'s subset check targets "the 4 ∪ {unknown}".

### 4.5 Per-file CLI contracts

**`build.sh`** — `build.sh --source <path> --out <artifact> [--arch <gpu_arch>] [--build-cmd <tmpl>] [--extra <flags>]`
- success: `{ "ok": true, "compiled": true, "artifact": "<path>", "build_latency_ms": 4210.0, "stderr_tail": "" }`;
  failure: `compiled:false, artifact:null, stderr_tail:"<last ≤40 lines>"`.
- Codes: `0` compiled · `2` compile error · `3` missing tool / bad args.
- `build_latency_ms` is wall-time measured **inside the script** (the only permitted
  non-determinism — it never touches the JS body, so the no-time/no-random rule holds).
  *Named `build_latency_ms`, NOT `compile_latency_ms`* — see the run.sh note below.

**`run.sh`** — `run.sh --artifact <path> --problem <problem.json> --out <result.json> [--reps N] [--rtol f] [--atol f] [--baseline eager|compile|both]`
- stdout — **exactly the keys `anti_cheat.py` reads** (verified: `compiled`, `correct`,
  `candidate_latency_ms`, `eager_latency_ms`, `compile_latency_ms`, `claimed_speedup`):
  ```json
  { "ok": true, "compiled": true, "correct": true,
    "candidate_latency_ms": 0.41, "eager_latency_ms": 0.78, "compile_latency_ms": 0.55,
    "claimed_speedup": 1.90, "max_abs_err": 3.1e-4, "max_rel_err": 8.0e-6, "reps": 50 }
  ```
  Also written to `--out` so `anti_cheat.py --metrics -` reads it directly.
- Codes: `0` ran (incorrectness is data, not error) · `2` runtime crash · `3` bad args.
- **Wire-format reconciliation (blocker fix).** `anti_cheat.robust_reward` uses
  `compile_latency_ms` as the **torch.compile baseline** latency for `beats_compile`, and
  reads `claimed_speedup` (not `speedup`). Therefore: (a) `run.sh`'s *JIT/build* time is
  `build_latency_ms` (in `build.sh`), and `compile_latency_ms` in `run.sh` is reserved for
  the **torch.compile baseline** — the two are different quantities and must not collide on
  one key; (b) the speedup key is `claimed_speedup`. A backend with no `torch.compile`
  equivalent (e.g. Metal/MPS) emits `compile_latency_ms: null` → `beats_compile = (comp is not
  None) and (cand < comp)` evaluates to `False` (not `None`) → reward caps at 2 (vendor-aware
  reward is an open risk, §10).
- **Honesty contract:** if `correct:false`, `run.sh` MUST emit `claimed_speedup ≤ 1.0` (so
  `evidence_schema.py`'s "incorrect cannot claim speedup>1.0" rule passes downstream).
  Latency uses device-side clocks (CUDA events / `triton.testing.do_bench` / Metal
  `GPUEndTime − GPUStartTime`), not wall clock.

**`profile.sh`** — `profile.sh --artifact <path> --problem <problem.json> --out <native_profile.<ext>>`
- stdout is a **pointer**: `{ "ok": true, "profiler": "ncu", "native_profile": "<path>", "format": "ncu-csv" }`.
  Raw blob → `--out`; native formats stay opaque to the substrate; all parsing is confined
  to `to_evidence.py`.
- Codes: `0` profiled · `4` profiler unavailable (degrade to `unknown`) · `3` bad args.
- Absent `manifest.profiler` ⇒ `profile.sh` absent ⇒ workflow skips this step.

### 4.6 `to_evidence.py` — the neutral interface (keystone)

`to_evidence.py --native <profile.<ext>|-> --format <ncu-csv|torch-json|rocprof-csv|metal-json> [--run <result.json>] [--problem <problem.json>]`

stdout — the canonical metrics dict `diagnose.py` consumes verbatim, **plus a vendor tag**:
```json
{ "ok": true,
  "metrics": {
    "latency_ms": 0.41, "dram_pct": 62.0, "sm_pct": 48.0, "occupancy": 0.51,
    "_vendor": "nvidia",
    "backend_native": { "l2_hit_pct": 73.0, "sectors_per_req": 4.1 }
  },
  "source_backend": "cuda",
  "coverage": ["dram_pct","sm_pct","occupancy","latency_ms"] }
```
- The three classifier keys live at the **top level of `metrics`** so the post-edit
  `diagnose.classify()` reads them unchanged.
- `_vendor` selects the `diagnose.py` threshold profile (§5.3.1); default/absent ⇒ `nvidia`.
- `backend_native` is free-form (ignored by `diagnose.py`); planner prompts and
  `idioms.yaml:triggers_on_native` may read it.
- `coverage` lists which canonical keys were honestly populated → drives the **null rule**
  below and lets the workflow log fidelity.

**Canonical-unit table (every backend's `to_evidence` MUST honor):**

| key | unit | example native source |
|---|---|---|
| `latency_ms` | milliseconds | `gpu__time_duration.sum` (ns) ÷ 1e6 |
| `dram_pct` | **0–100** | `dram__bytes_*.pct_of_peak…` |
| `sm_pct` | **0–100** | `sm__throughput.avg.pct_of_peak…` |
| `occupancy` | **0–1** | `sm__warps_active.avg.pct_of_peak…` **÷ 100** |

> **The single most error-prone line.** NCU reports occupancy as 0–100; `diagnose.py`
> compares `occ < 0.40` (0–1). `to_evidence.py` **MUST divide by 100**. Forgetting it makes
> every kernel look fully occupied → `latency_occupancy` never fires. Mandatory L2 fixture
> asserts the 0–1 *range* of occupancy and run-to-run determinism (§4.9).

**The null rule (decided behavior; works with the §5.3.2 `diagnose.py` edit):** a backend that
lacks a counter sets the key **`null`** and omits it from `coverage`. The §5.3.2 edit makes
`classify()` treat `None` as "cannot conclude" (not `0.0`). The **decided rule** is:
**a bottleneck branch may fire only when every metric it compares is measured.** Concretely
(decided, not accidental):
- `latency_occupancy` fires on `occupancy` alone (single positive signal);
- `compute_bound` fires on `sm_pct` alone (single positive signal);
- `memory_bound` and `overhead_bound` are two-sided checks and require **both** `dram_pct` and
  `sm_pct` measured;
- any case missing a required measured discriminator → **`unknown`**, never a fabricated label.

So `{dram:80, sm:null}` → `unknown` (memory_bound needs both; high DRAM alone cannot exclude
"both high"), `{dram:null, sm:80}` → `compute_bound`, `{dram:null, sm:30, occ:0.6}` → `unknown`.
A backend may optionally derive a flagged proxy (`backend_native.dram_pct_source: "stall_proxy"`)
to make a key measured; never required. Codes: `0` normalized · `2` native file unparseable ·
`3` bad args. Pure function.

### 4.7 `idioms.yaml` schema

```yaml
schema_version: 1
backend_id: <id>
lang_fence: cuda                 # the ```fence``` language for code blocks (cuda|python|metal|cpp)
impl_requirements: <string>      # ABI/scaffolding the executor must emit (e.g. PYBIND11_MODULE, host_wrapper.mm)
unsupported_methods: [ ... ]     # method_gate.TABLE names this backend cannot honor (gated OUT of the prompt)
read_metric_guide: <string>      # OPTIONAL per-backend "how to read profiler data" causal model (replaces AccelOpt L446-457)
methods:
  <abstract_method_name>:        # MUST appear in method_gate.TABLE (or be in unsupported_methods)
    idiom: <string>              # REQUIRED. concrete construct.
    prompt_guidance: <string>    # REQUIRED. instruction injected into executor prompt.
    anti_idiom: <string>         # OPTIONAL.
    triggers_on_native: [..]     # OPTIONAL. backend_native keys justifying this method.
    code_markers: [..]           # OPTIONAL. tokens the executor should emit (self-check).
```

Example abstract → concrete: `tensor_core_mma.idiom` is `"wmma / mma.sync"` (CUDA),
`"tl.dot"` (Triton), `"MFMA (v_mfma_*)"` (ROCm), `"simdgroup_matrix +
simdgroup_multiply_accumulate"` (Metal). A backend that cannot gate on `memory_bound` lists
`memory_coalescing` etc. under `unsupported_methods`. `method_gate.py` is never edited.

### 4.8 Taxonomy extension without breaking other backends

Three rules, all inside `to_evidence.py`, all enforced by the L2 fixture (§4.9):
1. **Null, don't fabricate** (paired with §5.3.2): unmeasured key → `null` + absent from
   `coverage` → `unknown`, never a wrong confident label.
2. **Map semantically-equivalent counters into the canonical slot** (`sm_pct ← alu_active_pct`,
   `occupancy ← simdgroup_occupancy/100`). These are **bets**, not equivalences (§5.2/§10).
3. **Optional flagged proxy** for a missing key; never required for correctness.

### 4.9 Conformance levels (L0--L3), checked by `validate_backend.py`

Mirrors solver L0–L3 in `SOLVER-SDK.md`, applied to the *driver*, validated against `_schema/`.

- **L0 — Declared:** `manifest.yaml` validates; `backend_id == dir == normalizeSuitabilityValue(id)`;
  `idioms.yaml` references only real `method_gate.TABLE` names (incl. `unsupported_methods`);
  `capabilities.metrics ⊆` canonical keys; `bottleneck_classes ⊆` the 4 meaningful ∪
  `{unknown}`. **Substrate diff-guard:** the six universal scripts are byte-identical to the
  baseline blob **except** the three §5.3 hunks, each covered by a golden test.
- **L1 — Buildable & Honest:** `build.sh`/`run.sh` executable; emit the contracted envelope
  + codes on a smoke fixture. `run.sh` output piped to `anti_cheat.py --source <kernel> --metrics -`
  is **accepted** (keys resolve: `claimed_speedup`/`compile_latency_ms`/etc.); an incorrect
  fixture yields `valid:false`/`reward:-1`; a `claimed_speedup>1.0` on `correct:false` is
  rejected end-to-end. For non-CUDA vendors the L1 fixture also feeds a **library-fallback
  kernel** (e.g. MPS for Metal) and asserts `valid:false` — guarding the per-vendor
  fallback-pattern requirement (§5.3.3).
- **L2 — Diagnostic (neutral-interface conformance):** `to_evidence.py` is pure (same native
  input ⇒ identical metrics; catches the ÷100 rot and asserts occupancy ∈ [0,1]). Its output
  into `diagnose.py --metrics -` yields a class in the enum and **only** classes the manifest
  declares. **Positive-classification fixtures (blocker fix), pinning the decided §4.6 null
  rule:** `{dram:null, sm:30, occ:0.6}` → **`unknown`** (forbid `overhead_bound`/`memory_bound`);
  `{dram:80, sm:null}` → **`unknown`** (memory_bound needs both measured — the case the first
  null-fix sketch missed); `{dram:null, sm:80}` → **`compute_bound`** (sm-alone positive); plus
  one fixture per reachable `(sm-band × occ-band)` cell. **Until §5.3.2 lands these fixtures FAIL
  — which is the correct signal that the substrate edit is a prerequisite, not optional.**
- **L3 — Compounding:** every gated method for every declared `bottleneck_class` resolves to
  an `idioms.yaml` entry (or `unsupported_methods`) — no gated method is un-translatable. A
  full driver round-trip on a reference fixture produces a valid Layer A envelope via the
  assembly step + `evidence_schema.py validate -` (positional, §4.10), with honest
  `coverage`. `requires_tools` preflight resolves; `optional_tools` absence degrades to the
  declared lower capability, not an error.

### 4.10 The three-consumer data flow (blocker fix — these are NOT one document)

```
run.sh  ──► result.json (metrics keys) ──► anti_cheat.py --source <kernel> --metrics -      → {valid,reward,recorded_speedup,flags}
profile.sh ─► native.<ext> ─► to_evidence.py --native - --format … ─► metrics.json ─► diagnose.py --metrics -   → {bottleneck_class,evidence}
                                                                                        method_gate.py --class <bclass>          → {allowed_methods}
ASSEMBLY step (a workflow agent) builds a Layer-A INSIGHT envelope from evidence_schema.py `template`
   (attempt_id, compiled, correct, speedup, metrics, insights[], …)  ──► evidence_schema.py validate -   → {valid, normalized}
```

Three distinct JSON documents, three distinct CLIs: `anti_cheat.py --metrics` and
`diagnose.py --metrics` take the **metrics dict**; `evidence_schema.py validate` takes a
**positional path** to a **typed-insight envelope** (kinds `bottleneck`/`failed_strategy`/…),
*not* a metrics dict. The assembly step (which agent builds the envelope, from which fields)
is an explicit deliverable in P4.

---

## 5. Reference drivers (CUDA, Triton, Metal)

### 5.1 The CUDA + Triton "vendor collapse" — precise scope

CUDA and Triton are **one vendor wearing two languages**: different source (`.cu` C++ vs
`@triton.jit` Python) lowering to the **same PTX → SASS** on the **same NVIDIA SMs**, so the
**same `ncu`** profiles both with the **same counter names**.

| | CUDA | Triton | Shared? |
|---|---|---|---|
| `source_ext` / `build.sh` | `.cu` / `nvcc -lineinfo` | `.py` / JIT warmup (materialize PTX into `TRITON_CACHE_DIR`) | **no** |
| runs on / profiler | NVIDIA SM / `ncu` | NVIDIA SM / `ncu` | **yes** |
| classifier counters (`dram_pct`,`sm_pct`,`occupancy`) | NCU | identical NCU | **yes** |
| `to_evidence.py` (classifier layer) | nvidia mapping | **same file** (`from ..cuda.to_evidence import main`) | **yes** |
| `diagnose.py` thresholds | nvidia | nvidia | **yes** |
| `method_gate.py` TABLE | abstract | abstract | **yes** |
| `profile.sh` (kernel-name targeting) | `ncu -k <name>` by `KERNEL_NAME_REGEX` | **NO** — mangled `triton_<fn>_<hash>` | **no** |
| `backend_native` source-attributed fields (sectors/req, per-line stalls) | needs `-lineinfo` | **partial** — weaker/absent Triton source attribution | **no** |
| `idioms.yaml` realization | `mma.sync`, `cp.async`, `__shared__` | `tl.dot`, `num_stages`, auto-smem | **no (spelling)** |

> **The collapse is in the metric/diagnosis layer; the load-bearing unsolved Triton work is
> kernel-name auto-discovery for `ncu -k` (mangled `triton_<fn>_<hash>` from
> `TRITON_CACHE_DIR`), which lives in `profile.sh`, NOT in `to_evidence.py`.** The
> shared-`to_evidence` claim is true **only for the three canonical classifier counters**;
> `backend_native` source-attributed evidence (sectors/request, per-line stalls) is *not*
> symmetric, because CUDA mandates `-lineinfo` (`AccelOpt:517`) and targets kernels by name
> (`KERNEL_NAME_REGEX`, `:123/:298`) while Triton's attribution is weaker. So CUDA↔Triton is
> **"identical for diagnosis, partial for source-line evidence."** The §8.3b Triton check
> asserts `ncu` actually attributed a non-empty CSV to the Triton kernel, not merely that
> `to_evidence` parsed a file.

### 5.2 Metal as the new-vendor proof

Metal is the only genuinely new vendor in the survey (zero Apple workflows exist). It
exercises the whole abstraction:

- **New language(s):** `.metal` device source + `.mm` Objective-C++ host wrapper + a
  `.metallib` intermediate — *two* source files (the `metal-kernel-batch-development` skill's
  `output/metal_tasks/<task>/{kernel.metal,host_wrapper.mm}` layout is the reference; note it
  breaks the single-`kernel_path`/`source_ext` assumption — §10).
- **New toolchain:** `xcrun metal` (.metal→.air) + `metallib` (.air→.metallib) +
  `clang++ -ObjC++ -framework Metal` host link.
- **New vendor, no NVIDIA counters:** unified memory (no discrete DRAM), 32-wide SIMD EUs (no
  warps/SMs); profiler is in-process `MTLCounterSampleBuffer` (or `xctrace`).

Metal's `to_evidence.py` is a **separate file** that synthesizes canonical fields:
```python
# metal/to_evidence.py — DIFFERENT FILE from the nvidia one
{ "latency_ms": c["gpu_time_ms"],
  "dram_pct":   None,                              # unified memory: NULL, not fabricated
  "sm_pct":     c["alu_active_pct"],               # SEMANTIC BET (unvalidated, §10)
  "occupancy":  c["simdgroup_occupancy"]/100.0,    # SEMANTIC BET, normalized to 0..1
  "_vendor":    "apple",
  "backend_native": { "mem_bandwidth_pct": c["mem_bandwidth_pct"], "llc_hit_pct": c.get("llc_hit_pct") } }
```
Its manifest declares `metrics.dram_pct: false`, `threshold_profile: apple`, and **omits
`memory_bound`** from `capabilities.bottleneck_classes`. With the §5.3.2 null edit,
`{dram_pct:null}` lands in `unknown` (not `overhead_bound`).

> **Honest framing (review fix):** `compute_bound` and `latency_occupancy` are **reachable**
> on Metal — they do not "work with zero substrate change." `sm_pct ← alu_active_pct` and
> `occupancy ← simdgroup_occupancy` are **unvalidated semantic bets** feeding NVIDIA-derived
> thresholds; the `apple` profile cutoffs (§5.3.1) are **estimates** until calibrated on real
> Apple GPU runs. Metal is **matrix-eligible only after** that calibration + the MPS
> fallback-pattern set (§5.3.3) land. `profile.sh` for Metal is partly a **contract bend**:
> counter sampling lives in the user-authored `host_wrapper.mm`; Metal's `profile.sh` *wraps*
> the host binary's in-process `MTLCounterSampleBuffer` output into the §4.5 native-pointer
> shape rather than being a standalone profiler.

### 5.3 The scoped substrate edits Metal/non-NVIDIA forces (the only substrate changes)

All three default to NVIDIA behavior and are golden-tested so CUDA/Triton are byte-identical
in effect.

**5.3.1 `diagnose.py` vendor threshold profile.** Replace the hardcoded literals with a
default-`nvidia` lookup:
```python
PROFILES = {
  "nvidia": dict(occ_lat=0.40, dram_mem=70, sm_mem=50, sm_comp=70, both_low=40),  # today's numbers
  "apple":  dict(occ_lat=0.30, dram_mem=65, sm_mem=55, sm_comp=65, both_low=35),  # ESTIMATES (§10)
}
prof = PROFILES.get(m.get("_vendor", "nvidia"), PROFILES["nvidia"])
```
Golden test: every existing `diagnose.py` unit case (no `_vendor`) is byte-identical.

**5.3.2 `diagnose.py` null-vs-zero handling (blocker fix).** Replace the `d = dram or 0.0` /
`s = sm or 0.0` coercion with the decided §4.6 "required-discriminator-measured" rule — a
two-sided branch requires both operands measured; a single-signal branch may fire alone:
```python
if dram is None and sm is None and occ is None: return "unknown", ["no classifier metrics"]
if occ is not None and occ < prof["occ_lat"]: return "latency_occupancy", [...]
if sm is not None and sm >= prof["sm_comp"]:   return "compute_bound", [...]        # sm alone
if dram is not None and sm is not None:                                             # two-sided: both measured
    if dram >= prof["dram_mem"] and sm < prof["sm_mem"]: return "memory_bound", [...]
    if dram < prof["both_low"] and sm < prof["both_low"]: return "overhead_bound", [...]
return "unknown", [...]
```
Golden test: every existing NVIDIA case (both measured) is byte-identical; new cases assert
`{dram:null, sm:30}` → `unknown`, **`{dram:80, sm:null}` → `unknown`** (the case the first
sketch missed — memory_bound no longer fires on unmeasured SM), `{dram:null, sm:80}` →
`compute_bound`. **This is the keystone fix** — without it a memory-bound Metal kernel is
misclassified (`overhead_bound` *or* a false `memory_bound`), then `method_gate` advises
`library_fallback_hybrid` → MPS, an *undetectable* cheat; §5.3.3 + §7 close that chain.

**5.3.3 `anti_cheat.py` per-vendor cheat patterns (BOTH lists — review fix).** Add **one
optional argument** `--vendor-patterns-file <path>` (body unchanged otherwise); the file has
two sections, `[fallback]` and `[skip]`, whose regex lines are appended to `FALLBACK_PATTERNS`
and `SKIP_PATTERNS` respectively. The driver supplies the path via `manifest.vendor_patterns_file`.
Examples — Metal `[fallback]`: `MPSMatrixMultiplication`, `MPSNDArray`, `MPSNNGraph`; Metal
`[skip]`: `//\s*TODO`, `return\s*;` (C++ stub), `__builtin_unreachable`; ROCm `[fallback]`:
`rocblas`, `miopen`. Absent ⇒ CUDA defaults ⇒ byte-identical behavior. **Covering only
`FALLBACK_PATTERNS` would leave Metal's skip-detection silently weak** (its Python/CUDA
defaults — `pass`, `raise NotImplementedError` — never match C++). Until Metal's set exists,
Metal's `idioms.yaml` lists `library_fallback_hybrid` under `unsupported_methods` so the planner
is never steered toward an undetectable fallback.

---

## 6. How a workflow consumes a driver at runtime

The JS body **cannot read files** (no `fs`/`require`/`import`/time/random — `ARCHITECTURE.md`).
Dispatch is two-layer, identical in spirit to `substrateInstruction()`: the body computes
string paths; an agent `cat`s the driver files.

### 6.1 Path helpers + single-sourced BACKEND (blocker fix)

`BACKEND` is resolved in **exactly one place** — the guard (§6.4) — via
`normalizeSuitabilityValue` (so `hip→rocm`, `_→-`, etc. apply consistently and match the
driver dir name). There is **no** second `const BACKEND`.
```js
const BACKEND = await assertWorkflowSuitability()       // the ONLY assignment (§6.4)
const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''          // python interpreter for .py only
const SH = args.driver_shell_prefix || ''               // optional; '' → rely on shebang
const DRIVER_DIR = `${SUBSTRATE}/backends/${BACKEND}`

function substrateInstruction(script, cliArgs) {        // .py substrate scripts
  const p = `${SUBSTRATE}/${script}`
  return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
            : `No substrate_command_prefix for ${p} ${cliArgs}; do not invent an interpreter.`
}
function driverPy(script, cliArgs) {                     // to_evidence.py — python prefix
  const p = `${DRIVER_DIR}/${script}`
  return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
            : `No substrate_command_prefix for ${p}; do not invent an interpreter.`
}
function driverSh(script, cliArgs) {                     // build/run/profile — shebang, NO python
  return `Run exactly: \`${SH ? SH + ' ' : ''}${DRIVER_DIR}/${script} ${cliArgs}\`.`
}
```

### 6.2 Setup phase — one agent `cat`s the driver files

```js
phase('Setup')
const driver = await agent(
  `Load the backend driver for backend="${BACKEND}".\n` +
  `1. Run exactly: \`cat ${DRIVER_DIR}/manifest.yaml\` and parse YAML.\n` +
  `2. Run exactly: \`cat ${DRIVER_DIR}/idioms.yaml\` and parse YAML.\n` +
  `If either is missing, return {present:false, reason:"no driver for backend ${BACKEND}"}.\n` +
  `Also compare manifest.capabilities against the required capability floor ${JSON.stringify(WORKFLOW_SUITABILITY.requires_capability)};\n` +
  `if a required metric/class is missing return {present:true, capability_ok:false, missing:[...]}.\n` +
  `Return {present, capability_ok, missing, backend_id, source_ext, lang_fence, hw_vendor,\n` +
  `  profiler_name|null, profiler_format, capability_metrics, supported_classes, problem_types,\n` +
  `  requires_tools, impl_requirements, read_metric_guide, idioms:{<method>:{idiom,prompt_guidance}}}.`,
  { label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH })

if (!driver.present) throw new Error(`No backend driver present for backend="${BACKEND}". Provide ${DRIVER_DIR}/ or pick a supported backend.`)
if (driver.capability_ok === false) throw new Error(`backend="${BACKEND}" lacks required capability: ${(driver.missing||[]).join(', ')}.`)
const IDIOMS = driver.idioms || {}
```
`$BACKEND` flows: `args.backend → guard → BACKEND → DRIVER_DIR → cat → driver.idioms`
(in-memory JS object) → injected into later prompts as strings. No file is touched by JS.

### 6.3 Build / Run / Profile / Diagnose / Gate / Idiom-translate (sketch)

```js
// Profile (native) → normalize → diagnose, in canonical metric space:
const prof = await agent(
  driver.profiler_name
    ? driverSh('profile.sh', `--artifact ${best.artifact} --problem ${PROBLEM_JSON} --out ${runDir}/prof.native`)
      + ` Then ` + driverPy('to_evidence.py', `--native ${runDir}/prof.native --format ${driver.profiler_format} --run ${runDir}/result.json`)
      + ` Return its stdout JSON verbatim ({metrics, coverage}).`
    : `Backend "${BACKEND}" declares no profiler. Return {metrics:{dram_pct:null,sm_pct:null,occupancy:null,latency_ms:null}, coverage:[]}.`,
  { label:`evidence-${iter}`, phase:'Profile', schema: JSON_PASSTHROUGH })

const diag = await agent(
  `Write these metrics to ${runDir}/metrics.json:\n${JSON.stringify(prof.metrics || {})}\n` +
  `${substrateInstruction('diagnose.py', `--metrics ${runDir}/metrics.json`)} Return stdout JSON verbatim.`,
  { label:`diagnose-${iter}`, phase:'Diagnose', schema: JSON_PASSTHROUGH })
const bclass = diag.bottleneck_class || 'unknown'

const gate = await agent(`${substrateInstruction('method_gate.py', `--class ${bclass}`)} Return stdout JSON.`,
  { label:`gate-${iter}`, phase:'Plan', schema: JSON_PASSTHROUGH })
const idiomBlock = (gate.allowed_methods || [])
  .filter(m => !(IDIOMS.unsupported_methods||[]).includes(m))
  .map(m => { const e = IDIOMS[m] || { idiom:m, prompt_guidance:'(no backend idiom; use abstract method)' }
              return `- ${m}  →  ${e.idiom}\n    ${e.prompt_guidance}` }).join('\n')

await agent(
  `You are an expert ${BACKEND} kernel developer (source ${driver.source_ext}, fence \`\`\`${driver.lang_fence}).\n` +
  `${driver.read_metric_guide || ''}\n` +
  `Bottleneck: ${bclass}. Choose exactly one method from this gated, backend-translated set:\n${idiomBlock}\n` +
  `Implementation requirements: ${driver.impl_requirements}\n` +
  `Return {method (the abstract name), plan, anchors}.`,
  { label:`plan-${iter}`, phase:'Plan', schema: JSON_PASSTHROUGH })
```
The agent picks an **abstract** `method` (so `memory_store`/`method_gate` keep universal
keys) but is *taught the concrete idiom*. Swapping `args.backend` changes only `idiomBlock`'s
right-hand side + `lang_fence` + `impl_requirements`; gate/memory/evidence contracts are
identical. Every `impl_requirements`/ABI string (`PYBIND11_MODULE`, `host_wrapper.mm`, …)
comes from `idioms.yaml` — the workflow body never names an ABI.

### 6.4 Suitability-guard flip (whitelist → method-support + driver-presence)

`WORKFLOW_SUITABILITY` splits the single language whitelist into **method support** (a
JS-body static check) and **driver presence + capability floor** (an agent-side check, since
only an agent can stat files — done in §6.2 Setup). The existing 4 keys are preserved.

```js
const WORKFLOW_SUITABILITY = {
  method_supported_backends: 'any',     // 'any' = clean; or ['cuda','triton'] | ['rocm']
  default_backend: 'cuda',              // backward-compatible default
  requires_capability: { bottleneck_classes: [], metrics: [] },  // method's capability floor (checked in Setup)
  supported_problem_types: ['kernel-optimization', 'kernel-generation'],  // UNCHANGED key
  problem_types: ['existing kernel optimization', 'generation from problem_definition'],  // UNCHANGED key
  reason: 'Backend-agnostic method; runs on any backend with a present driver.',
}

function resolveBackend() {
  const b = args.backend ? normalizeSuitabilityValue(args.backend) : null
  const l = args.language ? normalizeSuitabilityValue(args.language) : null
  if (b && l && b !== l) throw new Error(`Conflicting args: backend="${b}" vs language="${l}". Pass only one.`)
  if (b) return b
  if (l) return l                                   // legacy alias
  const ms = WORKFLOW_SUITABILITY.method_supported_backends
  if (Array.isArray(ms) && ms.length === 1) return normalizeSuitabilityValue(ms[0])
  return WORKFLOW_SUITABILITY.default_backend        // 'cuda'
}

function assertWorkflowSuitability() {               // SYNC — does no file/agent work itself
  const backend = resolveBackend()
  const ms = WORKFLOW_SUITABILITY.method_supported_backends            // (a) METHOD check (JS-only)
  if (ms !== 'any' && !ms.map(normalizeSuitabilityValue).includes(backend))
    throw new Error(`${meta.name}'s method does not support backend="${backend}". Method-supported: ${ms.join(', ')}. Reason: ${WORKFLOW_SUITABILITY.reason}`)
  // (b) problem_type check: unchanged supportsSuitabilityValue() suffix-match + 'auto' sentinel.
  // (c) driver presence + capability floor: deferred to the Setup 'load-driver' agent (§6.2).
  return backend
}
const BACKEND = assertWorkflowSuitability()          // sync; the ONLY BACKEND assignment
```
> `assertWorkflowSuitability()` is **synchronous** (it returns a computed string and does no
> I/O); driver presence/capability is validated in the Setup phase, where an agent can read
> files. (Earlier draft made it `async` with no `await` inside — misleading; corrected.)

**Backward compatibility is exact:** no `args.backend`/`args.language` → `'cuda'`;
`method_supported_backends:'any'` passes; CUDA driver present → identical to today's
`supported_languages:['cuda']` path. Old callers passing `language:'cuda'` still work via
`normalizeSuitabilityValue`. `backend` and `language` both set but **conflicting** → explicit
error (resolves the precedence open question).

---

## 7. Capability negotiation & portability tiers

| Question | Owner | Mechanism |
|---|---|---|
| "Can this **method** run on backend X?" | the workflow | `method_supported_backends` (`'any'`/list) + `requires_capability` |
| "Does backend X have a **driver** + needed **capabilities**?" | the driver `manifest.yaml` | `capabilities.{metrics, bottleneck_classes, problem_types}` + `requires_tools` |

Negotiation is **capability-driven, not vendor-name-driven**: a `vendor_locked` method
declares `requires_capability.metrics: ['dram_pct','sm_pct']`, so even a present `rocm` driver
is rejected unless it can supply the NCU-class signal; a future ROCm-with-DRAM-proxy could
satisfy a CUDA-authored method. The floor is checked in Setup (§6.2).

**Resolution order** (deterministic, no guessing): `args.backend` → `args.language` alias →
single-element `method_supported_backends` → `default_backend` (`cuda`); conflicting
`backend`+`language` → error. An absent driver is a **hard failure**, never a silent fallback.

### 7.1 The three portability tiers

- **clean** → `method_supported_backends: 'any'`; runs wherever a driver exists.
- **vendor_locked** → e.g. `['cuda','triton']` (both NVIDIA/NCU) **and**
  `requires_capability.metrics: ['dram_pct','sm_pct']`. Locked to a *tool*, not a language.
- **method_intrinsic** → a single-element hard whitelist (`['cpp']`/`['cutlass']`/`['rocm']`)
  + pinned `requires_capability.problem_types`. The method *is* the backend; `matrix_eligible:false`.

### 7.2 Per-workflow classification (27 workflows)

| Workflow | Tier | `method_supported_backends` | Matrix | Intrinsic to |
|---|---|---|---|---|
| AdaExplore, KernelAgent, AKO4X, Astra, CUDALLM, KDA, KSearch, ReGraphT, STARK, StitchCUDA, KernelFoundry, KernelFoundryDx, KernelSkill | clean | any | yes | — |
| KernelBand | clean\* | any | yes | NVIDIA utilization in φ-gate (driver-resolved threshold) |
| Generalist | clean | any | yes | — (substrate reference; migrate last) |
| AccelOpt, CUDAAgent, cuPilot, KEET, KernelBlaster, GPUForecasters | vendor_locked | ['cuda','triton'] (+ `{dram_pct,sm_pct}`) | partial | NVIDIA NCU |
| TritorX | vendor_locked | ['triton'] | no | Triton dialect + custom linter |
| Xe-Forge | vendor_locked | ['xpu'] | no | Intel XPU (XMX, EU, SPIR-V) |
| ArchAgent | method_intrinsic | ['cpp'] | no | LLC cache-replacement policy (ChampSim IPC) |
| CutlassGEMM | method_intrinsic | ['cutlass'] | no | CUTLASS device-level API |
| FACT | method_intrinsic | ['cutlass'] | no | CUTLASS pattern registry |
| ARGUS | method_intrinsic | ['rocm'] | no | AMD MI300X MFMA invariants |

Counts: **15 clean / matrix-eligible**, 6 vendor_locked-partial, 2 vendor_locked-single, 4
method_intrinsic. `method_intrinsic` workflows still adopt the driver *shape* (their metric
vocabulary lives in their single driver) but `matrix_eligible:false` excludes them from the
cross-product CI; an off-list backend request gets a precise refusal naming the one supported
backend, never a broken run.

> \* KernelBand is a clean topology whose hardware-masking gate leans on NVIDIA utilization;
> its threshold becomes driver-resolved (effort M).
>
> **ArchAgent decision (resolves open question):** ArchAgent's signal is ChampSim IPC, not a
> memory/compute bottleneck. It **bypasses `diagnose.py` entirely** and uses only the neutral
> substrate scripts, so the 5-class enum stays frozen (no enum extension for cache-policy work).

---

## 8. Pilot: AccelOpt retrofit

**Target:** `AccelOpt/accelopt-kernel-optimization.js`. AccelOpt is `vendor_locked` (intrinsic
to NCU), so the pilot proves the **driver wiring mechanics** on a workflow whose *method*
still needs an NVIDIA profiler — it stays single-vendor but becomes driver-*shaped* and gains
substrate Layer C/E conformance it has none of today. It is the reference the generator then
learns to emit.

### 8.1 Seam inventory (21 CUDA/NCU couplings)

| # | Seam | Lines | Hardcoded |
|---|---|---|---|
| S1 | Suitability declaration | 15–20 | `supported_languages:['cuda']`, problem types, NCU/CUDA reason |
| S2 | `args.language` default | 135 (+114 `OP_DESC='CUDA kernel'`) | `LANGUAGE = args.language \|\| 'cuda'` |
| S3 | meta description / whenToUse | 2–4 | "CUDA … NCU profiling", "Nsight Compute" |
| S4 | Phase detail strings | 6–11 | "NCU profile baseline/variants" |
| S5 | Header/usage doc | 90–96 | `kernel.cu`, `harness.cu`, `ncu_binary` |
| S6 | Harness/NCU arg wiring | 120–124 | `HARNESS_*`, `KERNEL_NAME_REGEX`, `NCU_BINARY` |
| S7 | Generate-seed prompt | 161–181 | "Generate … CUDA kernel candidates" |
| S8 | Setup read prompt | 258–268 | "Read the CUDA kernel", `__global__` |
| S9 | NCU baseline agent prompt | 292–337 | "Nsight Compute", ```` ```cuda ````, explicit NCU metric names, `ncu -k` |
| S10 | NCU baseline schema | 343–361 | `sm_throughput_pct`, `dram_throughput_pct`, `sectors_per_request`, `ncu_available` |
| S11 | `baselineNcuProfile` builder | 378–397 | SM/DRAM throughput, "Sectors/Request (ideal=4)" |
| S12 | `planAngles` array | 420–426 | `long_scoreboard`, `sectors/request`, tensor cores |
| S13 | "How to read NCU data" block | 428–465 | ```` ```cuda ````, causal model L446–457 |
| S14 | Plan schema | 472–484 | `ncu_evidence` |
| S15 | `buildBeamSection` fence | 246 | ```` ```cuda ```` |
| S16 | Execute prompt | 501–520 | "CUDA developer", `.cu`/`__global__`/`PYBIND11_MODULE`, `-lineinfo` |
| S17 | Evaluate prompt | 562–602 | "Nsight Compute", `cuda_runtime.h`, `__shared__`, sectors/request |
| S18 | Evaluate schema | 605–619 | `ncu_comparison` |
| S19 | Learn prompt + format | 741–806 | "NCU profiling", `NCU trigger:` |
| S20 | Learn schema | 783–794 | `ncu_trigger` |
| S21 | Final-report prompt + return | 820–875 | "AccelOpt + NCU", return key `ncu_baseline_profile` |

### 8.2 Before/after (representative; the retrofit is purely additive)

- **S1 (suitability):** to the §6.4 split with `method_supported_backends:['cuda','triton']`,
  `requires_capability.metrics:['dram_pct','sm_pct']`, default `'cuda'`. Existing
  problem-type keys + `normalizeSuitabilityValue` unchanged.
- **S6 (args):** keep `HARNESS_BUILD_CMD`/`NCU_BINARY` but treat them as the **cuda driver's
  inputs** — cuda `build.sh`/`profile.sh` accept `--build-cmd`/`--ncu` so existing callers'
  values flow straight through.
- **S9+S10+S11 (core swap):** the NCU-hand-rolling prompt becomes `driverSh('profile.sh', …)`
  then `driverPy('to_evidence.py', …)`, returning driver-normalized metrics; the NCU-shaped
  schema collapses to an `additionalProperties:true` passthrough; `baselineNcuProfile`
  renders present keys only; **new:** metrics feed `diagnose.py` (AccelOpt gains a
  `bottleneck_class`).
- **S12+S13 (planAngles + NCU-reading):** come from `idioms.yaml` (`read_metric_guide` +
  per-method guidance), optionally narrowed by `method_gate.py`. For cuda, `idioms.yaml`
  carries the **exact current strings** so the cuda path is verbatim-preserved.
- **S7/S8/S15/S16/S17/S19 (fences/syntax):** every ```` ```cuda ```` →
  ```` ```${driver.lang_fence} ````; the `PYBIND11_MODULE`/`torch/extension.h`/`forward()`
  ABI contract moves into `idioms.yaml:impl_requirements`.
- **S14/S18/S20 (schema fields):** rename `ncu_evidence`/`ncu_comparison`/`ncu_trigger` →
  neutral `profile_*`, keeping old keys as optional aliases (`x.profile_evidence ?? x.ncu_evidence`).
- **S21 (return):** keep `ncu_baseline_profile`, add neutral alias `baseline_profile` + a
  Layer-A `evidence` envelope (assembly step → `evidence_schema.py validate`) → AccelOpt
  becomes L0/L2 substrate-conformant.

New args: `backend` (default `'cuda'`), `backend_dir` (default `''` → legacy inline-prompt
path), `language` (deprecated alias), `driver_shell_prefix` (optional). **Every existing
caller works unchanged:** no `backend` → `'cuda'`; no `backend_dir` → driver calls fall back
to the legacy CUDA prompt path (gate the swap on `backend_dir ? driver-call : legacy-prompt`);
`IDIOMS` defaults equal the L420–457 / L513 / L585 literals.

### 8.3 Self-verification plan (with the falsifiability/prereq fixes)

- **(a) CUDA path byte-identical — needs the `--print-prompts` harness (its own deliverable).**
  AccelOpt has no dry-run mode; a `--print-prompts`/`agent()`-stub harness is a **sized
  prerequisite with its own acceptance test** (P4). With it: run today's AccelOpt and the
  retrofit with the **same args, no `backend_dir`**; diff fully-rendered prompt strings per
  phase → **byte-identical**. Assert `ncu_*` aliases + return keys preserved; assert
  `sampleWithoutReplacement` beam evolution identical on a **recorded fixture** (capture
  mechanism specified in P4). **What the pilot can falsifiably verify with ZERO drivers:** the
  entire CUDA legacy path (no `backend_dir`) — guard, beam, schemas, return — so §8.3a stands
  alone before any driver exists.
- **(b) Triton end-to-end — depends on the triton driver (P3, hard dependency).** Author
  `_substrate/backends/triton/` (build = JIT warmup, run = `torch.allclose` + `do_bench`,
  profile = `ncu` with **kernel-name auto-discovery from `TRITON_CACHE_DIR`**, `to_evidence` =
  shared nvidia map, `idioms.yaml` with `lang_fence:python`, no `PYBIND11`). Run
  `backend:'triton', backend_dir:…, kernel_path:<small .py>, substrate_command_prefix:'python3',
  iterations:1, breadth:1`. **Pass:** no `__global__`/`.cu` in any rendered prompt; `run.sh`
  returns `compiled:true,correct:true`; **`ncu` attributed a non-empty CSV to the Triton
  kernel** (proves kernel-name discovery, not just file parsing); `diagnose.py` returns a
  class; final return carries a Layer-A envelope validated by `evidence_schema.py`.
- **(c) Guard rejects a driver-less backend.** `backend:'metal'`, no driver →
  `method_supported_backends:['cuda','triton']` → guard throws before Setup, naming the set +
  reason. Inverse: `backend:'triton'` + valid driver does not throw. Negative driver test:
  `backend_dir` missing `to_evidence.py` → Setup agent returns the "no driver" sentinel →
  workflow degrades to static analysis (`profiler_available:false`) rather than fabricating
  metrics (honors the "do not invent a compiler" contract, L316/L319/L590).

---

## 9. Incremental migration

### 9.1 Rollout phases

- **Phase 0 — substrate becomes backend-parameterized (no workflow changes).** Add
  `_substrate/backends/_schema/` + `validate_backend.py` (in `_substrate/backends/`). **Golden-lock
  first:** freeze today's `diagnose.py`/`method_gate.py`/`anti_cheat.py` CUDA outputs, then
  land the three §5.3 edits (vendor profile, full measured-operand null-handling, anti-cheat `--vendor-patterns-file`)
  and prove the default-`nvidia`/no-file paths reproduce them byte-for-byte. Exit: CUDA
  behavior unchanged; diff-guard green (the only allowed substrate diff is the three §5.3
  hunks, each golden-tested).
- **Phase 1 — pilot AccelOpt** (§8), incl. the `--print-prompts` harness.
- **Phase 2 — all clean workflows.** **AdaExplore + KernelAgent first** (proven generic
  topologies — MCTS and routing+parallel-seeds — retrofittable by swapping only the evaluation
  harness). Then by decreasing complexity: KDA, CUDALLM, Astra, StitchCUDA, STARK, KSearch,
  ReGraphT, KernelFoundry, KernelFoundryDx, KernelSkill, AKO4X, KernelBand. **Generalist last**
  (substrate reference). Validate on a real new backend (Metal) only **after** AdaExplore +
  KernelAgent prove out **and** the §5.3.1 Apple calibration + §5.3.3 MPS patterns land.
- **Phase 3 — vendor_locked.** Declare the narrow legal set (`['cuda','triton']` / `['triton']`
  / `['xpu']`); still get driver wiring so the set widens by editing one field when a portable
  profiler appears.
- **Phase 4 — method_intrinsic.** Single-backend driver + `matrix_eligible:false` +
  `intrinsic_to`; never enter the matrix smoke test; off-list backend → guard throws the
  intrinsic-reason message.

### 9.2 Generator / template / schema changes — targeting what actually exists (blocker fix)

> **The generator and validator are LLM-agent pipelines, not substitution engines.** Verified:
> `generate-workflow.js` has no `readFile`, no `templates` read, no `BLOCK` logic — it passes
> the template *name* as a string (`# Template base: ${templateBase}`) and an agent authors the
> `.js`. The `{{TOKEN}}`/`[BLOCK]` markers in the templates are **authoring guidance for the
> LLM**, not a mechanical API. `validate-workflow.js` is likewise an agent prompt. So "add a
> token" / "replace a BLOCK" / "the generator injects the enum" are **not real levers**.

- **Manifest schema (`_meta/manifests/schema.yaml`):** add a top-level `backend:` block —
  `supported[]`, `default` (clean methods may set `null` to force explicit `--backend`),
  `matrix_eligible` (true|partial|false), `portability` (clean|vendor_locked|method_intrinsic),
  `intrinsic_to` (required when not clean). Extend `args.optional[]` to emit `backend` +
  neutral `profile_command` (`ncu_command` kept as a **documented alias** with a deprecation
  window — note this *demotes* a currently first-class arg, a deliberate contract change, not a
  "tightening"). This is the **deterministically enforceable** part (a data schema).
- **Generator (`_meta/tools/generate-workflow.js`):** add backend guidance to its **agent
  prompts** — the model-args prompt (≈L263–282) and the generate prompt (≈L492–524): extract
  `native_backend` + `portability_class`; instruct the model to emit the §6 driver-dispatch
  pattern + the §6.4 guard instead of naming `nvcc`/`ncu`. These are **prompt instructions**,
  not substitutions.
- **Validator:** the **deterministic** backend checks live in `validate_backend.py`
  (in `_substrate/backends/`; schema-validated: manifest/idioms conform, no driver edits a universal script, `backend_id
  == dir`). Any backend checks added to the LLM-driven `validate-workflow.js` are **prompt
  checklist items** (like its existing suitability checks), explicitly *not* hard gates.
- **Templates:** update the `{{…}}`/`[BLOCK]` guidance text + the input-policy comment
  (L70–75) to teach the LLM "the body never names a vendor profiler or vendor metric." These
  are documentation edits, not a substitution API.

### 9.3 CI / conformance / matrix-smoke-test (no GPU; synthetic fixtures)

- **Substrate diff-guard:** universal scripts byte-identical to baseline **except** the three
  golden-tested §5.3 hunks (protects all 26 CUDA/Triton workflows from a selector/null bug).
- **Driver conformance (L0–L3, §4.9):** every driver validates against `_schema/`;
  `to_evidence.py` purity + occupancy-range; **positive-classification** fixtures (null →
  `unknown`, not `overhead_bound`); idiom-completeness for every gated method.
- **Matrix smoke test:** restricted to workflows **already at the required conformance level**
  (i.e. that emit a Layer-A envelope — initially only AccelOpt post-pilot). For each
  `matrix_eligible:true` workflow × matrix driver, run a trivial kernel with a **mock harness**
  (a documented interface injecting canned `build.sh`/`run.sh`/`profile.sh` JSON without a GPU;
  fixtures in `_substrate/backends/_fixtures/`); assert structurally — guard passes, right
  driver dispatched, envelope conforms. **Negative cells** assert an **exact error
  substring/code** (`matrix_eligible:false` throws the intrinsic-reason; vendor_locked throws
  for illegal backends). Policy decision: **structural-only assertion is sufficient for
  `matrix:yes`**; a real-hardware tier (≥1 compiled kernel per backend) is opt-in behind a
  self-hosted-runner label.
- **anti_cheat per-backend patterns:** the L1 "honest" fixture for each non-CUDA backend feeds
  **two** kernels — a library-fallback kernel (e.g. MPS) **and** a skipped-compute stub (e.g. a
  C++ `return;` body) — and asserts `valid:false` for both, exercising the `[fallback]` *and*
  `[skip]` sections of `--vendor-patterns-file`.
- **Tiering:** conformance + diff-guard every PR; matrix smoke nightly; real-hardware tier opt-in.

### 9.4 Documentation deliverables

1. `_substrate/BACKEND-DRIVER-SDK.md` (companion to `SOLVER-SDK.md`/`ARCHITECTURE.md`): the
   contract, six-file layout, `manifest.yaml`/`idioms.yaml` schemas, the `to_evidence.py`
   neutral interface + the three extension rules + the unit table, the three portability
   tiers, the L0–L3 levels, and **the explicit list of the three scoped substrate edits**.
2. `_substrate/backends/REGISTRY.md` + per-driver README sections (profiler, emitted metric
   names, fallback patterns, threshold-profile deviations from `nvidia` and why).
3. Update `_meta/manifests/schema.yaml` header + `_meta/README.md` for the `backend:` block.
4. `_substrate/ARCHITECTURE.md`: one paragraph placing the Backend Driver axis as a
   cross-cutting data axis owned by neither solver nor generator.

---

## 10. Risks, blockers & resolved/open questions

**Resolved (were open questions; decided here so plans don't churn):**
- **`backend` vs `language` precedence:** explicit `args.backend` wins; conflicting non-equal
  `backend`+`language` → **error** (§6.4).
- **`triton` granularity:** **one backend id `triton`** for this iteration (triton-on-NVIDIA);
  a future `language × silicon` split is deferred.
- **ARGUS tier:** **method_intrinsic, `['rocm']`**, non-matrix (governs `supported_backends`).
- **ArchAgent + the enum:** ArchAgent **bypasses `diagnose.py`**; the 5-class enum stays frozen.

**Hard blockers (must land before dependent phases):**
- No `backends/` dir/contract exists yet — this spec defines it; drivers must be authored (P3)
  before §8.3b/Metal runs.
- The three §5.3 substrate edits are shared-substrate changes — default-safe + golden-tested
  before any non-NVIDIA backend relies on them; a bug regresses all 26 NVIDIA workflows.
- AccelOpt has no `--print-prompts` mode — required for §8.3a; a sized P4 deliverable.

**High-severity risks:**
- **Occupancy ÷100 rot** — the single most error-prone `to_evidence` line; mandatory L2 range fixture.
- **Metal threshold corruption** — Apple cutoffs are estimates until calibrated on real Apple
  GPU runs; Metal is matrix-eligible only after calibration.
- **Metal two-source build IO contract (third Metal prerequisite — review fix).** `manifest`
  exposes only scalar `source_ext` (+ `aux_ext[]`) and `build.sh --source <one path>`, which
  **cannot express** Metal's two co-primary sources (`.metal` device + `.mm` host). Beyond
  threshold calibration (§5.3.1) and MPS/skip patterns (§5.3.3), Metal needs a **build IO
  contract revision** (e.g. `source_ext` becomes a list, or a `--source-device`/`--source-host`
  pair) before its `build.sh` is authorable. P6-scoped; do not assume §4.5 already covers it.
- **Metal anti-cheat blind spot, chained to misclassification** — a memory-bound Metal kernel
  misdiagnosed `overhead_bound` → `method_gate` advises `library_fallback_hybrid` → MPS, which
  the default `FALLBACK_PATTERNS` cannot detect. Closed by §5.3.2 (null fix) + §5.3.3 (MPS
  patterns) + gating `library_fallback_hybrid` out of Metal idioms until detection exists.
  A negative conformance fixture (Metal kernel dispatching `MPSMatrixMultiplication` →
  `valid:false`) is a **hard prerequisite** for any Metal run.

**Medium-severity risks:**
- Substrate NVIDIA-tuning is advisory for non-NVIDIA backends (gate output is advisory; a
  mismatch degrades quality, not correctness).
- **`method_gate.gate()` dormant 0.8 cutoff** (`method_gate.py:32`) is neutral only while
  workflows call the gate **without `--metrics`** (§3.2). A migrated workflow that passes
  `--metrics` silently activates an NVIDIA-tuned `occupancy >= 0.8` threshold — fold it into the
  vendor profile if metrics-aware gating is ever needed.
- NCU-specific *reasoning* (AccelOpt L446–457) is a causal model, not string-substitutable;
  hand-authored per backend as `idioms.yaml:read_metric_guide`; backends without one fall back
  to class + allowed-methods (thinner heuristic).
- Triton kernel-name mangling needs auto-discovery from the JIT cache (the load-bearing
  un-built work, in `profile.sh`).
- Metal two-file materialization (`.metal`+`.mm`+`.metallib`) breaks the single-`source_ext`
  assumption (survey seams #4/#5).
- Metal CLI profiler fragility (in-process `MTLCounterSampleBuffer` in `host_wrapper.mm`).
- `anti_cheat` reward needs a per-vendor eager baseline (Metal has no `torch.compile` → reward
  caps at 2 → biases cross-vendor memory confidence; vendor-aware reward schedule, deferred).
- Profiler-availability asymmetry → cross-backend results are **not performance-comparable**
  (this iteration guarantees structural conformance only).

**Remaining open questions (genuinely deferrable, non-contract-affecting):**
- Apple/AMD measured threshold calibration numbers.
- `ncu_command` alias deprecation window length.
- Whether to ship the initial matrix as `cuda+triton` only, adding `rocm` after a real probe.

---

## 11. Out of scope / YAGNI

- Parameterizing `method_gate.TABLE` or `diagnose.py` thresholds by `--backend` *inside the
  script*. Rejected in favor of downstream `idioms.yaml` translation + the single default-safe
  vendor selector. (Note: the substrate is **not** fully frozen — three §5.3 edits — but no
  per-backend branching beyond a default-`nvidia` lookup.)
- A loadable driver *module* the `.js` imports — forbidden by the runtime; the driver is data
  an agent reads via Bash.
- Retargeting `method_intrinsic` workflows across backends.
- Backend *inference* beyond the explicit resolution order; an absent driver is a hard failure.
- Full ROCm/XPU driver implementations (specified as conformance targets; only CUDA/Triton/
  Metal authored as reference drivers this iteration).
- Measured Apple/AMD threshold calibration (estimated cutoffs ship; calibration is follow-up).
- Fully specifying the caller-supplied `build`/`run`/`profile` override arg surface (the
  capability exists as a back-compat shim; its full arg spec is deferred).
- Performance-comparable cross-backend scoring (needs normalized profiler fidelity + per-vendor
  reward schedules).
- Deduping the `_tools/`↔`_meta/tools/` and `_templates/`↔`_meta/templates/` trees (pre-existing).

---

## Appendix A — Implementation plan breakdown (for `writing-plans`)

Carved into shippable plans with dependency edges (the spec is one document; the *work* is not
one plan — review blocker #9).

| Plan | Scope | Depends on | Touches a workflow? |
|---|---|---|---|
| **P1** | Driver contract scaffolding: `_substrate/backends/_schema/{manifest,idioms}.schema.json`, `validate_backend.py` (`_substrate/backends/`), `REGISTRY.md`, `BACKEND-DRIVER-SDK.md` | — | no |
| **P2** | The scoped substrate edits + golden tests: `diagnose.py` (vendor profile + **full** measured-operand null rule — both `memory_bound` *and* `overhead_bound`), `anti_cheat.py` `--vendor-patterns-file` covering **both** `[fallback]` and `[skip]` lists | — | no |
| **P3** | Reference drivers `cuda/` + `triton/` (6 files each; shared nvidia `to_evidence`; Triton kernel-name auto-discovery) | P1, P2 | no |
| **P4** | AccelOpt pilot (§8) + the `--print-prompts` harness + beam-fixture capture + Layer-A assembly step | P1, P2, P3 | AccelOpt only |
| **P5+** | Clean-workflow migration in batches (§9.1 Phase 2 order), generator/validator/template/schema edits (§9.2) | P4 | yes (batched) |
| **P6 (later)** | Metal driver + Apple calibration + MPS fallback patterns + Metal matrix gating | P3, P5 | no (new driver) |

P1 and P2 are independent and can run in parallel. No plan is written until P1/P2's
contract-affecting decisions in §10 ("Resolved") are honored.

---

## Appendix B -- P5 outcomes (post-implementation record)

P5 (clean-workflow migration) was executed in six sub-phases (P5a--P5f). This
appendix records which open questions and spec assumptions P5 resolved.

### Decisions confirmed by implementation

1. **Validator is Python, not Node.** Spec §4.1/§4.9/§9.2 originally named
   `_meta/tools/validate-backend.js`. The implemented validator is
   `_substrate/backends/validate_backend.py` using stdlib `json` + live
   `method_gate.TABLE` import. No JSON-Schema files were shipped; the Python
   validator + `BACKEND-DRIVER-SDK.md` ARE the contract. All spec references
   have been corrected (P5f).

2. **Driver files are JSON, not YAML.** `manifest.json` / `idioms.json`
   (not `.yaml`). Rationale: Python has no stdlib YAML parser; adding `pyyaml`
   or a Node dependency was rejected. Field names are unchanged from the spec.

3. **`artifact_ext` may be empty string.** Triton's JIT model produces no
   persistent artifact file; `artifact_ext: ""` is valid and `build.sh` emits
   `artifact: null` on JIT-only paths.

4. **`to_evidence.py` vendor collapse.** CUDA and Triton share one nvidia
   `to_evidence` mapping via `from ..cuda.to_evidence import main` (Triton's
   `to_evidence.py` is a thin re-export). Confirmed correct for the three
   canonical classifier counters; `backend_native` source-attribution fields
   diverge as predicted in §5.1.

5. **Matrix-smoke is structural only.** The `matrix-smoke.test.js` CI tier
   asserts guard-pass, correct driver dispatch, and Layer-A envelope structure
   using canned agent returns -- no GPU required. Per §9.3 policy decision.

### Open questions P5 did NOT resolve (still deferred)

- Apple/AMD measured threshold calibration numbers (§5.3.1 estimates only).
- `ncu_command` alias deprecation timeline (retained as v1.1 alias; removal
  planned for v1.3 per schema.yaml changelog).
- Metal two-source build IO contract revision (§10 risk; P6-scoped).
- Vendor-aware reward schedule for `anti_cheat.py` (§10 medium risk).
- Triton kernel-name auto-discovery for `ncu -k` profiling (implemented as
  `TRITON_CACHE_DIR` glob in `profile.sh`; correctness on real hardware is
  deferred to GPU CI tier).
