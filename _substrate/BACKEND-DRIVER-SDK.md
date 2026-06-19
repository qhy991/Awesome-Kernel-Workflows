# Backend Driver SDK -- the (language x vendor) translation layer

Companion to [`SOLVER-SDK.md`](./SOLVER-SDK.md) and [`ARCHITECTURE.md`](./ARCHITECTURE.md). A
**backend driver** is a directory under `_substrate/backends/<backend_id>/` that adapts native
backend tooling (compiler, runner, profiler) to the universal substrate vocabulary, so any
clean optimization *method* can run on any *backend* by setting `args.backend` rather than
re-typing shell strings and re-authoring prompts. The driver is data an agent reads via Bash;
it is never `import`ed by a workflow `.js` body.

Full design rationale: `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md`.

## The six-file driver layout

```
_substrate/backends/<backend_id>/
  manifest.json      # static identity, toolchain, capabilities (machine-read; JSON)
  build.sh           # compile/JIT source -> artifact (executable; shebang; NO python prefix)
  run.sh             # artifact -> correctness + latencies in anti_cheat.py key shape (executable)
  profile.sh         # artifact -> native profiler output, emits a pointer (executable)
  to_evidence.py     # NEUTRAL INTERFACE: native profile -> canonical metrics dict (python prefix)
  idioms.json        # method translation table: abstract method_gate name -> backend idiom
```

`manifest.json`/`idioms.json` are machine-read JSON (see "Deviations from the spec"). The three
`.sh` files are executable and invoked WITHOUT a python interpreter; `to_evidence.py` IS invoked
with the python prefix. `validate_backend.py` checks the JSON files at L0.

**Validator location:** `_substrate/backends/validate_backend.py` (Python, stdlib `json` +
live `method_gate.TABLE` import). There is NO Node-based validator and NO JSON-Schema files
under `_schema/`. The Python validator + this document ARE the contract.

## Manifest v1.1: the `backend:` block in workflow manifests

Manifest schema [`docs/manifest-schema.yaml`](../docs/manifest-schema.yaml) is at **v1.1+**
(additive; older generation manifests still parse). The top-level `backend:` block declares a workflow's
`(language x vendor)` posture:

```yaml
backend:
  supported: ['any']          # or ['cuda','triton'] or ['rocm']
  default: null               # null -> callers must pass --backend explicitly
  matrix_eligible: true       # true | partial | false
  portability: clean          # clean | vendor_locked | method_intrinsic
  intrinsic_to: ""            # required when portability != clean
  requires_capability: {}     # capability floor (metrics, problem_types)
```

See [`docs/manifest-schema.yaml`](../docs/manifest-schema.yaml) for the full annotated reference and the v1.1 changelog comment.

Neutral backend-axis args added to `args.optional[]`: `backend`, `backend_dir`,
`profile_command`. `ncu_command` is retained as a DEPRECATED alias for `profile_command`
(deprecation window: v1.1 introduces alias; v1.2 emits warning; v1.3 removes).

## `manifest.json` fields

| field | req | meaning |
|---|---|---|
| `schema_version` | yes | manifest schema version (currently `1`) |
| `backend_id` | yes | canonical id; MUST equal the directory name; the dispatch key |
| `display_name` | yes | human label |
| `source_ext` | yes | primary kernel source extension (e.g. `.cu`); may be empty string for JIT-only backends |
| `aux_ext` | no | auxiliary source extensions |
| `artifact_ext` | yes | compiled artifact extension; may be empty string for JIT-only backends (e.g. Triton) |
| `hw_vendor` | yes | `nvidia \| amd \| intel \| apple \| cpu \| generic` |
| `threshold_profile` | yes | `diagnose.py` profile key (default `nvidia`) |
| `compiler` | yes | `{ name, invoke: "build.sh" }` |
| `runner` | yes | `{ invoke: "run.sh" }` |
| `profiler` | no | `{ name, invoke: "profile.sh", format, to_evidence }`; omit => no profiler => `unknown` |
| `capabilities.metrics` | yes | which canonical keys are honestly populated (subset of the 4 canonical keys) |
| `capabilities.bottleneck_classes` | yes | meaningful classes (subset of the 4; `unknown` implicit) |
| `capabilities.problem_types` | yes | supported problem types |
| `capabilities.precisions` | no | supported precisions |
| `requires_tools` | yes | preflight tools that must resolve |
| `optional_tools` | no | tools whose absence degrades gracefully |
| `vendor_patterns_file` | no | per-vendor anti-cheat regex file (`[fallback]`+`[skip]`) |
| `idioms` | yes | idioms filename (`idioms.json`) |
| `status` | yes | `stable \| experimental \| stub` |

## `idioms.json` fields

| field | req | meaning |
|---|---|---|
| `schema_version` | yes | idioms schema version (currently `1`) |
| `backend_id` | yes | canonical id (matches manifest) |
| `lang_fence` | yes | the ` ``` ` fence language for code blocks (`cuda`/`python`/`metal`/`cpp`) |
| `impl_requirements` | yes | ABI/scaffolding the executor must emit (e.g. `PYBIND11_MODULE`) |
| `unsupported_methods` | yes | `method_gate.TABLE` names this backend cannot honor (gated OUT) |
| `read_metric_guide` | no | per-backend causal model for reading profiler data |
| `methods.<name>.idiom` | yes | concrete construct for the abstract method `<name>` |
| `methods.<name>.prompt_guidance` | yes | instruction injected into the executor prompt |
| `methods.<name>.anti_idiom` | no | construct to avoid |
| `methods.<name>.triggers_on_native` | no | `backend_native` keys justifying the method |
| `methods.<name>.code_markers` | no | tokens the executor should emit (self-check) |

Every key under `methods` and every entry in `unsupported_methods` MUST be a real
`method_gate.TABLE` method name. `method_gate.py` is never edited.

## `to_evidence.py` -- the neutral interface

`to_evidence.py` is a pure function: native profiler output -> the canonical `metrics` dict that
`diagnose.py` consumes verbatim, plus a `_vendor` tag and a `coverage` list of honestly
populated keys.

### Canonical-unit table (every backend's `to_evidence` MUST honor)

| key | unit | example native source |
|---|---|---|
| `latency_ms` | milliseconds | duration (ns) / 1e6 |
| `dram_pct` | 0--100 | DRAM throughput pct-of-peak |
| `sm_pct` | 0--100 | SM throughput pct-of-peak |
| `occupancy` | **0--1** | warps-active pct-of-peak **/ 100** |

> The single most error-prone line: NCU reports occupancy as 0--100; `diagnose.py` compares
> `occ < threshold` on a 0--1 scale. `to_evidence.py` MUST divide by 100.

### The decided null rule

A backend that lacks a counter sets the key to **`null`** (JSON `null`) and omits it from
`coverage` -- never a fabricated `0.0`. A bottleneck branch may fire only when every metric it
compares is measured:
- `latency_occupancy` fires on `occupancy` alone;
- `compute_bound` fires on `sm_pct` alone;
- `memory_bound` and `overhead_bound` require **both** `dram_pct` and `sm_pct` measured;
- any case missing a required measured discriminator -> **`unknown`**, never a fabricated label.

So `{dram:80, sm:null}` -> `unknown`; `{dram:null, sm:80}` -> `compute_bound`;
`{dram:null, sm:30, occ:0.6}` -> `unknown`. (This rule is enforced by the SS5.3.2 `diagnose.py`
edit, delivered in Part A / P2 -- not in this part.)

## The three portability tiers

- **clean** -> `method_supported_backends: 'any'`; runs wherever a driver exists.
- **vendor_locked** -> e.g. `['cuda','triton']` plus a `requires_capability.metrics` floor.
  Locked to a *tool*, not a language.
- **method_intrinsic** -> a single-element hard whitelist (`['cpp']`/`['cutlass']`/`['rocm']`)
  plus pinned `requires_capability.problem_types`; the method *is* the backend;
  `matrix_eligible:false`.

## Conformance levels (L0--L3)

- **L0 -- Declared:** `manifest.json` validates; `backend_id == dir`; `idioms.json` references
  only real `method_gate.TABLE` names (including `unsupported_methods`);
  `capabilities.metrics` is a subset of the 4 canonical keys; `bottleneck_classes` is a subset
  of the 4 meaningful plus `{unknown}`. **Checked deterministically by
  `_substrate/backends/validate_backend.py`.**
- **L1 -- Buildable & Honest:** `build.sh`/`run.sh` executable; emit the contracted envelope +
  codes on a smoke fixture; `run.sh` output is accepted by `anti_cheat.py`; an incorrect
  fixture yields `valid:false`.
- **L2 -- Diagnostic:** `to_evidence.py` is pure (same native input => identical metrics; asserts
  `occupancy in [0,1]`); its output into `diagnose.py` yields a declared class; the decided null
  rule holds.
- **L3 -- Compounding:** every gated method for every declared `bottleneck_class` resolves to an
  `idioms.json` entry or `unsupported_methods`; a full round-trip produces a valid Layer-A
  envelope; `requires_tools` preflight resolves.

## Matrix-smoke contract (SS9.3)

Workflows declaring `matrix_eligible: true` (or `partial`) in their manifest's `backend:`
block participate in the matrix-smoke CI tier. For each eligible workflow x each matrix
driver, a trivial kernel is run with a **mock harness** (canned `build.sh`/`run.sh`/
`profile.sh` JSON without a GPU; fixtures in test infrastructure). Assertions are
**structural only**: guard passes, correct driver dispatched, Layer-A envelope conforms.

- `matrix_eligible: true` = run every `supported` x every matrix driver.
- `matrix_eligible: partial` = run a declared subset (e.g. `{cuda}` only).
- `matrix_eligible: false` = excluded (`method_intrinsic` always false).

Negative cells assert exact error substring/code: `method_intrinsic` throws the
intrinsic-reason; `vendor_locked` throws for illegal backends. Structural-only assertion
is sufficient; a real-hardware tier is opt-in behind a self-hosted-runner label.

The matrix-smoke test lives at `_meta/tools/test/matrix-smoke.test.js`.

## The scoped substrate edits (Part A / spec SS5.3) -- for cross-reference

Part B (this contract scaffolding) changes **no** substrate script. The driver axis does force
three scoped, default-`nvidia`, golden-tested substrate edits, delivered in **Part A / P2**:

1. `diagnose.py` -- a `_vendor` threshold-profile lookup (default `nvidia`).
2. `diagnose.py` -- null-vs-zero handling implementing the decided null rule above.
3. `anti_cheat.py` -- one optional `--vendor-patterns-file` argument feeding per-vendor
   `[fallback]` and `[skip]` regex sections.

`method_gate.py`, `evidence_schema.py`, `memory_store.py`, `verify_insight.py` stay
byte-identical.

## Deviations from the spec (recorded)

- **Machine-read driver files are JSON, not YAML.** Spec SS4.4/SS4.7 write `manifest.yaml`/
  `idioms.yaml`; Part B uses `manifest.json`/`idioms.json` so the validator parses them with
  stdlib `json` (Python has no stdlib YAML parser; we add no `pyyaml`/Node dependency). Field
  names are unchanged.
- **The L0 validator is Python in `_substrate/backends/`, not Node in `_meta/tools/`, and ships
  no JSON-Schema files.** Spec SS4.1/SS9.2 name `_meta/tools/validate-backend.js` + `_schema/`.
  Part B ships `_substrate/backends/validate_backend.py` with hand-rolled stdlib checks that
  `import method_gate` to read the live `TABLE`, keeping the validator + this doc as the
  contract. Any future generator (SS9.2) work expecting `_schema/*.json` must consume this
  Python validator instead. All spec references to `validate-backend.js` have been corrected
  to `validate_backend.py` (P5f).
- **L0 `backend_id` check is literal equality, not canonical-form.** Spec SS4.9 L0 requires
  `backend_id == dir == normalizeSuitabilityValue(id)` (a three-way equality pinning the id to
  post-normalize canonical form). This plan's validator asserts only `backend_id == basename(dir)`;
  it does NOT port `normalizeSuitabilityValue`'s lowercase/`_`->`-`/alias rules into Python. P3
  driver authoring MUST name driver dirs in already-canonical form (lowercase, `-` not `_`); the
  canonical-form assertion is a deferred L0 leg to add when the JS normalize logic is ported.
