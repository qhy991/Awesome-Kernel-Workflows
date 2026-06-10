# Backend Driver Registry

This is the human-facing index of backend drivers under `_substrate/backends/`. A backend
driver is the `(source language) x (hardware/profiler vendor)` translation layer that adapts
native backend tooling to the universal substrate vocabulary (see
[`../BACKEND-DRIVER-SDK.md`](../BACKEND-DRIVER-SDK.md) for the full contract). Each row maps a
canonical `backend id` (the `normalizeSuitabilityValue` form, which is also the driver
directory name) to its directory, hardware vendor, lifecycle status, and owner.

Add a row when you start a driver. Move it to `stable` only after it passes L0--L3 conformance
(`validate_backend.py` for L0; see the SDK doc for the full ladder).

| backend id | dir | hw_vendor | status | owner |
|---|---|---|---|---|
| cuda | `cuda/` | nvidia | experimental | (unassigned) |
| triton | `triton/` | nvidia | experimental | (unassigned) |

> **Note (P3):** the `cuda` and `triton` `build.sh`/`run.sh`/`profile.sh` are
> **GPU-untested** -- this repo runs on macOS where `nvcc`/`ncu`/`triton` are absent. What
> IS verified on macOS: `validate_backend.py` (L0) for both dirs, and `to_evidence.py`
> (the shared `_evidence_nvidia.py` NCU-to-canonical mapping, incl. `occupancy = warps / 100`
> and `dram_pct = read + write`) via fake-tool PATH stubs. The `.sh` scripts have
> arg-parsing, JSON-envelope, and exit-code coverage only. End-to-end compile/run/profile
> is **deferred to the GPU/CI tier** (spec SS8.3, SS9.3).

**Status vocabulary:** `planned` (row reserved, no files yet) . `stub` . `experimental` .
`stable` (L0--L3 conformant). `status` here is the registry lifecycle and is distinct from the
per-manifest `status` field, which only ranges over `stub | experimental | stable`.

---

## cuda -- NVIDIA CUDA C++

| Property | Value |
|---|---|
| **Directory** | `_substrate/backends/cuda/` |
| **Source extension** | `.cu` |
| **Artifact extension** | `.so` |
| **Hardware vendor** | nvidia |
| **Compiler** | `nvcc` (`build.sh`) |
| **Profiler** | `ncu` (Nsight Compute) via `profile.sh`; **`nsys` fallback** when `ncu` absent |
| **Profiler format** | `ncu-csv` (preferred) · `nsys-sqlite` (fallback) |
| **Threshold profile** | `nvidia` |
| **Status** | experimental |

### Emitted metric names

`to_evidence.py` delegates to the shared `_evidence_nvidia.py` mapper. Canonical metrics:

| Canonical key | NCU counter source | Unit |
|---|---|---|
| `latency_ms` | `gpu__time_duration.sum` (ns / 1e6) | milliseconds |
| `dram_pct` | `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` + `dram__bytes_write.sum.pct_of_peak_sustained_elapsed` | 0--100 |
| `sm_pct` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 0--100 |
| `occupancy` | `sm__warps_active.avg.pct_of_peak_sustained_elapsed` / 100 | 0--1 |

`backend_native` may include `l2_hit_pct`, `sectors_per_req`, and per-line stall data
when `-lineinfo` is passed to `nvcc`.

### nsys fallback (when `ncu` unavailable)

`profile.sh` tries `ncu` first; if absent, falls back to `nsys profile` when a runnable
`--source` launcher or executable `--artifact` is provided. `--out` must end with `.sqlite`.

| Canonical key | nsys source | Unit |
|---|---|---|
| `latency_ms` | dominant kernel `AVG(end-start)` from `CUPTI_ACTIVITY_KIND_KERNEL` (ns / 1e6) | milliseconds |
| `dram_pct` | **always null** — nsys has no DRAM throughput counter | — |
| `sm_pct` | **always null** — nsys has no SM throughput counter | — |
| `occupancy` | **always null** — nsys has no achieved-occupancy counter | — |

Only `latency_ms` enters `coverage`; `diagnose.py` yields `unknown` (honest degradation).
Use nsys for kernel timing / launch-overhead triage; use `ncu` when hardware counters are needed.

### Fallback patterns

Uses the substrate default `FALLBACK_PATTERNS` (no `vendor_patterns_file` in manifest):
`cublas`, `cudnn`, `torch.matmul`, `F.linear`, `torch.nn.functional`, `at::matmul`.

### Lang fence and idiom highlights

- `lang_fence`: `cuda`
- `impl_requirements`: PYBIND11_MODULE with `forward()` binding
- All `method_gate.TABLE` methods supported (`unsupported_methods: []`)
- Idiom examples: `tensor_core_mma` -> `wmma / mma.sync`, `shared_memory_tiling` ->
  `__shared__ double-buffered tiles`, `vectorized_load_store` -> `float4 / int4`

---

## triton -- Triton (NVIDIA, @triton.jit)

| Property | Value |
|---|---|
| **Directory** | `_substrate/backends/triton/` |
| **Source extension** | `.py` |
| **Artifact extension** | (empty -- JIT-only, no persistent artifact) |
| **Hardware vendor** | nvidia |
| **Compiler** | Triton JIT (`build.sh` triggers warmup to materialize PTX) |
| **Profiler** | `proton` (Triton's own `triton.profiler`, NOT ncu) via `profile.sh` |
| **Profiler format** | `proton-hatchet` |
| **Threshold profile** | `nvidia` |
| **Status** | experimental |

### Why Proton instead of ncu

ncu mangles the Triton JIT kernel symbol as `triton_<fn>_<hash>` (so `ncu -k` can't target
it statically) and needs elevated perf-counter access (`perf_event_paranoid` / sudo). Proton
instruments **inside Triton**, names kernels directly, and uses CUPTI device timing — no
kernel-name regex and no sudo. It emits a `.hatchet` JSON profile.

### Emitted metric names

`to_evidence.py` is a **standalone** proton-hatchet parser (it does NOT delegate to
`_evidence_nvidia.py`; that mapper is ncu-CSV-specific and stays cuda-only). It lowers the
proton profile onto the canonical keys via a device-derived roofline:

| Canonical key | Proton source | Unit |
|---|---|---|
| `latency_ms` | leaf `time (ns)` / `count` / 1e6 (per-invocation device time) | milliseconds |
| `dram_pct` | `bytes` ÷ latency ÷ peak_BW × 100, peak_BW = `memory_clock_rate`·2·(`bus_width`/8) | 0–~ (MAY exceed 100) |
| `sm_pct` | `flops` ÷ latency ÷ peak_FLOP × 100, peak_FLOP = `num_sms`·fp32_cores·2·`clock_rate` | 0–~ |
| `occupancy` | **always null** — CUPTI/proton does not report achieved occupancy | — |

`dram_pct` / `sm_pct` are produced ONLY when the launcher annotates the proton scope with
`bytes` / `flops` (the Triton-idiomatic roofline annotation); otherwise they are null and
omitted from `coverage` (null rule), so `diagnose.py` yields `unknown` rather than a wrong
label. Because `occupancy` is never measured, triton declares no `latency_occupancy`
capability (`capabilities.metrics` = `{latency_ms, dram_pct, sm_pct}`).

These are **device-derived roofline estimates**, not hardware counters (marked
`backend_native.estimated`); they are weaker than CUDA's ncu counters but require no sudo and
correctly target the JIT kernel.

### profile.sh launcher contract

Because proton must actually run the kernel, `profile.sh --source <launcher.py>` expects a
plain Python module exposing `make_inputs()` (or `make_inputs_from_problem(problem)`) and
`forward(*inputs)` / `launch(*inputs)`, with optional `BYTES` / `FLOPS` module attributes (or
`problem["bytes"]` / `problem["flops"]`) for the roofline annotation. Absent that contract
(or absent a CUDA device / triton runtime), `profile.sh` degrades to exit 4
(profiler unavailable → `unknown`).

### Fallback patterns

Uses the substrate default `FALLBACK_PATTERNS` (no `vendor_patterns_file` in manifest).

### No ncu kernel-name problem

Because Proton instruments inside Triton, it attributes timing to the kernel by its Python
name directly — there is no `triton_<fn>_<hash>` mangling to discover and no `ncu -k`
regex. The optional `--kernel-name` arg only labels the proton scope.

### Lang fence and idiom highlights

- `lang_fence`: `python`
- `impl_requirements`: `@triton.jit` kernel + plain Python launcher, no PYBIND11
- All `method_gate.TABLE` methods supported (`unsupported_methods: []`)
- Idiom examples: `tensor_core_mma` -> `tl.dot`, `shared_memory_tiling` ->
  `num_stages` (auto-managed SMEM via compiler), `vectorized_load_store` ->
  `block-shaped tl.load / tl.store`

### Key differences from cuda (same vendor, different language)

| Aspect | cuda | triton |
|---|---|---|
| Source language | C++ (`.cu`) | Python (`.py`) |
| Build model | `nvcc` explicit compilation -> `.so` | JIT warmup, PTX cached in `TRITON_CACHE_DIR` |
| ABI requirement | `PYBIND11_MODULE` + `forward()` | Plain Python launcher, returns tensor |
| Profiler | `ncu` (hardware counters) | `proton` (CUPTI timing + roofline) |
| Kernel targeting | `ncu -k <KERNEL_NAME_REGEX>` | Proton names the kernel directly (no regex) |
| dram_pct / sm_pct | direct ncu counters | device-derived roofline estimate (needs bytes/flops) |
| occupancy | ncu `sm__warps_active` | not available (always null) |
| Shared memory | Explicit `__shared__` double-buffering | Compiler-managed via `num_stages` |
