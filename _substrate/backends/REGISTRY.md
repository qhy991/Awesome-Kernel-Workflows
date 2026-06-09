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
| rocm | `rocm/` | amd | experimental | (unassigned) |

> **Note (P3):** the `cuda`, `triton`, and `rocm` `build.sh`/`run.sh`/`profile.sh` are
> **GPU-untested** -- this repo runs without NVIDIA/AMD hardware, so `nvcc`/`ncu`/`triton`/
> `hipcc`/`rocprofv3` are absent. What IS verified locally: `validate_backend.py` (L0) for
> all three dirs, and `to_evidence.py` (the shared `_evidence_nvidia.py` NCU mapping for
> cuda/triton; the shared `_evidence_amd.py` rocprof mapping for rocm). The `.sh` scripts
> have arg-parsing, JSON-envelope, and exit-code coverage only. End-to-end compile/run/
> profile is **deferred to the GPU/CI tier** (spec SS8.3, SS9.3).

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
| **Profiler** | `ncu` (Nsight Compute) via `profile.sh` |
| **Profiler format** | `ncu-csv` |
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
| **Profiler** | `ncu` (same profiler as cuda) via `profile.sh` |
| **Profiler format** | `ncu-csv` |
| **Threshold profile** | `nvidia` |
| **Status** | experimental |

### Emitted metric names

`to_evidence.py` delegates to the same shared `_evidence_nvidia.py` mapper as cuda
(spec SS5.1 vendor-collapse). Canonical metrics are identical to cuda above.

`backend_native` source-attributed fields (sectors/request, per-line stalls) may be
**weaker or absent** under Triton because Triton's source attribution through NCU is
less reliable than CUDA's `-lineinfo` path. The vendor-collapse is in the
metric/diagnosis layer only; source-line evidence is not symmetric.

### Fallback patterns

Uses the substrate default `FALLBACK_PATTERNS` (no `vendor_patterns_file` in manifest).

### Profiler caveat: kernel-name auto-discovery

Triton mangles kernel names as `triton_<fn>_<hash>`. The `profile.sh` script discovers
the kernel name by globbing `TRITON_CACHE_DIR` after JIT warmup, rather than relying on
a user-supplied `KERNEL_NAME_REGEX` as cuda does with `ncu -k`. Correctness of this
auto-discovery on real hardware is deferred to the GPU CI tier.

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
| Kernel targeting for ncu | `ncu -k <KERNEL_NAME_REGEX>` | Auto-discovery from JIT cache |
| Source-line attribution | Full with `-lineinfo` | Partial/weaker |
| Shared memory | Explicit `__shared__` double-buffering | Compiler-managed via `num_stages` |

---

## rocm -- AMD HIP / ROCm

| Property | Value |
|---|---|
| **Directory** | `_substrate/backends/rocm/` |
| **Source extension** | `.hip` |
| **Artifact extension** | `.so` |
| **Hardware vendor** | amd |
| **Compiler** | `hipcc` (`build.sh`) |
| **Profiler** | `rocprofv3` (ROCm 6.x+) or `rocprof` (fallback) via `profile.sh` |
| **Profiler format** | `rocprof-csv` |
| **Threshold profile** | `amd` |
| **Status** | experimental |

### Emitted metric names

`to_evidence.py` delegates to the shared `_evidence_amd.py` mapper. Canonical metrics:

| Canonical key | rocprof counter source | Unit |
|---|---|---|
| `latency_ms` | `KernelDuration` (ns / 1e6) | milliseconds |
| `dram_pct` | `MemUnitBusy` | 0--100 |
| `sm_pct` | `VALUBusy` | 0--100 |
| `occupancy` | `Wavefronts / MaxWavefronts` | 0--1 |

`backend_native` may include additional counters (L2CacheHit, FetchSize, WriteSize, etc.)
depending on which pmc counters the user requests.

### GPU architecture defaults

- Default `--arch`: `gfx942` (MI300X / CDNA3). Use `gfx90a` for MI200/MI250,
  `gfx1100` for RDNA3 desktop, `gfx1200` for RDNA4.
- Wavefront size: 64 on CDNA (MI200/MI300), 32 on RDNA (7900 XTX / gfx11+).
- LDS: 64 KB per CU on CDNA; shared memory idioms must respect this budget.
- VGPRs: 512 per SIMD on CDNA3; register-tiling tiles must fit within this.

### Fallback patterns

Uses the substrate default `FALLBACK_PATTERNS` (no `vendor_patterns_file` in manifest):
`hipblaslt`, `rocblas`, `torch.matmul`, `F.linear`, `torch.nn.functional`, `at::matmul`.

### Profiler caveat: rocprofv3 vs rocprof

`profile.sh` prefers `rocprofv3` (ROCm 6.x+), falling back to `rocprof` (v1/v2) if v3
is not on PATH. The CSV column headers differ between versions — `_evidence_amd.py` handles
both via case-insensitive column lookup and long/wide format auto-detection. GPU-tier
confirmation is deferred (same status as cuda/triton).

### Lang fence and idiom highlights

- `lang_fence`: `cpp` (HIP is C++ with AMD extensions)
- `impl_requirements`: PYBIND11_MODULE with `forward()` binding (same ABI as cuda)
- All `method_gate.TABLE` methods supported (`unsupported_methods: []`)
- Idiom examples: `tensor_core_mma` -> `MFMA intrinsics (__builtin_amdgcn_mfma_f32_*)`,
  `shared_memory_tiling` -> `__shared__ (LDS) tiling`, `vectorized_load_store` -> `float4`
  / AMD global_load_lds

### Key differences from cuda (different vendor, similar language)

| Aspect | cuda | rocm |
|---|---|---|
| Source language | C++ (`.cu`) | C++ (`.hip` / `.hip.cpp`) |
| Compiler | `nvcc` | `hipcc` (Clang-based) |
| Arch flag | `-arch=sm_80` | `--offload-arch=gfx942` |
| Debug info for profiler | `-lineinfo` | `-gline-tables-only` |
| Profiler | `ncu` (Nsight Compute) | `rocprofv3` / `rocprof` / `omniperf` |
| Profiler output | NCU CSV | rocprof CSV |
| Tensor core intrinsics | `wmma` / `mma.sync` | `MFMA` / `WMMA` (RDNA3) |
| Shared memory | `__shared__` (up to 164 KB on Hopper) | `__shared__` = LDS (64 KB/CU on CDNA) |
| Wavefront size | 32 (warp) | 64 (CDNA) / 32 (RDNA) |
| Graph API | CUDA Graphs | HIP Graphs (`hipGraphLaunch`) |
| Vendor GEMM library | cuBLAS | rocBLAS / hipBLASLt |
