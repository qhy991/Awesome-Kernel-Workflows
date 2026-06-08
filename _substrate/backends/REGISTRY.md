# Backend Driver Registry

This is the human-facing index of backend drivers under `_substrate/backends/`. A backend
driver is the `(source language) × (hardware/profiler vendor)` translation layer that adapts
native backend tooling to the universal substrate vocabulary (see
[`../BACKEND-DRIVER-SDK.md`](../BACKEND-DRIVER-SDK.md) for the full contract). Each row maps a
canonical `backend id` (the `normalizeSuitabilityValue` form, which is also the driver
directory name) to its directory, hardware vendor, lifecycle status, and owner.

Add a row when you start a driver. Move it to `stable` only after it passes L0–L3 conformance
(`validate_backend.py` for L0; see the SDK doc for the full ladder).

| backend id | dir | hw_vendor | status | owner |
|---|---|---|---|---|
| cuda | `cuda/` | nvidia | experimental | (unassigned) |
| triton | `triton/` | nvidia | experimental | (unassigned) |

> **Note (P3):** the `cuda` and `triton` `build.sh`/`run.sh`/`profile.sh` are
> **GPU-untested** — this repo runs on macOS where `nvcc`/`ncu`/`triton` are absent. What
> IS verified on macOS: `validate_backend.py` (L0) for both dirs, and `to_evidence.py`
> (the shared `_evidence_nvidia.py` NCU→canonical mapping, incl. `occupancy = warps ÷ 100`
> and `dram_pct = read + write`) via fake-tool PATH stubs. The `.sh` scripts have
> arg-parsing, JSON-envelope, and exit-code coverage only. End-to-end compile/run/profile
> is **deferred to the GPU/CI tier** (spec §8.3, §9.3).

**Status vocabulary:** `planned` (row reserved, no files yet) · `stub` · `experimental` ·
`stable` (L0–L3 conformant). `status` here is the registry lifecycle and is distinct from the
per-manifest `status` field, which only ranges over `stub | experimental | stable`.
