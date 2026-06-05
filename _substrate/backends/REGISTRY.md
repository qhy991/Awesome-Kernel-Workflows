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
| cuda | `cuda/` | nvidia | planned | (unassigned) |

**Status vocabulary:** `planned` (row reserved, no files yet) · `stub` · `experimental` ·
`stable` (L0–L3 conformant). `status` here is the registry lifecycle and is distinct from the
per-manifest `status` field, which only ranges over `stub | experimental | stable`.
