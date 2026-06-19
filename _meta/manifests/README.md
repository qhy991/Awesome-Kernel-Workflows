# DEPRECATED — removed

Per-workflow **`manifest.yaml`** at `<Workflow>/manifest.yaml` is the source of truth.

- **Schema reference:** [`docs/manifest-schema.yaml`](../docs/manifest-schema.yaml)
- **Routing contract:** top-level `routing:` block (and `variants[].routing` for multi-entrypoint dirs)
- **Validation:** `./scripts/validate-manifests.sh`

Legacy copies in this directory were deleted in the manifest SoT consolidation (2026-06).
