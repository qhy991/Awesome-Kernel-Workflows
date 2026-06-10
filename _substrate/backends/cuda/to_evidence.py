#!/usr/bin/env python3
"""cuda/to_evidence.py — THIN wrapper over the shared NVIDIA mapper.

Spec 5.1 vendor-collapse: cuda and triton lower to PTX and are profiled by the same
ncu, so they share ONE mapping in backends/_evidence_nvidia.py. The spec's literal
`from ..cuda.to_evidence import main` relative import fails for this PATH-invoked
standalone script (no package context), so we sys.path-insert the backends dir and
import the shared mapper explicitly.

Invoked WITH a python prefix:
  to_evidence.py --native <ncu.csv|prof.sqlite> [--source-backend <id>] [--format ncu-csv|nsys-sqlite] [--run <result.json>]
"""
import os, sys

_BACKENDS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # _substrate/backends
if _BACKENDS not in sys.path:
    sys.path.insert(0, _BACKENDS)

import _evidence_nvidia  # noqa: E402

if __name__ == "__main__":
    sys.exit(_evidence_nvidia.main(source_backend="cuda"))
