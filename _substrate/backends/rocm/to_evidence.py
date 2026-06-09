#!/usr/bin/env python3
"""rocm/to_evidence.py — THIN wrapper over the shared AMD mapper.

Parallel to cuda/to_evidence.py. The rocm driver delegates to the shared
backends/_evidence_amd.py mapper so any future AMD-language backend (e.g. a
hypothetical OpenCL-on-AMD driver) can reuse the same rocprof CSV parser.

Invoked WITH a python prefix:
  to_evidence.py --native <rocprof.csv|-> [--source-backend <id>] [--format rocprof-csv] [--run <result.json>]
"""
import os, sys

_BACKENDS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # _substrate/backends
if _BACKENDS not in sys.path:
    sys.path.insert(0, _BACKENDS)

import _evidence_amd  # noqa: E402

if __name__ == "__main__":
    sys.exit(_evidence_amd.main(source_backend="rocm"))
