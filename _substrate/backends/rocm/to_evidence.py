#!/usr/bin/env python3
"""rocm/to_evidence.py -- THIN wrapper over the shared AMD mapper.

Mirrors cuda/to_evidence.py. The single real mapping lives in
backends/_evidence_amd.py; this wrapper sys.path-inserts the backends dir and
calls main() with source_backend="rocm".

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
