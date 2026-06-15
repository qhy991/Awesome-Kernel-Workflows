#!/usr/bin/env python3
"""ascend/to_evidence.py -- THIN wrapper over the shared Ascend msprof mapper.

Mirrors cuda/to_evidence.py and rocm/to_evidence.py. The single real mapping lives in
backends/_evidence_ascend.py; this wrapper sys.path-inserts the backends dir and calls
main() with source_backend="ascend".

Invoked WITH a python prefix:
  to_evidence.py --native <msprof.csv|-> [--source-backend ascend] [--format msprof-csv] [--run <result.json>]
"""
import os, sys

_BACKENDS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # _substrate/backends
if _BACKENDS not in sys.path:
    sys.path.insert(0, _BACKENDS)

import _evidence_ascend  # noqa: E402

if __name__ == "__main__":
    sys.exit(_evidence_ascend.main(source_backend="ascend"))
