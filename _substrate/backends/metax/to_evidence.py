#!/usr/bin/env python3
"""metax/to_evidence.py — THIN wrapper over the MetaX mcprof-CSV mapper.

MetaX uses mcTracer + mcProfiler (MACA SDK) which produce proprietary CSV output.
The format differs from NVIDIA ncu-csv, so this driver uses its own standalone mapper
(_evidence_metax.py) rather than the shared _evidence_nvidia.py.

Invoked WITH a python prefix:
  to_evidence.py --native <mcprof.csv|mctrace.log> [--source-backend <id>] [--format mcprof-csv|mctrace-log] [--run <result.json>]
"""
import os, sys

_BACKENDS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # _substrate/backends
if _BACKENDS not in sys.path:
    sys.path.insert(0, _BACKENDS)

import _evidence_metax  # noqa: E402

if __name__ == "__main__":
    sys.exit(_evidence_metax.main(source_backend="metax"))