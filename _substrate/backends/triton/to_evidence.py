#!/usr/bin/env python3
"""triton/to_evidence.py — THIN wrapper over the shared NVIDIA mapper.

Spec 5.1 vendor-collapse: triton lowers to PTX and is profiled by the same ncu as
cuda, so it reuses backends/_evidence_nvidia.py verbatim — only the source_backend id
differs. (The spec's `from ..cuda.to_evidence import main` relative import fails for
this PATH-invoked standalone script; we sys.path-insert the backends dir instead.)

Note: the Triton-specific profiling ENV (TRITON_CACHE_DIR etc.) lives in profile.sh,
NOT here — the shared mapper concerns only the 3 canonical classifier counters
(spec 5.1 caveat). backend_native source-attributed fields may be weaker under Triton.

Invoked WITH a python prefix:
  to_evidence.py --native <ncu.csv|-> [--source-backend <id>] [--format ncu-csv] [--run <result.json>]
"""
import os, sys

_BACKENDS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # _substrate/backends
if _BACKENDS not in sys.path:
    sys.path.insert(0, _BACKENDS)

import _evidence_nvidia  # noqa: E402

if __name__ == "__main__":
    sys.exit(_evidence_nvidia.main(source_backend="triton"))
