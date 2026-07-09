#!/usr/bin/env bash
# generic/profile.sh — the generic substrate has NO vendor profiler.
# Emits the honest "no profiler" pointer. Per the manifest contract (profiler
# name "none"), an omitted/absent profiler is a legal, honest state: diagnose.py
# will classify the bottleneck as 'unknown' rather than inventing one.
# Prints the POINTER (not metrics), matching the other drivers' envelope:
#   {ok:false, profiler:"none", native_profile:null, error:"..."}
# exit 4 profiler unavailable (this is the ONLY outcome — there is no profiler).
set -u

emit() { printf '%s\n' "$1"; }

json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

# Parse (and ignore) the standard profile args so a caller using the uniform
# invocation contract gets a clean envelope, not an arg-parse crash.
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact|--problem|--out|--source) shift 2 ;;
    *) shift ;;
  esac
done

MSG="generic substrate has no vendor profiler (dram_pct/sm_pct/occupancy unavailable); wall-clock latency from run.sh is the only signal — diagnose classifies bottleneck as 'unknown'"
ESC_MSG="$(json_escape "$MSG")"
emit "{\"ok\":false,\"profiler\":\"none\",\"native_profile\":null,\"error\":$ESC_MSG}"
exit 4
