#!/bin/bash
# Durable test for the shared profiling-strategist. Uses the REAL backend manifests
# and a captured llama.cpp FLASH_ATTN_EXT perf fixture (no GPU / no llama.cpp needed).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
P="$HERE/.."                                   # _substrate/profiling
SUB="$P/.."                                     # _substrate
B="$SUB/backends"
ST="python3 $P/profiling_strategist.py"
SCHEMA="python3 $SUB/evidence_schema.py validate"
cd "$HERE"
rm -f cache.json traj.jsonl
echo '{}' > empty_manifest.json
echo '{"backend_id":"sycl","profiler":{"name":"sycl-prof","invoke":"profile.sh"},"capabilities":{"metrics":{"latency_ms":true}}}' > sycl_manifest.json
pass=0; fail=0
g(){ echo "$1" | python3 -c "import sys,json;print(json.load(sys.stdin).get('$2'))"; }
chk(){ if [ "$2" = "$3" ]; then echo "  PASS  $1 ($2)"; pass=$((pass+1)); else echo "  FAIL  $1: got [$2] want [$3]"; fail=$((fail+1)); fi; }

J=$($ST resolve --backend-manifest $B/cuda/manifest.json --task gemm --host-probe '{"ncu":true}')
chk "S1 cuda+ncu->native"        "$(g "$J" method)" native_profiler
chk "S1 stamp measured"          "$(g "$J" confidence)" measured
J=$($ST resolve --backend-manifest $B/cuda/manifest.json --task attention --host-probe '{"ncu":false,"nsys":false}')
chk "S2 cuda/no-ncu->perf"       "$(g "$J" method)" perf_heuristic
chk "S2 stamp inferred"          "$(g "$J" confidence)" inferred
J=$($ST resolve --backend-manifest $B/metal/manifest.json --task attention --host-probe '{"xcrun":true}')
chk "S3 metal latency-only->perf" "$(g "$J" method)" perf_heuristic
J=$($ST resolve --backend-manifest $B/cuda/manifest.json --task elementwise --size tiny --host-probe '{"ncu":true}')
chk "S4 tiny->latency+occ"       "$(g "$J" requested_metrics)" "['latency_ms', 'occupancy']"
J=$($ST resolve --backend-manifest $B/cuda/manifest.json --task gemm --host-probe '{"ncu":false,"nsys":false,"throughput_runnable":false}')
chk "S5 nothing->static"         "$(g "$J" method)" static
J=$($ST resolve --backend-manifest $B/rocm/manifest.json --task gemm --host-probe '{"rocprofv3":true}')
chk "S6 rocm->native(from manifest)" "$(g "$J" profiler_name)" rocprofv3
chk "S6 rocm evidence=native_profiler (not借用 ncu)" "$(g "$J" evidence_source)" native_profiler
J=$($ST resolve --backend-manifest $B/ascend/manifest.json --task attention --host-probe '{"msprof":false}')
chk "S7 ascend/no-msprof->perf"  "$(g "$J" method)" perf_heuristic
J=$($ST resolve --backend-manifest sycl_manifest.json --task gemm --host-probe '{"sycl-prof":false}')
chk "S8 NEW backend w/manifest->perf (generality)" "$(g "$J" method)" perf_heuristic
J=$($ST resolve --backend-manifest empty_manifest.json --task gemm)
chk "S9 no manifest->derive (autonomy)" "$(g "$J" method)" derive_adapter
D1=$($ST resolve --backend-manifest $B/cuda/manifest.json --task attention --host-probe '{"ncu":false}' --cache cache.json)
D2=$($ST resolve --backend-manifest $B/cuda/manifest.json --task attention --host-probe '{"ncu":false}' --cache cache.json)
chk "S10 reproducible+cache hit" "$(g "$D2" _cache)" hit
$ST run --backend-manifest $B/cuda/manifest.json --task attention --host-probe '{"ncu":false}' \
   --baseline fixture_perf_baseline.txt --candidate fixture_perf_candidate.txt >s_ev.json 2>/dev/null
$SCHEMA s_ev.json >/dev/null 2>&1
chk "S11 e2e -> schema-legal evidence" "$?" 0
chk "S11 evidence source"        "$(python3 -c 'import json;print([i["evidence"] for i in json.load(open("s_ev.json"))["insights"] if i["kind"]=="bottleneck"][0])')" profile_heuristic

# #36: perf_heuristic honors the 'no Nsight' contract — used_tools=[] (no nsys/
# cuobjdump borrow even when probed available), and deny_tools forces perf_heuristic
# regardless of host probe.
J=$($ST resolve --backend-manifest $B/cuda/manifest.json --task gemm --host-probe '{"ncu":false,"nsys":true}')
chk "S12 cuda/no-ncu+nsys->perf" "$(g "$J" method)" perf_heuristic
chk "S12 used_tools empty (no Nsight borrow)" "$(g "$J" used_tools)" "[]"
J=$($ST resolve --backend-manifest $B/cuda/manifest.json --task gemm --host-probe '{"ncu":true,"nsys":true}')
chk "S13 cuda+ncu+nsys->native"  "$(g "$J" method)" native_profiler
chk "S13 used_tools ncu+nsys"    "$(g "$J" used_tools)" "['ncu', 'nsys']"
J=$($ST resolve --backend-manifest $B/cuda/manifest.json --task gemm --host-probe '{"ncu":true,"nsys":true}' --deny-tools ncu,nsys,cuobjdump)
chk "S14 deny_tools->perf"       "$(g "$J" method)" perf_heuristic
chk "S14 used_tools empty (denied)" "$(g "$J" used_tools)" "[]"

rm -f empty_manifest.json sycl_manifest.json cache.json traj.jsonl s_ev.json
echo "================  $pass passed, $fail failed  ================"
exit $fail
