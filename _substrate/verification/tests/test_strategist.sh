#!/bin/bash
# Durable test for the shared verification-strategist (generator verification-depth
# router). No GPU / no harness needed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
P="$HERE/.."                                   # _substrate/verification
SUB="$P/.."                                     # _substrate
ST="python3 $P/verification_strategist.py"
SCHEMA="python3 $SUB/evidence_schema.py validate"
cd "$HERE"
rm -f cache.json traj.jsonl
pass=0; fail=0
g(){ echo "$1" | python3 -c "import sys,json;print(json.load(sys.stdin).get('$2'))"; }
chk(){ if [ "$2" = "$3" ]; then echo "  PASS  $1 ($2)"; pass=$((pass+1)); else echo "  FAIL  $1: got [$2] want [$3]"; fail=$((fail+1)); fi; }

# S1: confirm + full host => reference_test (measured, correctness)
J=$($ST resolve --need confirm --host-probe '{"compiler":true,"harness_runnable":true,"reference_available":true}')
chk "S1 confirm->reference_test" "$(g "$J" method)" reference_test
chk "S1 conf measured"           "$(g "$J" confidence)" measured
chk "S1 evidence correctness"    "$(g "$J" evidence_source)" correctness

# S2: confirm but NO reference => smoke_test (measured, runtime)
J=$($ST resolve --need confirm --host-probe '{"compiler":true,"harness_runnable":true,"reference_available":false}')
chk "S2 confirm no-ref->smoke"   "$(g "$J" method)" smoke_test
chk "S2 evidence runtime"        "$(g "$J" evidence_source)" runtime
chk "S2 abstained reference"     "$(g "$J" abstained_from)" "['reference_test']"

# S3: confirm but not runnable => compile_lint (inferred, compile)
J=$($ST resolve --need confirm --host-probe '{"compiler":true,"harness_runnable":false,"reference_available":false}')
chk "S3 confirm no-run->compile" "$(g "$J" method)" compile_lint
chk "S3 inferred"                "$(g "$J" confidence)" inferred

# S4: PRUNE never executes -> compile_lint even when harness+reference present
J=$($ST resolve --need prune --host-probe '{"compiler":true,"harness_runnable":true,"reference_available":true}')
chk "S4 prune->compile_lint"     "$(g "$J" method)" compile_lint
chk "S4 prune inferred"          "$(g "$J" confidence)" inferred
chk "S4 prune abstained exec"    "$(g "$J" abstained_from)" "[]"

# S5: prune + no compiler => static (hypothesized)
J=$($ST resolve --need prune --host-probe '{"compiler":false,"harness_runnable":false,"reference_available":false}')
chk "S5 prune no-compile->static" "$(g "$J" method)" static
chk "S5 hypothesized"            "$(g "$J" confidence)" hypothesized

# S6: reproducibility + cache hit
D1=$($ST resolve --need confirm --host-probe '{"compiler":true,"harness_runnable":true,"reference_available":true}' --cache cache.json)
D2=$($ST resolve --need confirm --host-probe '{"compiler":true,"harness_runnable":true,"reference_available":true}' --cache cache.json)
chk "S6 identical" "$(g "$D1" method)|$(g "$D1" cache_key)" "$(g "$D2" method)|$(g "$D2" cache_key)"
chk "S6 cache hit" "$(g "$D2" _cache)" hit

# S7: a decided method's evidence must be schema-legal
EV=$(g "$D1" evidence_source)
python3 -c "import json;print(json.dumps({'attempt_id':'a1','parent_id':None,'compiled':True,'correct':True,'speedup':None,'metrics':{},'best_kernel':None,'convergence_status':'unknown','insights':[{'kind':'bottleneck','directive':'explore','evidence':'$EV','confidence':'measured','claim':'x'}],'failed_strategies':[]}))" | $SCHEMA - >/dev/null 2>&1
chk "S7 evidence schema-legal" "$?" 0
rm -f cache.json traj.jsonl
echo "================  $pass passed, $fail failed  ================"
exit $fail
