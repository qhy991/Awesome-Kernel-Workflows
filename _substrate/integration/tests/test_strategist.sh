#!/bin/bash
# Durable test for the shared integration-strategist (build/integration-mode router).
# No GPU / no project needed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
P="$HERE/.."                                   # _substrate/integration
SUB="$P/.."                                     # _substrate
ST="python3 $P/integration_strategist.py"
cd "$HERE"
rm -f cache.json traj.jsonl
pass=0; fail=0
g(){ echo "$1" | python3 -c "import sys,json;print(json.load(sys.stdin).get('$2'))"; }
chk(){ if [ "$2" = "$3" ]; then echo "  PASS  $1 ($2)"; pass=$((pass+1)); else echo "  FAIL  $1: got [$2] want [$3]"; fail=$((fail+1)); fi; }

# S1: kernel CAN compile standalone + compiler present -> standalone (isolated)
J=$($ST resolve --can-standalone yes --host-probe '{"compiler":true}')
chk "S1 standalone-capable->standalone" "$(g "$J" method)" standalone
chk "S1 fidelity isolated"             "$(g "$J" build_fidelity)" isolated
chk "S1 reversible"                    "$(g "$J" reversible)" True

# S2: llama.cpp mmq.cuh — CANNOT compile standalone; full embedded host -> embedded_dispatch (production, non-mutating)
J=$($ST resolve --can-standalone no --host-probe '{"compiler":true,"project_build":true,"register_script":true,"runtime_registry":false,"reversibility_net":true}')
chk "S2 llama.cpp->embedded_dispatch"  "$(g "$J" method)" embedded_dispatch
chk "S2 fidelity production"           "$(g "$J" build_fidelity)" production
chk "S2 reversible"                    "$(g "$J" reversible)" True
chk "S2 eval mechanism"                "$(g "$J" eval_mechanism)" "embedded/embedded_eval.js (register -> build -> test -> bench -> unregister)"

# S3: non-standalone, NO register_script (no adapter) but project build + reversibility -> embedded_inplace
J=$($ST resolve --can-standalone no --host-probe '{"compiler":true,"project_build":true,"register_script":false,"runtime_registry":false,"reversibility_net":true}')
chk "S3 no-adapter->embedded_inplace"  "$(g "$J" method)" embedded_inplace
chk "S3 fidelity project_native"       "$(g "$J" build_fidelity)" project_native
chk "S3 abstained dispatch"            "$(g "$J" abstained_from)" "['embedded_dispatch']"

# S4: non-standalone, only runtime_registry (vLLM-style) -> registry_dispatch
J=$($ST resolve --can-standalone no --host-probe '{"compiler":true,"project_build":false,"register_script":false,"runtime_registry":true,"reversibility_net":false}')
chk "S4 runtime-only->registry_dispatch" "$(g "$J" method)" registry_dispatch

# S5: non-standalone, bare host (no project build/adapter/registry) -> derive_adapter (autonomy)
J=$($ST resolve --can-standalone no --host-probe '{"compiler":true,"project_build":false,"register_script":false,"runtime_registry":false,"reversibility_net":false}')
chk "S5 bare host->derive_adapter"     "$(g "$J" method)" derive_adapter
chk "S5 autonomy directive"            "$(g "$J" autonomy_directive)" "{'action': 'derive_integration_adapter', 'must_emit': ['kind', 'register_script', 'build_command', 'reversibility_net'], 'stamp_rule': 'build_fidelity + reversible are assigned by method kind, not by you'}"

# S6: uncertain + compiler (bare) -> optimistically try standalone (cheap, reversible; build failure reveals embedded-ness)
J=$($ST resolve --can-standalone uncertain --host-probe '{"compiler":true,"project_build":false,"register_script":false,"runtime_registry":false,"reversibility_net":false}')
chk "S6 uncertain+compiler->standalone(opt)" "$(g "$J" method)" standalone

# S6b: uncertain + NO compiler -> can't try standalone, no embedded -> derive_adapter
J=$($ST resolve --can-standalone uncertain --host-probe '{"compiler":false,"project_build":false,"register_script":false,"runtime_registry":false,"reversibility_net":false}')
chk "S6b uncertain+nocompiler->derive" "$(g "$J" method)" derive_adapter

# S7: reproducibility + cache hit
D1=$($ST resolve --can-standalone no --host-probe '{"compiler":true,"project_build":true,"register_script":true,"runtime_registry":false,"reversibility_net":true}' --cache cache.json)
D2=$($ST resolve --can-standalone no --host-probe '{"compiler":true,"project_build":true,"register_script":true,"runtime_registry":false,"reversibility_net":true}' --cache cache.json)
chk "S7 identical" "$(g "$D1" method)|$(g "$D1" cache_key)" "$(g "$D2" method)|$(g "$D2" cache_key)"
chk "S7 cache hit" "$(g "$D2" _cache)" hit

# S8: preference order — embedded_dispatch preferred over inplace when both available (non-mutating wins)
chk "S8 dispatch>inplace (preference)" "$(g "$D1" method)" embedded_dispatch

# S9: non-standalone + sol-execbench CLI present (no project build/adapter) -> sol_execbench_solution
J=$($ST resolve --can-standalone no --host-probe '{"compiler":true,"project_build":false,"register_script":false,"runtime_registry":false,"reversibility_net":false,"sol_execbench_cli":true}')
chk "S9 sol host->sol_execbench_solution" "$(g "$J" method)" sol_execbench_solution
chk "S9 fidelity production"              "$(g "$J" build_fidelity)" production

# S9b: SAME host but NO sol CLI -> unchanged derive_adapter (backward compat)
J=$($ST resolve --can-standalone no --host-probe '{"compiler":true,"project_build":false,"register_script":false,"runtime_registry":false,"reversibility_net":false}')
chk "S9b no-sol-cli->derive_adapter (compat)" "$(g "$J" method)" derive_adapter

rm -f cache.json traj.jsonl
echo "================  $pass passed, $fail failed  ================"
exit $fail
