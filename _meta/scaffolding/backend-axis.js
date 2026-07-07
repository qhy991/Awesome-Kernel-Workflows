// backend-axis.js — CANONICAL default-scaffolding snippet for the backend-axis
// (driver-dispatch) plumbing shared by every `agent()`-based workflow that
// routes on a backend / language axis and loads a backend driver.
//
// This is NOT a runnable workflow. It is the single source of truth for the
// backend-axis helper cluster that 15+ workflows inline verbatim
// (workflow .js files run in the Claude Code Workflow sandbox and cannot
// `import`, so the helpers are copied into each file — the same convention
// used for the inlined arg_guard and agent-retry scaffolding).
//
// WHY: `normalizeSuitabilityValue` + `resolveBackendAxis` + `driverPath` +
// `driverSh` were copy-pasted across 15 workflows and had drifted
// (e.g. `driverSh`'s prefix variable and message wording diverged in 4 of 16).
// The #42 bug (`anti_cheat.py --kernel` vs `--source`) was the same class of
// failure — substrate-script flags hand-written per workflow. Centralizing the
// cluster here + a `patch-backend-axis.js --refresh` codemod lets a one-line
// fix to the canonical block propagate to every workflow, and a guard test
// surfaces drift before it ships.
//
// The cluster has TWO sub-clusters (they live at different points in each
// workflow — the resolve block runs early, right after arg_guard; the driver
// helpers after BACKEND_DIR is known). Both are tagged with their own
// BEGIN/END sentinels so `patch-backend-axis.js --refresh` can re-sync each
// independently.
//
// USAGE (a workflow inlines both blocks; `patch-backend-axis.js` does this):
//
//   // --- BEGIN inlined backend-axis (resolve) scaffolding (from _meta/scaffolding/backend-axis.js) ---
//   <paste normalizeSuitabilityValue + resolveBackendAxis + RESOLVED_BACKEND + USE_DRIVER>
//   // --- END inlined backend-axis (resolve) scaffolding ---
//   ...
//   // --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---
//   <paste driverPath + driverSh>
//   // --- END inlined backend-axis (driver) scaffolding ---
//
// SUB-CLUSTER A (resolve) — byte-identical across all 15 workflows that have it.

function normalizeSuitabilityValue(value) {
  const raw = String(value || '').trim().toLowerCase().replace(/_/g, '-')
  const aliases = {
    'c++': 'cpp',
    cxx: 'cpp',
    cplusplus: 'cpp',
    cute: 'cute-dsl',
    hip: 'rocm',
    'intel-xpu': 'xpu',
    optimize: 'kernel-optimization',
    optimization: 'kernel-optimization',
    generate: 'kernel-generation',
    generation: 'kernel-generation',
    explain: 'performance-explanation',
    explanation: 'performance-explanation',
  }
  return aliases[raw] || raw
}


function resolveBackendAxis() {
  const b = args.backend ? normalizeSuitabilityValue(args.backend) : null
  const l = args.language ? normalizeSuitabilityValue(args.language) : null
  if (b && l && b !== l) {
    throw new Error(`Conflicting args: backend="${args.backend}" vs language="${args.language}". Pass only one.`)
  }
  if (args.backend && !args.backend_dir) {
    throw new Error(`args.backend="${args.backend}" requires args.backend_dir; driver dispatch has no implicit-resolve path.`)
  }
  return b || l || null
}
const RESOLVED_BACKEND = resolveBackendAxis()
const USE_DRIVER = !!args.backend_dir

// SUB-CLUSTER B (driver) — `driverPath` is byte-identical across all 15; `driverSh`
// is the CANONICAL form (shared by 11 of 15). 4 workflows have drifted `driverSh`
// (prefix var / wording) and are NOT inlined from this SSOT yet —
// `patch-backend-axis.js` skips them and the guard test flags them for review.

function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
