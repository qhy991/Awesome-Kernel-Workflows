// substrate-invocation.js — CANONICAL command builders for substrate scripts.
//
// This is NOT a runnable workflow. It is the single source of truth for the
// substrate-script CLI flag schema, so a flag change (e.g. #42 anti_cheat
// --kernel -> --source) is one fix here, not 17 hand-edits across workflows.
// Workflows that call a substrate script inline the relevant builder and invoke
// it from a prompt template literal; the builder returns the
// "Run exactly: `...`." string.
//
// WHY: anti_cheat.py's argparse requires --source / --metrics, but 17 workflows
// hand-wrote --kernel / --result (argparse rejected the call; the anti_cheat
// check silently failed for every candidate). #42 fixed KSearch only; this SSOT
// + scripts/patch-anti-cheat-flags.js fix the other 17 and prevent recurrence
// (the guard test asserts no `anti_cheat.py --kernel` remains).
//
// CONSTRAINT: the builders reference PY (python prefix) and SUBSTRATE (substrate
// dir) — top-level consts every substrate-calling workflow defines
// (`const PY = args.substrate_command_prefix || ''`,
//  `const SUBSTRATE = args.substrate_dir || '_substrate'`).
//
// USAGE (inline the builder, then call from a prompt):
//
//   // --- BEGIN inlined substrate-invocation scaffolding (from _meta/scaffolding/substrate-invocation.js) ---
//   function substrateAntiCheat({ source, metrics }) { ... }
//   // --- END inlined substrate-invocation scaffolding ---
//
//   const cmd = substrateAntiCheat({ source: kPath, metrics: `${EXP_DIR}/${candidateId}.result.json` })
//   `...${cmd}...`   // -> '...Run exactly: `<PY>.../anti_cheat.py --source ... --metrics ....'

/**
 * Build the anti_cheat.py invocation. anti_cheat.py's argparse REQUIRES
 * --source <kernel file> and --metrics <result.json>. (#42: hand-written
 * --kernel / --result was rejected by argparse; the check silently failed.)
 */
function substrateAntiCheat({ source, metrics }) {
  return `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --source ${source} --metrics ${metrics}\`.`
}
