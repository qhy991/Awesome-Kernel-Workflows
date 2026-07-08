// _substrate/embedded/sol_execbench_eval.js
//
// Shared "sol-execbench-solution" evaluation substrate. A workflow uses this to
// evaluate a candidate kernel for a FlashInfer-Bench / sol-execbench task: the
// candidate is packaged into a solution.json and handed to the sol-execbench CLI,
// which compiles it internally against the task harness, then per-workload
// speedup_factor is reduced to a geomean.
//
// Two roles, like _substrate/embedded/embedded_eval.js:
//   1. Canonical source (human-readable + single source of truth).
//   2. The region between the SOL_INLINE_BEGIN / END sentinels is INLINED verbatim
//      into each opted-in workflow .js by scripts/patch-sol-execbench-eval.js
//      (the workflow runtime has no ES module import; inline, don't import).
//
// What a workflow gets after inlining:
//   SOL_SOLUTION_CONTRACT      -- prompt fragment for proposal subagents in sol
//                                 mode (emit kernel + torch binding; NOT a main()).
//   __solExecbenchEvalPlan(ctx) -- ordered Bash strings: pack -> run -> parse.
//
// The standalone/embedded paths are unchanged: a workflow only calls this when
// INTEGRATION_DECISION.method === 'sol_execbench_solution'.

// >>> SOL_INLINE_BEGIN (do not edit inlined copies; edit this source + re-run scripts/patch-sol-execbench-eval.js) <<<
const SOL_SOLUTION_CONTRACT = [
  'SOL-EXECBENCH SOLUTION CONTRACT (this task is evaluated by the sol-execbench CLI):',
  '',
  'You are authoring a kernel that will be packaged into a solution.json and run by',
  'the sol-execbench harness, which compiles it internally. Therefore:',
  '',
  '1. Emit a COMPLETE kernel source plus a torch binding that exposes the task',
  '   entry point (run(...)). Do NOT write a standalone main()/CLI harness.',
  '2. Match the task reference signature exactly (same argument order/dtypes).',
  '3. Do NOT package, compile, or benchmark yourself — the workflow + substrate',
  '   handle pack -> sol-execbench -> parse. Return only the kernel + binding.',
].join('\n')

function __solQ(s) { return `"${String(s).replace(/"/g, '\\"')}"` }

function __solExecbenchEvalPlan(ctx) {
  const substrateDir = ctx.substrateDir            // abs path to _substrate/integration
  const kernelSource = ctx.kernelSource            // path to candidate kernel on disk
  const contractEnv = ctx.contractEnv              // path to session contract.env
  const solutionOut = ctx.solutionOut              // where to write solution.json
  const benchOut = ctx.benchOut                    // where sol-execbench writes bench.jsonl
  const solCli = ctx.solCli                        // e.g. /abs/sol-execbench/.venv/bin/sol-execbench
  const taskDir = ctx.taskDir                      // FlashInfer-Bench/<task> dir
  const benchConfig = ctx.benchConfig              // --config path
  const seedDir = ctx.seedDir                      // cd target for the run
  const cvd = ctx.cudaVisibleDevices || '0'
  const ld = ctx.ldLibraryPath ? `LD_LIBRARY_PATH=${__solQ(ctx.ldLibraryPath)}:$LD_LIBRARY_PATH ` : ''

  const pack = `python3 ${__solQ(substrateDir + '/pack_sol_candidate.py')} --kernel ${__solQ(kernelSource)} --contract ${__solQ(contractEnv)} --out ${__solQ(solutionOut)}`
  const run = `cd ${__solQ(seedDir)} && ${ld}CUDA_VISIBLE_DEVICES=${cvd} ${__solQ(solCli)} ${__solQ(taskDir)} --solution ${__solQ(solutionOut)} --config ${__solQ(benchConfig)} -o ${__solQ(benchOut)}`
  const parse = `python3 ${__solQ(substrateDir + '/parse_sol_bench.py')} ${__solQ(benchOut)}`

  return {
    pack,
    run,
    parse,
    order: ['pack', 'run', 'parse'],
    cleanupInvariant: 'solution.json + bench.jsonl are per-candidate scratch files in the run dir; overwrite freely. No project source is mutated (non-mutating method).',
  }
}
// <<< SOL_INLINE_END >>>

export { SOL_SOLUTION_CONTRACT, __solExecbenchEvalPlan }
