// _substrate/embedded/sol_execbench_eval.js
//
// Shared "sol-execbench-solution" evaluation substrate. A workflow uses this to
// evaluate a candidate kernel for a FlashInfer-Bench / sol-execbench task: the
// candidate is packaged into a solution.json and handed to the sol-execbench CLI,
// which compiles it internally against the task harness, then reduces the
// per-workload measurements according to the session contract.
//
// Two roles, like _substrate/embedded/embedded_eval.js:
//   1. Canonical source (human-readable + single source of truth).
//   2. The region between the SOL_INLINE_BEGIN / END sentinels is INLINED verbatim
//      into each opted-in workflow .js by scripts/patch-sol-execbench-eval.js
//      (the workflow runtime has no ES module import; inline, don't import).
//
// What a workflow gets after inlining:
//   SOL_SOLUTION_CONTRACT      -- prompt fragment for proposal subagents in sol
//                                 mode (emit a runnable source; NOT a main()).
//   __solExecbenchEvalPlan(ctx) -- ordered Bash strings: pack -> run -> parse.
//   __solExecbenchEvaluate(ctx) -- Host-owned deterministic transaction when the
//                                 runtime injects evaluate(); null otherwise.
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
  '1. Emit a COMPLETE candidate with the task entry point run(...). CUDA C++',
  '   requires a torch PYBIND11_MODULE binding; Python/Triton requires a',
  '   module-level def run(...). Do NOT write a standalone main()/CLI harness.',
  '2. Match the task reference signature exactly (same argument order/dtypes).',
  '3. Do NOT package, compile, or benchmark yourself — the workflow + substrate',
  '   handle pack -> sol-execbench -> parse. Return only the runnable source.',
].join('\n')

function __solQ(s) { return `"${String(s).replace(/"/g, '\\"')}"` }

function __solExecbenchEvalPlan(ctx) {
  const substrateDir = ctx.substrateDir            // abs path to _substrate/integration
  const kernelSource = ctx.kernelSource            // path to candidate kernel on disk
  const contractEnv = ctx.contractEnv              // path to session contract.env
  const solutionOut = ctx.solutionOut              // where to write solution.json
  const benchOut = ctx.benchOut                    // where sol-execbench writes bench.jsonl
  const normalizedOut = ctx.normalizedOut || ''    // optional canonical measurement JSON
  const solCli = ctx.solCli                        // e.g. /abs/sol-execbench/.venv/bin/sol-execbench
  const taskDir = ctx.taskDir                      // FlashInfer-Bench/<task> dir
  const benchConfig = ctx.benchConfig              // --config path
  const seedDir = ctx.seedDir                      // cd target for the run
  const cvd = ctx.cudaVisibleDevices || '0'
  const ld = ctx.ldLibraryPath ? `LD_LIBRARY_PATH=${__solQ(ctx.ldLibraryPath)}:$LD_LIBRARY_PATH ` : ''
  const env = ctx.envPrefix ? `${String(ctx.envPrefix).trim()} ` : ''
  const definition = ctx.definitionPath ? ` --definition ${__solQ(ctx.definitionPath)}` : ''

  const pack = `rm -f -- ${__solQ(solutionOut)} && python3 ${__solQ(substrateDir + '/pack_sol_candidate.py')} --kernel ${__solQ(kernelSource)} --contract ${__solQ(contractEnv)} --out ${__solQ(solutionOut)}`
  const clearRunOutputs = [benchOut, normalizedOut].filter(Boolean).map(__solQ).join(' ')
  const run = `rm -f -- ${clearRunOutputs} && test -s ${__solQ(solutionOut)} && cd ${__solQ(seedDir)} && ${env}${ld}CUDA_VISIBLE_DEVICES=${cvd} ${__solQ(solCli)} ${__solQ(taskDir)}${definition} --solution ${__solQ(solutionOut)} --config ${__solQ(benchConfig)} -o ${__solQ(benchOut)}`
  const parse = `test -s ${__solQ(benchOut)} && python3 ${__solQ(substrateDir + '/parse_sol_bench.py')} ${__solQ(benchOut)} --contract ${__solQ(contractEnv)}${normalizedOut ? ` --out ${__solQ(normalizedOut)}` : ''}`

  return {
    pack,
    run,
    parse,
    order: ['pack', 'run', 'parse'],
    cleanupInvariant: 'solution.json + bench.jsonl are per-candidate scratch files in the run dir; each stage clears its own stale outputs and requires the preceding artifact. No project source is mutated (non-mutating method).',
  }
}

async function __solExecbenchEvaluate(ctx) {
  // Claude's legacy Workflow host does not yet expose this optional primitive.
  // Keep the prompt-driven path as a compatibility edge, while KerSor's Host
  // owns exact source materialization and PACK/RUN/PARSE without an LLM turn.
  if (typeof evaluate !== 'function') return null
  return evaluate({
    protocol: 'sol-execbench-v1',
    label: ctx.label || 'sol-eval',
    phase: ctx.phase || 'Evaluate',
    candidatePath: ctx.kernelSource,
    candidateSource: ctx.candidateSource,
    substrateDir: ctx.substrateDir,
    contractEnv: ctx.contractEnv,
    solutionOut: ctx.solutionOut,
    benchOut: ctx.benchOut,
    normalizedOut: ctx.normalizedOut || `${ctx.benchOut}.result.json`,
    solCli: ctx.solCli,
    taskDir: ctx.taskDir,
    benchConfig: ctx.benchConfig,
    seedDir: ctx.seedDir,
    cudaVisibleDevices: ctx.cudaVisibleDevices || '0',
    ldLibraryPath: ctx.ldLibraryPath || '',
    envPrefix: ctx.envPrefix || '',
    definitionPath: ctx.definitionPath || '',
    timeoutSeconds: ctx.timeoutSeconds || 0,
  })
}
// <<< SOL_INLINE_END >>>

export { SOL_SOLUTION_CONTRACT, __solExecbenchEvalPlan, __solExecbenchEvaluate }
