export const meta = {
  name: 'llamacpp-embedded-search',
  description: 'Multi-variant fan-out search for kernels embedded in llama.cpp ggml-cuda dispatch (each variant = new .cuh + register + project build + test + bench + unregister)',
  whenToUse: 'When the kernel under optimization is embedded in llama.cpp ggml-cuda (cannot be compiled standalone) AND the bottleneck is structural enough that you want to fan out N independent kernel designs rather than iteratively patch one. Each subagent authors a complete new .cuh; the workflow registers it into fattn.cu via scripts/llamacpp_register_variant.py, runs the project build/test/bench, then unregisters. Variants are evaluated SERIALLY (project build is the bottleneck) but proposals are drafted in parallel.',
  phases: [
    { title: 'Setup',     detail: 'Resolve paths, sanity-check register script, snapshot fattn.cu' },
    { title: 'Baseline',  detail: 'Build + test + bench the unmodified project to set the baseline number' },
    { title: 'Propose',   detail: 'N subagents in parallel each draft one complete variant .cuh file' },
    { title: 'Evaluate',  detail: 'For each variant: register -> build -> test -> bench -> unregister (serial)' },
    { title: 'Report',    detail: 'Rank by speedup, emit best variant code + history' },
  ],
}

// --- BEGIN model-tier (auto-inserted by scripts/patch-model-tier.js) ---
// Tier-based model routing: mechanical steps (run substrate scripts, parse
// JSON) use cheaper models; profile steps (run eval/ncu) use mid-tier;
// judgment steps (plan/implement/report) use the top tier. Tuneable via
// args.model_{mechanical,profile,judgment}.
const MODEL = {
  mechanical: (typeof args !== 'undefined' && args && args.model_mechanical) || 'haiku',
  profile: (typeof args !== 'undefined' && args && args.model_profile) || 'sonnet',
  judgment: (typeof args !== 'undefined' && args && args.model_judgment) || 'opus',
}
// __modelTierApplied
// --- END model-tier ---

const WORKFLOW_NAME = 'llamacpp-embedded-search'

// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

// --- BEGIN inlined arg_guard ---
function __unwrapArgs(rawArgs) {
  if (rawArgs == null) return {}
  if (typeof rawArgs === 'object' && !Array.isArray(rawArgs)) return rawArgs
  if (typeof rawArgs === 'string') {
    const trimmed = rawArgs.trim()
    if (trimmed === '') return {}
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed
        throw new Error('arg_guard: parsed JSON value is not a plain object')
      } catch (e) { throw new Error(`arg_guard: invalid JSON args: ${e.message}`) }
    }
    const out = {}
    const re = /(\w[\w.-]*)=("(?:\\"|[^"])*"|'(?:\\'|[^'])*'|\S+)/g
    let m
    while ((m = re.exec(trimmed)) !== null) {
      let v = m[2]
      if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
        v = v.slice(1, -1)
      }
      out[m[1]] = v
    }
    return out
  }
  return {}
}
// eslint-disable-next-line no-global-assign
args = __unwrapArgs(typeof args === 'undefined' ? undefined : args)
// --- END inlined arg_guard ---

// --- BEGIN inlined grounding contract ---
const GROUNDING_INSTRUCTION = [
  'GROUNDING CONTRACT (mandatory):',
  '',
  '1. Before reporting any numeric or categorical result, you MUST have executed',
  '   a real Bash command that produced the value. Citing a value from imagination,',
  '   prior knowledge, or analogy is a contract violation.',
  '',
  '2. If the workflow tells you to run a tool or script that does NOT exist on disk',
  '   or on PATH, do NOT substitute or simulate. Return:',
  '       { "grounded": false, "missing": "<exact tool or script that was absent>" }',
  '   and stop. The schema accepts this shape.',
  '',
  '3. If a Bash command fails (non-zero exit), do NOT invent the value it would',
  '   have produced. Return:',
  '       { "grounded": false, "error": "<stderr tail or short summary>" }',
  '   and stop.',
  '',
  '4. Numeric fields in the schema are REJECTED downstream when grounded === false.',
  '   Do not fill them with placeholders, zeros, or "typical" values.',
  '',
  '5. The string "fabricated", "simulated", "estimated", or "placeholder" must',
  '   never appear in a value you report as measured.',
].join('\n')

function withGroundingFields(schema) {
  schema.properties = schema.properties || {}
  if (!schema.properties.grounded) schema.properties.grounded = { type: 'boolean' }
  if (!schema.properties.missing)  schema.properties.missing  = { type: 'string' }
  if (!schema.properties.error)    schema.properties.error    = { type: 'string' }
  return schema
}

function classifyResult(r) {
  if (r == null) return { state: 'failed', error: 'agent returned null' }
  if (r.grounded === false) return { state: 'not_grounded', missing: r.missing || null, error: r.error || null }
  return { state: 'grounded', value: r }
}
// --- END inlined grounding contract ---

// =============================================================================
// Llamacpp Embedded Multi-Variant Search
// =============================================================================
//
// Required args:
//   ggml_root          Absolute path to llama.cpp/ggml (parent of src/ggml-cuda)
//   register_script    Absolute path to scripts/llamacpp_register_variant.py
//   build_command      Project build command (run verbatim)
//   test_command       Project correctness command (KERSOR_VARIANT will be prefixed)
//   benchmark_command  Project benchmark command (KERSOR_VARIANT will be prefixed)
//   reference_cuh      Path to the current best variant .cuh (e.g. fattn-rdna-apu.cuh).
//                      Subagents read this for inspiration and as the structural anchor.
//
// Optional args:
//   n_variants            Number of parallel proposals per round (default 3)
//   max_rounds            Number of search rounds (default 1)
//   target_speedup        Stop early if reached (default 1.05)
//   correctness_regex     Regex that must match test stdout for PASS (default: exit code only)
//   latency_regex         Regex with one capture group for latency number (default first float + us/ms)
//   latency_unit          us|ms (default us)
//   tolerance             NMSE tolerance (informational, default 5e-4)
//   op_description        Task context, prepended to proposer prompt
//   handoff_context       KerSor cross-round handoff text
//   workload_axes         Comma-separated workload labels (informational)
//   dkq, dv               Head dimensions for template instantiation (default 256, 256)
//   variants_dir          Where to write each variant .cuh before register (default: $exp_dir/variants/)
//   exp_dir               Run scratch dir (default: process cwd)
//
// =============================================================================

const GGML_ROOT      = args.ggml_root || ''
const REG_SCRIPT     = args.register_script || ''
const BUILD_CMD      = args.build_command || ''
const TEST_CMD       = args.test_command || ''
const BENCH_CMD      = args.benchmark_command || ''
const REFERENCE_CUH  = args.reference_cuh || ''
const N_VARIANTS     = Math.max(1, parseInt(args.n_variants || 3, 10))
const MAX_ROUNDS     = Math.max(1, parseInt(args.max_rounds || 1, 10))
const TARGET_SPEEDUP = parseFloat(args.target_speedup || 1.05)
const CORRECT_REGEX  = args.correctness_regex || ''
const LATENCY_REGEX  = args.latency_regex || '([0-9]+\\.?[0-9]*)\\s*(?:us|ms)'
const LATENCY_UNIT   = (args.latency_unit || 'us').toLowerCase()
const TOLERANCE      = args.tolerance != null ? Number(args.tolerance) : 5e-4
const OP_DESC        = args.op_description || ''
const HANDOFF        = args.handoff_context || ''
const WORKLOAD_AXES  = args.workload_axes || ''
const DKQ            = parseInt(args.dkq || 256, 10)
const DV             = parseInt(args.dv  || 256, 10)
const EXP_DIR        = args.exp_dir || '.'
const VARIANTS_DIR   = args.variants_dir || `${EXP_DIR}/variants`
// Optional: if provided, the register/unregister calls pass --cmake-build-dir
// so they reglob automatically (CMake file(GLOB) without CONFIGURE_DEPENDS
// would otherwise leave the new .cu sources invisible to incremental builds).
const CMAKE_BUILD_DIR = args.cmake_build_dir || ''

const MISSING = []
if (!GGML_ROOT)     MISSING.push('ggml_root')
if (!REG_SCRIPT)    MISSING.push('register_script')
if (!BUILD_CMD)     MISSING.push('build_command')
if (!TEST_CMD)      MISSING.push('test_command')
if (!BENCH_CMD)     MISSING.push('benchmark_command')
if (!REFERENCE_CUH) MISSING.push('reference_cuh')
if (MISSING.length) {
  throw new Error(`llamacpp-embedded-search: missing required arg(s): ${MISSING.join(', ')}`)
}

// =============================================================================
// Schemas
// =============================================================================

const PROPOSAL_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    written: { type: 'boolean', description: 'true iff the .cuh file was successfully written' },
    variant_name: { type: 'string', description: 'lowercase [a-z0-9_]{1,32} - this is what KERSOR_VARIANT must be set to' },
    cuh_path: { type: 'string', description: 'absolute path to the new variant .cuh that was written' },
    title: { type: 'string', description: 'short label for this design' },
    rationale: { type: 'string', description: 'why this design should be faster, citing the reference kernel' },
    risk: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
  required: ['written', 'variant_name', 'cuh_path', 'title', 'rationale'],
})

const BUILD_SCHEMA = withGroundingFields({
  type: 'object',
  properties: { ok: { type: 'boolean' }, stderr_tail: { type: 'string' }, duration_sec: { type: 'number' } },
  required: ['ok'],
})

const TEST_SCHEMA = withGroundingFields({
  type: 'object',
  properties: { correct: { type: 'boolean' }, detail: { type: 'string' } },
  required: ['correct'],
})

const BENCH_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    latency_per_workload: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          workload: { type: 'string' },
          latency: { type: 'number' },
          unit: { type: 'string' },
        },
        required: ['workload', 'latency'],
      },
    },
    aggregate_latency: { type: 'number' },
    aggregate_strategy: { type: 'string' },
  },
  required: ['aggregate_latency'],
})

// =============================================================================
// Helpers (each helper runs ONE Bash command via a subagent so we get grounding)
// =============================================================================

async function runBuild(label, env) {
  const cmd = env ? `${env} ${BUILD_CMD}` : BUILD_CMD
  const r = await agent(
    [
      `Run the project build command and report whether it succeeded.`,
      `Command: ${cmd}`,
      `If exit is non-zero, include the last ~30 lines of stderr in stderr_tail.`,
      `Measure wall-clock duration in seconds.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    { phase: 'Evaluate', label: `build:${label}`, schema: BUILD_SCHEMA }
  )
  return classifyResult(r)
}

async function runTest(label, env) {
  const cmd = env ? `${env} ${TEST_CMD}` : TEST_CMD
  const correctnessHint = CORRECT_REGEX
    ? `CORRECT iff /${CORRECT_REGEX}/ matches stdout AND exit is 0.`
    : `CORRECT iff exit is 0.`
  const r = await agent(
    [
      `Run the project correctness command and report PASS/FAIL.`,
      `Command: ${cmd}`,
      correctnessHint,
      `Tolerance reference (NMSE): ${TOLERANCE}.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    { phase: 'Evaluate', label: `test:${label}`, schema: TEST_SCHEMA }
  )
  return classifyResult(r)
}

async function runBench(label, env) {
  const cmd = env ? `${env} ${BENCH_CMD}` : BENCH_CMD
  const r = await agent(
    [
      `Run the project benchmark command and extract per-workload latency.`,
      `Command: ${cmd}`,
      `Latency regex: /${LATENCY_REGEX}/ - first capture group is the number, unit is ${LATENCY_UNIT}.`,
      `If output has multiple workload rows, emit one entry per workload in latency_per_workload.`,
      `Set aggregate_latency = GEOMEAN of the per-workload latencies (single regressions cannot dominate).`,
      `Set aggregate_strategy to a short description.`,
      WORKLOAD_AXES ? `Workload axes hint: ${WORKLOAD_AXES}` : '',
      GROUNDING_INSTRUCTION,
    ].filter(Boolean).join('\n\n'),
    { phase: 'Evaluate', label: `bench:${label}`, schema: BENCH_SCHEMA }
  )
  return classifyResult(r)
}

async function registerVariant(variantName, cuhPath) {
  const cmakeFlag = CMAKE_BUILD_DIR ? ` --cmake-build-dir "${CMAKE_BUILD_DIR}"` : ''
  const r = await agent(
    [
      `Run the register script to wire this variant into fattn.cu:`,
      `  python "${REG_SCRIPT}" register --variant ${variantName} --cuh-src "${cuhPath}" --ggml-root "${GGML_ROOT}" --dkq ${DKQ} --dv ${DV}${cmakeFlag}`,
      `Then run "python ${REG_SCRIPT} list --ggml-root ${GGML_ROOT}" and confirm ${variantName} appears in the list.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    {
      phase: 'Evaluate',
      label: `register:${variantName}`,
      schema: withGroundingFields({
        type: 'object',
        properties: { ok: { type: 'boolean' }, stderr_tail: { type: 'string' } },
        required: ['ok'],
      }),
    }
  )
  return classifyResult(r)
}

async function unregisterVariant(variantName) {
  const cmakeFlag = CMAKE_BUILD_DIR ? ` --cmake-build-dir "${CMAKE_BUILD_DIR}"` : ''
  const r = await agent(
    [
      `Run the register script to remove this variant from fattn.cu (must be byte-exact reversible):`,
      `  python "${REG_SCRIPT}" unregister --variant ${variantName} --ggml-root "${GGML_ROOT}"${cmakeFlag}`,
      `Confirm by running "python ${REG_SCRIPT} list --ggml-root ${GGML_ROOT}" and reporting that ${variantName} is no longer listed.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    {
      phase: 'Evaluate',
      label: `unregister:${variantName}`,
      schema: withGroundingFields({
        type: 'object',
        properties: { ok: { type: 'boolean' } },
        required: ['ok'],
      }),
    }
  )
  return classifyResult(r)
}

// =============================================================================
// Phase: Setup + Baseline
// =============================================================================

phase('Setup')
log(`ggml_root         = ${GGML_ROOT}`)
log(`register_script   = ${REG_SCRIPT}`)
log(`reference_cuh     = ${REFERENCE_CUH}`)
log(`variants_dir      = ${VARIANTS_DIR}`)
log(`n_variants/round  = ${N_VARIANTS}`)
log(`max_rounds        = ${MAX_ROUNDS}`)

await agent(
  [
    `Ensure the variants directory exists: mkdir -p "${VARIANTS_DIR}".`,
    `Also confirm the register script is executable by running:`,
    `  python "${REG_SCRIPT}" list --ggml-root "${GGML_ROOT}"`,
    `Report ok=true iff both succeeded.`,
    GROUNDING_INSTRUCTION,
    `# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\nAppend exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\nThen append:\n{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"variants_dir_and_register_sanity","note":"<variants dir created? register script listable? one line>"}`,
  ].join('\n\n'),
  {
    phase: 'Setup',
    label: 'sanity-check',
    schema: withGroundingFields({
      type: 'object',
      properties: { ok: { type: 'boolean' } },
      required: ['ok'],
    }),
  }
)

phase('Baseline')
const baseBuild = await runBuild('baseline', '')
if (baseBuild.state !== 'grounded' || !baseBuild.value.ok) {
  return { ok: false, grounded: baseBuild.state === 'grounded',
    error: `baseline build failed: ${JSON.stringify(baseBuild)}`,
    speedup: null, best_kernel_code: null, history: [] }
}
const baseTest = await runTest('baseline', '')
if (baseTest.state !== 'grounded' || !baseTest.value.correct) {
  return { ok: false, grounded: baseTest.state === 'grounded',
    error: `baseline correctness failed: ${JSON.stringify(baseTest)}`,
    speedup: null, best_kernel_code: null, history: [] }
}
const baseBench = await runBench('baseline', '')
if (baseBench.state !== 'grounded' || !baseBench.value.aggregate_latency) {
  return { ok: false, grounded: baseBench.state === 'grounded',
    error: `baseline bench could not parse latency: ${JSON.stringify(baseBench)}`,
    speedup: null, best_kernel_code: null, history: [] }
}
const BASELINE_LATENCY = baseBench.value.aggregate_latency
log(`Baseline aggregate latency: ${BASELINE_LATENCY} ${LATENCY_UNIT}`)

// =============================================================================
// Search rounds
// =============================================================================

const history = []
let bestSpeedup = 1.0
let bestLatency = BASELINE_LATENCY
let bestVariantName = '(baseline)'
let bestCuhPath = REFERENCE_CUH

for (let round = 1; round <= MAX_ROUNDS; ++round) {
  if (bestSpeedup >= TARGET_SPEEDUP) {
    log(`Target ${TARGET_SPEEDUP}x reached. Stopping after round ${round - 1}.`)
    break
  }

  phase('Propose')
  log(`\n=== Round ${round}/${MAX_ROUNDS} - proposing ${N_VARIANTS} variants in parallel ===`)

  const priorTitles = history
    .map(h => `  - ${h.variant_name} (${h.title}) -> ${h.verdict} (${h.speedup?.toFixed?.(3) ?? 'n/a'}x)`)
    .join('\n') || '  (none)'

  const proposals = await parallel(
    Array.from({ length: N_VARIANTS }, (_, i) => () => {
      const slot = `r${round}v${i + 1}`
      return agent(
        [
          OP_DESC ? `Task context: ${OP_DESC}` : '',
          HANDOFF ? `Handoff from prior rounds:\n${HANDOFF}` : '',
          ``,
          `You are designing ONE complete new fattn variant kernel for llama.cpp ggml-cuda.`,
          ``,
          `Reference (current best) kernel: ${REFERENCE_CUH}`,
          `Read this file in full first. Your variant MUST use the SAME canonical symbol`,
          `names as the reference - the registration script will suffix every "rdna_apu"`,
          `identifier with the variant name automatically, so you must NOT pre-suffix them:`,
          `  - the kernel function:           flash_attn_rdna_apu`,
          `  - launch wrappers:               launch_fattn_rdna_apu_switch_ncols1 / _switch_ncols2`,
          `  - case entry template:           ggml_cuda_flash_attn_ext_rdna_apu_case<DKQ,DV>`,
          `  - declaration macro:             DECL_FATTN_RDNA_APU_CASE(DKQ, DV)`,
          `  - any helpers you add should also use the prefix  fattn_rdna_apu_*`,
          `Do NOT name any symbol with your variant name suffix - the register script does that.`,
          `The .cuh must keep the fattn_kernel_t signature for the kernel function.`,
          `Constraints:`,
          `  - Pick a variant_name matching [a-z0-9_]{1,32}; suggested: "${slot}"`,
          `  - Write the complete .cuh to: ${VARIANTS_DIR}/<variant_name>.cuh`,
          `  - Do NOT register the variant yourself - the workflow does that.`,
          `  - Do NOT modify fattn.cu or any other file - the workflow does dispatch wiring.`,
          `  - Make ONE design distinctly different from prior attempts:`,
          ``,
          `Prior attempts this session:\n${priorTitles}`,
          ``,
          `Workload context: ${WORKLOAD_AXES || '(none)'}`,
          `Tolerance reference (NMSE): ${TOLERANCE}`,
          ``,
          `Return the absolute cuh_path you wrote, the variant_name, a short title,`,
          `and a rationale citing the specific design choice (tile shape, register`,
          `layout, sync pattern, ...).`,
          GROUNDING_INSTRUCTION,
          `# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\nAppend exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\nThen append (this is variant ${slot} of round ${round}):\n{"workflow":"${WORKFLOW_NAME}","phase":"Propose","ts":"<ts>","status":"done","candidate_id":"${slot}","technique":"<the distinct design choice for this variant, e.g. tile_shape_or_register_layout>","speedup":null,"note":"<one-line rationale for why this variant should be faster>"}`,
        ].filter(Boolean).join('\n\n'),
        { phase: 'Propose', label: `propose:${slot}`, schema: PROPOSAL_SCHEMA }
      )
    })
  )

  const drafted = proposals
    .map(classifyResult)
    .filter(p => p.state === 'grounded' && p.value.written)
    .map(p => p.value)

  log(`Drafted ${drafted.length}/${N_VARIANTS} variants for round ${round}`)

  // ----- Serial evaluate -----
  phase('Evaluate')
  for (const v of drafted) {
    log(`\n--- Evaluating ${v.variant_name} ("${v.title}") ---`)
    const reg = await registerVariant(v.variant_name, v.cuh_path)
    if (reg.state !== 'grounded' || !reg.value.ok) {
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'register_failed', speedup: null,
        detail: JSON.stringify(reg).slice(0, 400) })
      continue
    }

    const b = await runBuild(v.variant_name, '')
    if (b.state !== 'grounded' || !b.value.ok) {
      await unregisterVariant(v.variant_name)
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'build_failed', speedup: null,
        detail: b.state === 'grounded' ? (b.value.stderr_tail || '').slice(-400) : JSON.stringify(b) })
      continue
    }

    const t = await runTest(v.variant_name, `KERSOR_VARIANT=${v.variant_name}`)
    if (t.state !== 'grounded' || !t.value.correct) {
      await unregisterVariant(v.variant_name)
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'correctness_failed', speedup: null,
        detail: t.state === 'grounded' ? t.value.detail : JSON.stringify(t) })
      continue
    }

    const m = await runBench(v.variant_name, `KERSOR_VARIANT=${v.variant_name}`)
    if (m.state !== 'grounded' || !m.value.aggregate_latency) {
      await unregisterVariant(v.variant_name)
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'bench_unparseable', speedup: null,
        detail: JSON.stringify(m) })
      continue
    }

    const latency = m.value.aggregate_latency
    const speedup = BASELINE_LATENCY / latency
    log(`  ${v.variant_name}: ${latency} ${LATENCY_UNIT} -> ${speedup.toFixed(3)}x`)

    if (speedup > bestSpeedup) {
      bestSpeedup = speedup
      bestLatency = latency
      bestVariantName = v.variant_name
      bestCuhPath = v.cuh_path
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'kept_best', speedup,
        detail: `latency ${latency} ${LATENCY_UNIT}`, cuh_path: v.cuh_path })
      // Leave registered so the next round can read the actual current-best from disk.
      // But unregister it before we exit the loop overall - we do that in Report.
    } else {
      await unregisterVariant(v.variant_name)
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'unkept', speedup,
        detail: `latency ${latency} ${LATENCY_UNIT}` })
    }
  }
}

// =============================================================================
// Report
// =============================================================================

phase('Report')

// Read the winning kernel back (or null if no variant beat baseline).
let bestCode = null
if (bestVariantName !== '(baseline)') {
  const r = await agent(
    `Read the file at ${bestCuhPath} and return its full contents in best_kernel_code.\n\n# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\nAppend exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\nThen append (the search winner is variant ${bestVariantName} at ${bestSpeedup}x over baseline):\n{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","candidate_id":"${bestVariantName}","speedup":${bestSpeedup},"technique":"best_variant_selected","note":"<one-line summary of the winning design and its latency>"}`,
    {
      phase: 'Report',
      label: 'final-read',
      schema: withGroundingFields({
        type: 'object',
        properties: { best_kernel_code: { type: 'string' } },
        required: ['best_kernel_code'],
      }),
    }
  )
  bestCode = classifyResult(r).value?.best_kernel_code || null
  // Unregister the winner too: leave the project in its original byte-exact state.
  // The caller decides whether to re-register the kept variant (e.g. via best-kernel/).
  await unregisterVariant(bestVariantName)
}

return {
  ok: true,
  grounded: true,
  speedup: bestSpeedup,
  overall_speedup: bestSpeedup,
  rounds_completed: history.length,
  baseline_latency: BASELINE_LATENCY,
  best_latency: bestLatency,
  best_variant_name: bestVariantName,
  best_cuh_path: bestCuhPath,
  best_kernel_code: bestCode,
  latency_unit: LATENCY_UNIT,
  variants_evaluated: history.length,
  history,
}
