// WarpSpeed orchestrator dry-run tests.
// Runs warpspeed-kernel-search.js under a vm sandbox (same textual async-wrap
// as _meta/tools/lib/run-workflow.js) with fixture agent returns keyed by
// label, and asserts the ROUTING / SIGNIFICANCE / ALLOCATION / REVIEW-LOOP /
// POSTMORTEM-REWIND decisions that live in the script's pure JS.
//
//   node WarpSpeed/tests/workflow/dryrun.test.mjs
import assert from 'node:assert'
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { fileURLToPath } from 'node:url'

const WS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const SOURCE = fs.readFileSync(path.join(WS, 'warpspeed-kernel-search.js'), 'utf8')

async function run(args, fixtures) {
  const calls = []
  const unfixtured = []
  let currentPhase = ''
  function agentStub(prompt, opts) {
    const label = opts && opts.label
    calls.push({ label, phase: (opts && opts.phase) || currentPhase, prompt })
    if (fixtures && Object.prototype.hasOwnProperty.call(fixtures, label)) return fixtures[label]
    unfixtured.push(label)
    return {} // grounded-by-default empty object; flow must tolerate it on non-critical paths
  }
  const sandbox = {
    args,
    agent: agentStub,
    phase: t => { currentPhase = t },
    parallel: async thunks => { const o = []; for (const t of thunks) o.push(await t()); return o },
    pipeline: async (items, fn) => { const o = []; for (const it of items) o.push(await fn(it)); return o },
    log: () => {},
    budget: () => undefined,
    console, JSON, Math, Promise,
  }
  vm.createContext(sandbox)
  const wrapped = '(async function(){\n' + SOURCE.replace(/^export\s+/, '') + '\n})()'
  const result = await vm.runInContext(wrapped, sandbox, { filename: 'warpspeed-kernel-search.js' })
  return { result, calls, unfixtured, labels: new Set(calls.map(c => c.label)) }
}

function promptOf(calls, label) {
  const c = calls.find(c => c.label === label)
  assert.ok(c, `expected agent call with label ${label}`)
  return c.prompt
}

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const CFG = (over = {}) => Object.assign({
  commands: {
    wsdb: 'WSDB', render_config: 'RC', gpu_run: 'GR', correctness: 'CORR',
    screen: 'SCR', confirm: 'CONF', ncu_profile: 'NCUP', calibrate: 'CAL',
    sanitizer_racecheck: 'SRACE', sanitizer_memcheck: 'SMEM',
  },
  paths: {
    warpspeed_dir: '/ws', project_dir: '/proj', state_dir: '/proj/.warpspeed',
    harness_dir: '/proj/harness', db: '/proj/.warpspeed/search.sqlite',
    worktrees: '/proj/.warpspeed/worktrees', review: '/proj/.warpspeed/review',
    builds: '/proj/.warpspeed/builds', ncu_cache: '/proj/.warpspeed/ncu_cache',
    logs: '/proj/.warpspeed/logs', results: '/proj/.warpspeed/results',
    bitlessons: '/proj/.warpspeed/bitlessons.jsonl',
    hardware_facts: '/ws/config/hardware-facts-sm90.md', wiki_dir: '/ws/wiki',
  },
  PARALLEL_AGENTS: 2, MAX_REVIEW_ITERS: 3, K_BLOCKED: 4,
  MIN_GAIN_PCT: 1.0, CONFIRM_MARGIN_PCT: 5.0, COMPACT_EVERY: 10,
  ALLOCATION: { exploit: 0.6, explore: 0.3, wildcard: 0.1 },
  REVIEWER_CMD: 'codex exec', arch: 'sm90',
  wiki_index: {
    tma: ['/ws/wiki/tma-bulk-async-copy.md'],
    persistent_kernel: ['/ws/wiki/persistent-kernel-scheduling.md'],
  },
}, over)

const INIT = (cfgOver) => ({
  grounded: true, config: CFG(cfgOver),
  preflight: { gpus: 8, git_ok: true, wsdb_ok: true, reviewer_ok: true, ncu_ok: true, sanitizer_ok: true },
  harness_ready: true, harness_scaffolded: false,
})
const CAL = { grounded: true, skipped: false, cross_device_sigma_pct: 0.3, per_device_means_us: {} }
const SEED = { grounded: true, commit: 'basecommit00', latency_us: 1000, ncu_fingerprint: 'fp0', skipped: false }

const SNAP = (over = {}) => ({
  grounded: true,
  snapshot: Object.assign({
    round: 1, baseline_commit: 'basecommit00', frontier_commit: 'basecommit00',
    frontier: [{ commit: 'basecommit00', parent: null, latency_us: 1000, assumptions: [], blocked_count: 0, strategy_set_hash: 'h0', headroom_pct: 38, is_frontier: true }],
    leaderboard: [{ commit: 'basecommit00', latency_us: 1000, assumptions: [], round_created: 0 }],
    recent_lessons: [], lessons_digest: '',
    budget: { gpu_minutes_used: 0, gpu_minutes_total: 100, gpu_minutes_remaining: 100, tokens_used: 0, quartile: 0 },
    calibration: { cross_device_sigma_pct: 0.3 },
    prediction_calibration: [], queued_specs: [], postmortem_due: [], running_experiments: 0,
  }, over),
})

const IMPL_OK = commit => ({ grounded: true, commit, build_ok: true, correctness_passed: true, summary: 'changed the copy loop', self_reported_changes: ['swapped to bulk copies'], gave_up: false })
const REVIEW_OK = head => ({ grounded: true, clone_head: head, independent_correctness: true, sanitizer_clean: true, paths_ok: true, codex_ran: true, diff_review_pass: true, intent_compliance: true, objections: [], accept: true })
const SCREENF = (rel, cand) => ({ grounded: true, rel_speedup_pct: rel, parent_mean_us: 1000, cand_mean_us: cand, within_device_std_pct: 0.3, device: '2' })
const NCUF = { grounded: true, cand: { fingerprint: 'f1', key_metrics: { sm_pct: 70, mem_pct: 50 } }, parent: { fingerprint: 'f0', key_metrics: { sm_pct: 62, mem_pct: 58 } }, parent_was_cached: true, ncu_path: '/proj/.warpspeed/ncu_cache/exp.json' }
const ANALYSTF = { grounded: true, bottleneck: 'DRAM bandwidth', evidence: 'mem down', hypothesis_verdict: 'confirmed', suggested_directions: ['tma'], escalate_full_ncu: false, diagnosis: 'mechanism moved as predicted' }
const FINISHED = { grounded: true, written: true, status: 'recorded' }
const REG = ids => ({ grounded: true, registered: ids.map(exp_id => ({ exp_id, worktree: `/proj/.warpspeed/worktrees/${exp_id}`, branch: 'b' })), failed: [] })

const SPEC = (parent, tags, pred) => ({ type: 'exploit', parent_commit: parent, hypothesis: 'use tma bulk copies for the stage loads', direction_tags: tags, predicted_gain_pct: pred, predicted_mechanism: 'fewer issued load instructions' })

const BASE_ARGS = {
  project_dir: '/proj', build_command: './build.sh', binary_path: 'bin/k',
  kernel_paths: 'src/kernel.cu', target_gpu: 'H100', iterations: 5,
}

let failures = 0
async function scenario(name, fn) {
  try {
    await fn()
    console.log(`PASS  ${name}`)
  } catch (e) {
    failures++
    console.log(`FAIL  ${name}`)
    console.log('      ' + String(e && e.stack || e).split('\n').slice(0, 6).join('\n      '))
  }
}

// ---------------------------------------------------------------------------
// A. Happy path: new_best with confirm routing; compile_error sibling; stop on
//    target latency at the next round boundary.
// ---------------------------------------------------------------------------
await scenario('A: new_best + confirm routing + target-latency stop', async () => {
  const fixtures = {
    'init:materialize': INIT(), calibrate: CAL, 'seed:baseline': SEED,
    'r1:snapshot': SNAP(),
    'r1:plan': { grounded: true, specs: [SPEC('basecommit00', ['tma'], 10), Object.assign(SPEC('basecommit00', ['persistent_kernel'], 5), { type: 'wildcard' })] },
    'r1:register': REG(['r1e1', 'r1e2']),
    'r1e1:impl1': IMPL_OK('c1cand'), 'r1e1:review1': REVIEW_OK('c1cand'),
    'r1e1:screen': SCREENF(6.0, 943),
    'r1e1:confirm': { grounded: true, latency_us_mean: 940, latency_us_std: 2, reps: 200, clocks: { graphics_mhz: 1980 } },
    'r1e1:ncu': NCUF, 'r1e1:analyst': ANALYSTF, 'r1e1:finish': FINISHED,
    'r1e2:impl1': { grounded: true, commit: 'c2cand', build_ok: false, correctness_passed: false, gave_up: false, failure_reason: 'compiler exploded' },
    'r1e2:finish': FINISHED,
    'r1:checkpoints': { grounded: true, merged: [], new_checkpoints: ['c1cand'], blocked: [], postmortem_due: [] },
    'r1e1:lesson': { grounded: true, appended_ids: ['L0001'] },
    'r1:cleanup': { grounded: true, worktrees_removed: 2, review_dirs_removed: 2 },
    'r2:snapshot': SNAP({
      round: 2, frontier_commit: 'c1cand',
      frontier: [
        { commit: 'c1cand', parent: 'basecommit00', latency_us: 940, assumptions: ['tma'], blocked_count: 0, strategy_set_hash: 'h1', headroom_pct: 30, is_frontier: true },
        { commit: 'basecommit00', parent: null, latency_us: 1000, assumptions: [], blocked_count: 0, strategy_set_hash: 'h0', headroom_pct: 38, is_frontier: false },
      ],
      leaderboard: [{ commit: 'c1cand', latency_us: 940, assumptions: ['tma'], round_created: 1 }],
    }),
    report: { grounded: true, report_path: '/proj/.warpspeed/results/report.md', best_commit: 'c1cand', best_latency_us: 940, baseline_latency_us: 1000, lessons_total: 1 },
  }
  const { result, calls, labels } = await run(Object.assign({ target_latency_us: 950 }, BASE_ARGS), fixtures)

  assert.equal(result.ok, true)
  assert.equal(result.stop_reason, 'target_latency_reached')
  assert.equal(result.rounds_run, 1)
  assert.equal(result.speedup, 1.064)
  assert.equal(result.best_commit, 'c1cand')

  // routing: new-best claim -> confirm ran; failed sibling never reached screen
  assert.ok(labels.has('r1e1:confirm'), 'confirm must run on a new-best claim')
  assert.ok(!labels.has('r1e2:screen'), 'compile_error candidate must not be screened')
  assert.deepEqual(result.history[0].statuses, { new_best: 1, compile_error: 1 })

  // allocation quotas rendered for 2 fresh specs at quartile 0: 1/1/0
  const plan = promptOf(calls, 'r1:plan')
  assert.ok(plan.includes('exploit (parent = the [frontier] checkpoint): 1'), 'exploit quota')
  assert.ok(plan.includes('): 1') && plan.includes('wildcard') && plan.match(/wildcard.*: 0/), 'wildcard quota 0')

  // implementor context: wiki pages by tag + scoped lesson query, never the whole wiki
  const impl = promptOf(calls, 'r1e1:impl1')
  assert.ok(impl.includes('/ws/wiki/tma-bulk-async-copy.md'))
  assert.ok(impl.includes('lessons-query --tags tma --arch sm90 --max 15'))
  assert.ok(impl.includes('[r1e1] use tma bulk copies'), 'structured commit message format')

  // reviewer: cross-model CLI is invoked and relayed
  const rev = promptOf(calls, 'r1e1:review1')
  assert.ok(rev.includes('codex exec'))
  assert.ok(rev.includes('never substitute'), 'relay-not-override instruction')

  // recorded row: status + prediction gap (10 predicted - 6 achieved = 4)
  const fin = promptOf(calls, 'r1e1:finish')
  assert.ok(fin.includes('"status":"new_best"'))
  assert.ok(fin.includes('"prediction_gap_pct":4'))
  assert.ok(fin.includes('"bench_tier":"confirm"'))

  // record barrier: dedup candidates include hash + fingerprint
  const ck = promptOf(calls, 'r1:checkpoints')
  assert.ok(ck.includes('"fingerprint":"f1"') && ck.includes('"strategy_set_hash"'))

  // stop fired BEFORE round-2 planning
  assert.ok(!labels.has('r2:plan'))
})

// ---------------------------------------------------------------------------
// B. Review-objection loop + both no-confirm and fuzzy-band-confirm routing.
// ---------------------------------------------------------------------------
await scenario('B: review loop + routing margins + correct_slower', async () => {
  const snap = SNAP({
    frontier: [
      { commit: 'basecommit00', parent: null, latency_us: 1000, assumptions: [], blocked_count: 0, strategy_set_hash: 'h0', headroom_pct: 38, is_frontier: true },
      { commit: 'othernode000', parent: 'basecommit00', latency_us: 1200, assumptions: ['x'], blocked_count: 1, strategy_set_hash: 'hx', headroom_pct: 20, is_frontier: false },
    ],
  })
  const fixtures = {
    'init:materialize': INIT(), calibrate: CAL, 'seed:baseline': SEED,
    'r1:snapshot': snap,
    'r1:plan': { grounded: true, specs: [Object.assign(SPEC('othernode000', ['tma'], 12), { type: 'explore' }), SPEC('basecommit00', ['tma'], 2)] },
    'r1:register': REG(['r1e1', 'r1e2']),
    // r1e1: reviewer objects once, implementor fixes, second review accepts.
    'r1e1:impl1': IMPL_OK('c1b'),
    'r1e1:review1': Object.assign(REVIEW_OK('c1b'), { accept: false, diff_review_pass: false, objections: ['missing async-proxy fence before the TMA store'], objection: 'missing async-proxy fence before the TMA store' }),
    'r1e1:impl2': IMPL_OK('c1b2'),
    'r1e1:review2': REVIEW_OK('c1b2'),
    // rel +8% on a NON-best parent (1200us): projected 1111us > best 1000us
    // -> no new-best claim; |8| >= margin 5 -> NO confirm.
    'r1e1:screen': SCREENF(8.0, 1111),
    'r1e1:ncu': NCUF, 'r1e1:analyst': ANALYSTF, 'r1e1:finish': FINISHED,
    // r1e2: +0.5% -> insignificant, but inside the fuzzy band -> confirm runs.
    'r1e2:impl1': IMPL_OK('c2b'), 'r1e2:review1': REVIEW_OK('c2b'),
    'r1e2:screen': SCREENF(0.5, 995),
    'r1e2:confirm': { grounded: true, latency_us_mean: 996, latency_us_std: 1, reps: 200, clocks: {} },
    'r1e2:ncu': NCUF, 'r1e2:analyst': Object.assign({}, ANALYSTF, { hypothesis_verdict: 'refuted' }), 'r1e2:finish': FINISHED,
    'r1:checkpoints': { grounded: true, merged: [], new_checkpoints: ['c1b2'], blocked: [{ commit: 'basecommit00', blocked_count: 1 }], postmortem_due: [] },
  }
  const { result, calls, labels } = await run(Object.assign({}, BASE_ARGS, { iterations: 1 }), fixtures)

  assert.equal(result.stop_reason, 'iteration_limit_reached')
  assert.deepEqual(result.history[0].statuses, { correct_faster: 1, correct_slower: 1 })

  // review loop: objection threaded into the second implementor call
  assert.ok(labels.has('r1e1:impl2') && labels.has('r1e1:review2'))
  assert.ok(promptOf(calls, 'r1e1:impl2').includes('missing async-proxy fence'))
  assert.ok(promptOf(calls, 'r1e1:finish').includes('"review_iterations":2'))

  // routing: big-delta non-best skips confirm; fuzzy band confirms
  assert.ok(!labels.has('r1e1:confirm'), 'clear delta on non-best parent must skip tier-2')
  assert.ok(labels.has('r1e2:confirm'), 'inside CONFIRM_MARGIN the fuzzy band must confirm')

  // correct_slower still profiled + diagnosed (highest-value lessons)
  assert.ok(labels.has('r1e2:ncu') && labels.has('r1e2:analyst'))
  assert.ok(promptOf(calls, 'r1e2:finish').includes('"status":"correct_slower"'))
})

// ---------------------------------------------------------------------------
// C. Post-mortem: blame confirmed -> rewind + replay queue.
// D. Post-mortem: blame refuted -> subtree kept, refutation lesson.
// ---------------------------------------------------------------------------
function pmFixtures(ablationConfirmLat) {
  const snap = SNAP({
    frontier: [
      { commit: 'basecommit00', parent: null, latency_us: 1000, assumptions: [], blocked_count: 0, strategy_set_hash: 'h0', headroom_pct: 38, is_frontier: false },
      { commit: 'ckbad1234567', parent: 'basecommit00', latency_us: 995, assumptions: ['persistent_kernel'], blocked_count: 4, strategy_set_hash: 'hb', headroom_pct: 10, is_frontier: true },
    ],
    frontier_commit: 'ckbad1234567',
  })
  return {
    'init:materialize': INIT({ PARALLEL_AGENTS: 1 }), calibrate: CAL, 'seed:baseline': SEED,
    'r1:snapshot': snap,
    'r1:plan': { grounded: true, specs: [SPEC('ckbad1234567', ['tma'], 6)] },
    'r1:register': REG(['r1e1']),
    'r1e1:impl1': { grounded: true, commit: '', build_ok: true, correctness_passed: true, gave_up: true, failure_reason: 'no viable change found' },
    'r1e1:finish': FINISHED,
    'r1:checkpoints': { grounded: true, merged: [], new_checkpoints: [], blocked: [{ commit: 'ckbad1234567', blocked_count: 5 }], postmortem_due: ['ckbad1234567'] },
    'r1:pm:ckbad123': {
      grounded: true, suspect_strategy: 'persistent_kernel', suspect_checkpoint: 'ckbad1234567',
      suspect_parent: 'basecommit00', mechanism: 'persistent grid eats the SMEM the pipeline needs',
      ablation_hypothesis: 'rebuild stage pipeline without the persistent grid',
      ablation_tags: ['tma'], ablation_predicted_gain_pct: 4, ablation_predicted_mechanism: 'frees smem for depth-3',
    },
    'r1:ablation:register': REG(['r1ab1']),
    'r1ab1:impl1': IMPL_OK('abcommit0001'), 'r1ab1:review1': REVIEW_OK('abcommit0001'),
    'r1ab1:screen': SCREENF(0.2, 998),
    'r1ab1:confirm': { grounded: true, latency_us_mean: ablationConfirmLat, latency_us_std: 1, reps: 200, clocks: {} },
    'r1ab1:ncu': NCUF, 'r1ab1:analyst': ANALYSTF, 'r1ab1:finish': FINISHED,
    'r1:rewind': { grounded: true, rewound_to: 'basecommit00', retired: ['ckbad1234567'], lesson_id: 'L0009', replay_queued: true },
    'r1:pm-refuted': { grounded: true, appended_ids: ['L0010'] },
  }
}

await scenario('C: postmortem blame CONFIRMED -> rewind + replay', async () => {
  // ablation reaches 997us vs best 995us (within MIN_GAIN margin) => suspect not load-bearing
  const { calls, labels } = await run(Object.assign({}, BASE_ARGS, { iterations: 1 }), pmFixtures(997))
  assert.ok(labels.has('r1:pm:ckbad123'), 'postmortem must trigger on blocked checkpoint')
  assert.ok(labels.has('r1ab1:impl1'), 'ablation runs through the SAME candidate path')
  assert.ok(labels.has('r1:rewind'), 'confirmed blame must rewind')
  assert.ok(!labels.has('r1:pm-refuted'))
  const rw = promptOf(calls, 'r1:rewind')
  assert.ok(rw.includes('"suspect_checkpoint": "ckbad1234567"'))
  assert.ok(rw.includes('"type":"replay"'), 'replay spec must be queued (knowledge merge)')
  assert.ok(rw.includes('assumption_invalidated'))
})

await scenario('D: postmortem blame REFUTED -> subtree kept', async () => {
  // ablation lands at 1100us, far from best 995us => the strategy was load-bearing
  const { labels } = await run(Object.assign({}, BASE_ARGS, { iterations: 1 }), pmFixtures(1100))
  assert.ok(labels.has('r1:pm:ckbad123'))
  assert.ok(!labels.has('r1:rewind'), 'refuted blame must NOT rewind')
  assert.ok(labels.has('r1:pm-refuted'), 'refutation lesson must be recorded')
})

// ---------------------------------------------------------------------------
// E. Guards: missing args fail fast; reviewer unavailable aborts before GPU work.
// ---------------------------------------------------------------------------
await scenario('E: fail-fast arg validation + reviewer preflight gate', async () => {
  await assert.rejects(() => run({ project_dir: '/proj' }, {}), /missing required arg/)

  const init = INIT()
  init.preflight.reviewer_ok = false
  const { result, labels } = await run(BASE_ARGS, { 'init:materialize': init })
  assert.equal(result.ok, false)
  assert.ok(/reviewer/i.test(result.action_required))
  assert.ok(!labels.has('calibrate'), 'no GPU work after failed reviewer preflight')

  // harness-not-ready returns the human action early
  const init2 = INIT()
  init2.harness_ready = false
  init2.harness_scaffolded = true
  init2.action_required = 'Complete the scaffolded harness'
  const r2 = await run(BASE_ARGS, { 'init:materialize': init2 })
  assert.equal(r2.result.ok, false)
  assert.ok(r2.result.harness_scaffolded)
  assert.ok(!r2.labels.has('calibrate'))
})

console.log('----------------------------------------')
if (failures) {
  console.log(`dryrun: ${failures} scenario(s) FAILED`)
  process.exit(1)
}
console.log('dryrun: all scenarios passed')
