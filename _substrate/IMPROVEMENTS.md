# Pattern improvements (from Claude Code dynamic-workflows best practices)

Concrete, prioritized improvements to the substrate / generalist / KerSor patterns,
mapped from Anthropic's dynamic-workflows guidance. Companion to
[`SOLVER-SDK.md`](./SOLVER-SDK.md) and [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Organizing principle (the guardrail)

> **Deterministic scripts where ground truth exists; LLM-orchestration patterns only
> where judgment is needed.**

Kernel optimization has executable ground truth (compile / correctness / speedup /
NCU). So we keep the deterministic substrate (Layers A–F) for those, and adopt the
article's LLM patterns (adversarial verification, tournament, fan-out) **only** for
the genuinely subjective parts (plan choice, insight attribution, code review). This
is our advantage over generic agent-orchestration — don't dilute it.

## Priority table

| # | Improvement | Borrows pattern | Touches | Value / Cost | Pri |
|---|---|---|---|---|---|
| 1 | Token-budget wiring | token budgets | generalist loop, experiment | scale depth to budget; hard cap | **P0 ✅ done** |
| 2 | Worktree isolation for parallel candidates | migrations | generalist Evaluate | prevents parallel file-edit conflicts | **P0 ✅ done** |
| 3 | Model/intelligence routing | model routing | generalist `agent({model})`, profile | big token savings | **P0 ✅ done** |
| 4 | Adversarial insight verifier | adversarial verification | generalist Learn (+ KerSor analyzer TODO) | sharper transfer-object confidence | P1 ✅ done (generalist) |
| 5 | Loop-until-dry stop condition | loop until done | generalist (+ KerSor hook TODO) | covers the tail, fewer premature stops | P1 ✅ done (generalist) |
| 6 | `/loop` + `/goal` experiment harness | loop+goal | experiment-design.md | reproducible, unattended batch runs | P1 ✅ documented |
| 7 | Dynamic solver synthesis | dynamic vs static | KerSor + `_meta` generator | bespoke-per-kernel (future work) | P2 |
| 8 | Tournament for subjective sub-choices | tournament | generalist Plan | only for plan choice, NOT kernel ranking | P2 |
| 9 | Quarantine for spec-to-kernel mode | triage quarantine | KerSor task-directory mode | security hygiene on untrusted task dirs | P2 |

---

## P0 — implement now (high value, low cost)

### 1. Token-budget wiring
The Workflow runtime exposes a `budget` object. Scale search depth to it and hard-stop.
- **generalist**: `while (budget.total && budget.remaining() > PER_ROUND_EST) { ... }`;
  set `BREADTH = budget.total ? Math.min(3, Math.floor(budget.remaining()/EST)) : 3`.
- **experiment**: pair each KerSor run with an explicit token cap (prompt "use Nk tokens"),
  recorded alongside the GPU-hour budget already in `EXPERIMENT-SOLVER-SET.md`.

### 2. Worktree isolation for parallel candidates
The generalist's Evaluate phase implements `BREADTH` candidates in parallel; they edit
files. Pass `isolation: 'worktree'` to those `agent()` calls so concurrent edits don't
clobber each other; worktrees auto-clean if unchanged. (Article's migration pattern:
"spin off a subagent per fix in a worktree.")

### 3. Model / intelligence routing
Most agent steps are mechanical (run a substrate script, run `eval_command`, parse JSON)
and do not need Opus. Route by role:
- **mechanical** (run `diagnose.py`/`memory_store.py`/`anti_cheat.py`, profiling) → Haiku/Sonnet.
- **judgment** (planning, code generation/repair) → Opus, and only escalate for
  high-complexity kernels (signal from profile: design-space size, kernel LOC, op type).
Add a tiny `complexity` field to the profile and set `agent({model})` accordingly. This is
the cheapest large win — most tokens are spent on mechanical steps.

## P1 — valuable, moderate cost

### 4. Adversarial insight verifier (complements, does not replace, anti_cheat)
`anti_cheat.py` already gives deterministic ground-truth verdicts for
compile/correct/speedup. But **insight attribution** ("the bottleneck is long_scoreboard at
M∈[144,192]") has no executable ground truth. Before such an insight is written to the
transfer object as `confidence: measured`, spawn a refuter agent ("try to refute this
attribution from the profile data; default to refuted if uncertain"). If refuted →
downgrade to `hypothesized`. This raises the trust of transferred evidence without
touching the deterministic path. (Article: "panel of verifiers and refuters.")
**Done in the generalist**: a refuter agent (judgment) produces `{refuted}`, then the
deterministic `verify_insight.py` applies the downgrade rules (non-executable evidence
caps at `inferred`; refuted downgrades one level). **TODO**: wire the same
`verify_insight.py` into KerSor's `result-analyzer` so the cross-solver transfer object
gets the same treatment (a KernelNav-repo change, deferred).

### 5. Loop-until-dry stop condition
Today the meta-loop stops at `max_workflows` or an LLM `STALLED`. Add a deterministic
"dry" counter: stop when **K consecutive rounds produce no new `confidence: measured`
insight** (loop-until-done). Catches the tail that a fixed budget misses, and is more
principled than an LLM self-declared stall.

### 6. `/loop` + `/goal` experiment harness
Run the experiment matrix unattended: `/loop` at an interval to step through
(task × condition × seed), `/goal` as the hard completion gate (all cells done), with a
token budget per cell. Document the exact invocation in the experiment runner so a batch
run is reproducible and survives interruption (workflows resume).

## P2 — future / speculative

### 7. Dynamic solver synthesis (future work, already flagged in the paper)
The substrate + `_meta/generate-workflow.js` can assemble a bespoke solver from components
per kernel, instead of selecting from the fixed pool. Pushes "集大成" further but blurs the
fixed-portfolio thesis — keep out of the main experiment.

### 8. Tournament — ONLY for subjective sub-choices
Comparative judgment beats absolute scoring for **subjective** picks (which plan to pursue).
Use a pairwise-tournament among `BREADTH` plans **before** implementation if plans are hard
to score a priori. Do **not** use it to rank finished kernels — speedup is objective ground
truth; absolute measurement wins there.

### 9. Quarantine for spec-to-kernel mode
In task-directory mode KerSor reads arbitrary files (README/spec). Apply the triage
quarantine pattern: the agent that reads untrusted task content must not take high-privilege
actions (file writes outside the run dir, shell beyond the allowlist); acting is done by a
separate agent on the structured task-spec only.

---

## What NOT to change (keep the advantage)
- Keep deterministic ground-truth checks (compile/correct/speedup, anti-cheat reward,
  diagnosis thresholds, method gate, beam top-K). Do not replace them with LLM verifiers
  or tournaments — that would forfeit the executable-feedback advantage.
- Keep one topology per solver. None of these improvements add a second search topology to
  a single solver; topology heterogeneity stays at the KerSor layer.
