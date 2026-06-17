# Genome / trajectory self-report contract

Status: proposed 2026-06-17

Workflows run inside the Claude Code `Workflow` tool, whose orchestration script
is sandboxed: it cannot touch the filesystem and cannot call `Date.now()` /
`Math.random()`. The only way a running workflow can emit data to disk is through
a subagent (`agent()`), which has Bash/Write tools. This contract defines a
lightweight **self-report** that makes a running workflow observable from the
outside in real time, and records the ordered stages it actually performed.

For the full picture of *why* this is the only option — how the `Workflow` tool
stores state and what (little) is observable from outside a running workflow —
see [`workflow-tool-storage-and-observability.md`](./workflow-tool-storage-and-observability.md).

## Why

KerSor (the orchestrator) sees a workflow as a black box: it gets the final
return envelope, but nothing about *which stages ran, in what order, and what
each produced* while the workflow is running. `docs/experiment-ledger.md`
(KerSor side) calls this the deferred **genome** (record #3, ordered stages) and
**trajectory** (record #4, per-candidate technique→Δ). This contract is how a
workflow supplies them.

## Files (written under `args.exp_dir`)

KerSor passes `args.exp_dir` pointing inside its session directory
(`.kersor/<session>/`), so these land where an external process can `tail` them:

- `genome.jsonl` — one line per **stage entry** (and per loop iteration), emitted
  live as the workflow crosses each `phase()`.
- `trajectory.jsonl` — optional, one line per **candidate evaluated**
  (technique → measured Δ). Emitted by evaluation phases that already produce
  candidates. (Phase 2 — not auto-injected yet.)

Both are append-only JSONL. A missing file simply means the workflow predates the
contract or `exp_dir` was unset.

## `genome.jsonl` line schema

```json
{
  "workflow": "ksearch-kernel-optimization",
  "phase": "Evaluate",
  "ts": "2026-06-17T09:12:33Z",
  "status": "entered",
  "candidate_id": "cycle_2_a1",
  "technique": "async_copy_pipeline",
  "speedup": 1.31,
  "parent_origin": "accelopt-kernel-optimization"
}
```

| field | required | meaning |
|---|---|---|
| `workflow` | yes | `meta.name` of the running (or recombined) workflow |
| `phase` | yes | the `phase()` title just entered |
| `ts` | yes | UTC `date -u +%Y-%m-%dT%H:%M:%SZ` (from the scribe's shell — the script cannot call Date) |
| `status` | yes | `entered` \| `done` \| `error` |
| `candidate_id` | no | when the stage is about one candidate |
| `technique` | no | primitive move (see `KerSor config/technique-taxonomy.json`) |
| `speedup` | no | measured speedup at this point, if known |
| `parent_origin` | no | for recombined workflows: which parent this phase came from |

## Trust boundary (read this)

`genome.jsonl` / `trajectory.jsonl` are **self-reported by the workflow** — they
live on the *work plane* and are agent-writable, i.e. forgeable. They are for
**observability, technique mining, and feeding the recombiner** — never as the
trust anchor for loop-completion. Loop verdicts must still rest on KerSor's
hook-side re-measurement and the unforgeable `run-N/.dispatch-witness.jsonl`. Do
not conflate observability with anti-cheat provenance.

## How it is injected

`scripts/patch-genome-report.js` is an idempotent codemod that, for each
workflow, injects a small `__genomeReport()` helper and a scribe call after every
standalone `phase('X')`. New workflows inherit it via `_templates/` +
`_tools/generate-workflow.js`. KerSor's `workflow-recombiner` emits the same
helper so recombined workflows are observable from birth.
