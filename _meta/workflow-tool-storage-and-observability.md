# Claude Code `Workflow` tool — storage & observability

Status: reference, compiled 2026-06-17

Reference notes on **how the Claude Code `Workflow` tool stores state and what is
observable from outside a running workflow**. This is the factual basis for the
genome self-report contract (`genome-trajectory-schema.md`): we added a scribe
because the runtime gives you almost nothing to observe a running workflow from
outside.

> **Reliability marker.** Each claim is tagged **[D]** documented in the tool
> contract / Claude Code docs, or **[I]** inferred / implementation-detail
> observed in practice (may change between versions — verify before relying).
> When in doubt, treat **[I]** as a hypothesis, not a guarantee.

## 1. Execution model

- **[D]** A `Workflow({ scriptPath | name, args })` call runs the orchestration
  script in an **isolated background runtime, separate from the calling
  conversation**. The tool returns almost immediately with a **run id**; a
  completion notification arrives later. You watch progress with `/workflows`.
- **[D]** Intermediate results stay in **script variables**, NOT in the calling
  agent's context. This is the whole point — it keeps a multi-agent fan-out's
  chatter out of the main context window (token efficiency).
- **[D]** The script body is **sandboxed**: no filesystem access, no Node.js
  APIs, and `Date.now()` / `Math.random()` / argless `new Date()` **throw**
  (they would break deterministic resume). Standard JS built-ins otherwise work.
- **[D]** Concurrency inside one workflow is capped at ~`min(16, cores−2)`
  simultaneous `agent()` calls; total agents per run are capped (a runaway
  backstop); a single `parallel()`/`pipeline()` takes at most a few thousand items.

**Consequence:** the only component inside a workflow that can touch the
filesystem is a **subagent** spawned by `agent()` — subagents are real agents
with tools (Bash/Write). The orchestration script itself cannot write a file.

## 2. What is persisted to disk

| Artifact | Where | Tag | Notes |
|---|---|---|---|
| The orchestration **script** | under the run's session directory | **[D]** | Every invocation persists its script; the tool result returns its path. Used for resume. |
| Per-subagent **transcript** | `agent-<id>.jsonl` in the run's transcript directory | **[I]** | The resume guidance says, as a fallback, to "read `agent-<id>.jsonl` files in the transcript directory." So per-agent JSONL exists on disk — but the exact path, retention, and whether it is written live vs at completion are NOT contractually specified. |
| Resume **journal** | run-scoped | **[I]** | `resumeFromRunId` replays completed `agent()` results from a journal; the journal format is internal. |
| The **main conversation** | `~/.claude/projects/<project>/<session-uuid>.jsonl` | **[D]** | This is the *calling* session's transcript. It records that the `Workflow` tool was called and its return value — **not** the workflow's internal agents. |

**Key distinction:** the main-session JSONL (by session UUID) and the workflow's
internal agent transcripts are **different things**. A workflow's internals do
**not** appear in the calling session's `<uuid>.jsonl`.

## 3. What is observable from outside while it runs

| Surface | Tag | What you get |
|---|---|---|
| `/workflows` TUI | **[D]** | Live, in-Claude-Code view: per-phase agent count / token totals / elapsed; drill into an agent to see its prompt, recent tool calls, result. **Visual only — not a machine-readable stream.** |
| `log(msg)` | **[D]** | A narrator line above the progress tree (for the human). **Not written to a file.** |
| `phase(title)` | **[D]** | Groups subsequent `agent()` calls under that title in the progress display. **Not written to a file.** |
| Task panel | **[I]** | A one-line summary of the run while it executes. |
| External tail / hook | **[D-absent]** | There is **no documented** machine-readable live stream, file protocol, or hook for an external process to follow a running workflow's internals. |

**Net:** to a program (a hook, a script, a different process), a running workflow
is a **black box**. The only built-in window is the `/workflows` UI, for a human.

## 4. Session vs. workflow-agent identity

- **[D]** A Claude Code **session** is a conversation with its own UUID under
  `~/.claude/projects/<project>/`.
- **[I]** Agents spawned by `agent()` inside a workflow are **not** separate
  sessions and do **not** get their own session UUIDs; the runtime tracks them by
  an internal agent id (the `agent-<id>` of §2).
- **The opacity is a property of the `Workflow` tool, not of sessions.** Wrapping
  a workflow inside another session does **not** expose its internals — the
  calling session still only sees the tool call + return. To observe the steps,
  you must run them as something *other* than the Workflow tool (see §6).

## 5. Resume / determinism

- **[D]** Re-invoking with `resumeFromRunId` replays the longest unchanged prefix
  of `agent()` calls from cache instantly; the first changed/new call and
  everything after it runs live. Same script + same args → full cache hit.
- **[D]** This is why the sandbox bans `Date.now()`/`Math.random()`/`new Date()`
  — nondeterminism would break the replay. Stamp timestamps *after* the workflow
  returns, or have a subagent read the clock via Bash.

## 6. Three ways to get observability (and the trade-off)

| Route | How | Observability | Cost |
|---|---|---|---|
| **A. Workflow tool (default)** | `Workflow({scriptPath})` | Black box; only `/workflows` UI + the return envelope | Cheapest; built-in caching / budget / concurrency / resume |
| **B. Self-orchestrated agents / child session** | drive steps yourself via `Agent`/`Task` or a child `claude` session | Each step has a readable transcript an external process can tail | You lose the Workflow runtime's caching/resume/budget; more orchestration to build; AKW workflows are written as Workflow scripts (`agent()`/`phase()`/`parallel()`), so they can't run as a plain `claude -p` without re-hosting |
| **C. Cooperative self-report (what AKW does)** | keep the Workflow tool, but each `phase()` has a subagent append to `${args.exp_dir}/genome.jsonl` | Real-time, tail-able stage trace, with no change to the execution model | One cheap subagent per stage entry; needs the workflow author's cooperation (the codemod) |

AKW chose **C**: see `genome-trajectory-schema.md` and
`scripts/patch-genome-report.js`. KerSor points `args.exp_dir` inside its session
directory (`.kersor/<session>/`), so `tail -f .kersor/<session>/genome.jsonl`
follows a running workflow live.

## 7. Trust boundary (important)

Anything a workflow self-reports (genome.jsonl, the return envelope, written
kernels) is **work-plane and agent-writable — forgeable**. Use it for
observability, technique mining, and feeding the recombiner. **Never** treat it
as the trust anchor for "did the work actually happen / is the speedup real."
That must rest on the orchestrator's out-of-band re-measurement and an
unforgeable dispatch witness (in KerSor: a `Workflow` PreToolUse/PostToolUse hook
writing `run-N/.dispatch-witness.jsonl`, which the agent's own tools cannot
write). Do not conflate observability with anti-cheat provenance.

## 8. One-line summary

The `Workflow` tool deliberately runs as an isolated, resumable black box: its
internal agents are not sessions, leave no externally-streamable trail, and only
surface to a human via `/workflows`. Real-time external observability requires
either abandoning the tool (route B) or cooperative self-report from inside it
(route C) — and either way the emitted data is forgeable, not provenance.
