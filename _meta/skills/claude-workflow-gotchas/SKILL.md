---
name: claude-workflow-gotchas
description: Collects hard rules and common pitfalls when creating Claude Code workflows. Use when writing, reviewing, or debugging `.claude/workflows/*.js`, especially for runtime-incompatible patterns (for example `new Date()` / `Date.now()`), schema mismatches, and phase/agent contract issues.
disable-model-invocation: true
---

# Claude Workflow Gotchas

## Quick checks (run before saving)

- [ ] No runtime time APIs in workflow code (`new Date()`, `Date.now()`)
- [ ] Any timestamp/date needed by prompts is passed through `args`
- [ ] All `agent(..., { schema })` required fields match downstream usage
- [ ] `phase(...)` names are stable and meaningful
- [ ] Return object only includes serializable values

## Hard rule: no `new Date()` / `Date.now()`

Claude Code Workflow runtime does not allow direct time API calls in workflow JS.

### Bad

```js
const startedAt = new Date().toISOString()
const today = new Date().toISOString().split('T')[0]
```

### Good

```js
const RUN_TS = args.run_timestamp_iso || 'unknown'
const RUN_DAY = args.run_date || 'unknown-date'
```

Use `RUN_TS` / `RUN_DAY` in prompt templates, report headers, and metadata output.

## Case from production failure

Observed failure pattern during workflow authoring:

1. `new Date().toISOString().split('T')[0]` used in generated markdown header
2. `new Date().toISOString().split('T')[0]` used in TRAPS template
3. `new Date().toISOString()` written to final JSON (`started_at`)

Safe fix applied:

- remove inline date from template text when not critical
- replace runtime date with args-derived field when needed
- for immutable record fields, use `args.run_timestamp_iso` or fallback placeholder

## Recommended arg contract

When a workflow may need timestamps, define these optional args:

- `run_timestamp_iso`: full ISO timestamp, example `2026-06-02T15:00:00+08:00`
- `run_date`: date-only string, example `2026-06-02`

Example invocation:

```js
Workflow({
  name: 'your-workflow',
  args: {
    run_timestamp_iso: '2026-06-02T15:00:00+08:00',
    run_date: '2026-06-02',
  },
})
```

## Review checklist for workflow PRs

- Search for forbidden time calls: `new Date(`, `Date.now(`
- Verify every schema `required` key is actually produced by the agent
- Verify fields consumed later are present in schema (avoid hidden undefined)
- Ensure fallback values are explicit for optional args
- Keep prompts deterministic; avoid hidden ambient assumptions

## Minimal remediation playbook

If workflow fails after introducing dynamic time text:

1. Remove all direct time API calls
2. Introduce timestamp/date args in workflow invocation contract
3. Replace affected templates with args-based variables
4. Re-run and confirm no runtime rejection
