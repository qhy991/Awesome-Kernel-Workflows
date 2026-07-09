# AGENTS.md — working rules for agents in this repo

Instructions for any AI agent (Claude Code, Codex, …) changing
Awesome-Kernel-Workflows (AKW). Human contributors should follow them too.

When adding or modifying a kernel-optimization workflow, follow
[Agent.md](Agent.md) — the workflow maintenance guide. This file adds the
**runtime & consolidation rules** (below) and the **versioning & changelog
policy**. The common `AGENTS.md` filename serves as the entry point, so the
non-negotiable rules that keep 32 workflows consistent live here.

## Workflow code: the non-negotiable rules

A workflow is dispatched through Claude Code's `Workflow` tool, which loads the
`.js` as a **sandboxed script** (not a module) and injects
`agent/phase/parallel/pipeline/log/budget/args` as globals. Everything below
follows from that fact, and every rule is enforced by a CI guard — a PR that
violates one hard-fails. Getting it right by construction is cheaper than a
bounced PR.

### 1. Never hand-author the shared helper blocks — they have a single source

The mechanical helpers every workflow needs (`agentRetry`, `withTurnTimeout`,
`guard`, `expect`, `__attemptBlock`, `__experienceBlock`,
`normalizeSuitabilityValue`, `resolveBackendAxis`, `driverSh`, `driverPath`, the
arg-guard and typed-args blocks) are **single-sourced** in `_meta/scaffolding/`
and inlined into each workflow between sentinels:

```js
// --- BEGIN inlined backend-axis (resolve) scaffolding (from _meta/scaffolding/backend-axis.js) ---
...canonical block, byte-for-byte...
// --- END inlined backend-axis (resolve) scaffolding ---
```

Rules:

- **The region between `BEGIN inlined X` / `END inlined X` is generated. Do NOT
  edit it in a workflow `.js`.** A hand-edit to one copy is drift, and the
  matching guard test fails.
- To change a helper, edit the SSOT in `_meta/scaffolding/<X>.js`, then re-sync
  every workflow with that helper's codemod under `scripts/patch-<X>.js`
  (`patch-backend-axis.js`, `patch-arg-guard.js`, `patch-typed-args.js`,
  `patch-embedded-eval.js` take `--refresh`; `patch-turn-timeout.js` /
  `add-agent-retry-scaffolding.js` take a file list or `--all`). **The failure
  message of each guard test prints the exact command to run** — use that; don't
  guess the flag. Guards: `backend-axis-ssot-guard`, `scaffolding-ssot-debt-guard`,
  `agent-retry-guard-lint`, `turn-timeout-propagation-guard`, `typed-args-ssot`.
- When you write a **new** workflow, do not paraphrase these helpers from memory
  or from a nearby workflow — run the codemod so you get the current canonical
  block verbatim. A paraphrase that "looks equivalent" is the #1 source of drift.
- If a helper genuinely has no SSOT yet (currently `langToken` / `fenceToken`),
  say so in the PR and add the SSOT file — do not silently inline a fresh copy.

### 2. Eligibility lives in the manifest, not in the workflow body

Backend / language / problem-type eligibility is declared in `manifest.yaml`
`routing.accepts:` and enforced by the **KerSor selector** (issue #24). Therefore:

- **Do NOT emit `WORKFLOW_SUITABILITY` or `assertWorkflowSuitability()`.** They are
  retired; only two legacy workflows still carry them and both are being migrated.
  A new workflow with them fails the generator checklist.
- If a workflow resolves a backend from args, emit a small `resolveBackend()` that
  **normalizes/derives only — it must NOT throw on eligibility.** The selector
  owns rejection; the workflow does not re-litigate it.
- The manifest is the **contract SSOT**. Declare the contract there
  (`routing.accepts`, `backend.*`, args) and let the workflow read injected
  values — do not re-derive in JS what the manifest already states. Match the
  vocabulary of `docs/manifest-schema.yaml`.

### 3. Runtime sandbox constraints (all CI-enforced)

- **No line-leading ESM `import`.** The entrypoint is a script, not a module.
  There is no `require`/`import` of shared code — that is *why* the helpers in
  rule 1 are inlined rather than imported.
- **No `Date.now()`, `Math.random()`, or argless `new Date()`.** They break
  Workflow-tool resume (values differ across resumes → cached branches diverge).
  Derive any id from `args.run_index` / `args.round_index` / a filesystem counter.
  Enforced by the catalog forbidden-API scan (marks the workflow `known_broken`).
- **Wrap every `await agent()` in `agentRetry(fn, {retries, allowNull})`.** A bare
  `agent()` returning null on a transient 429 crashes the workflow via property
  deref. Enforced by `agent-retry-guard-lint` + `agent-retry-null-safety`.
- **Substrate CLI flags are `--artifact` / `--problem` / `--out`** — never
  `--kernel` / `--test` / `--result` (the run.sh parsers `exit 3` on those).
  Enforced by `substrate-flags-contract`.
- **All file writes go to `args.exp_dir`, never process CWD.** Writing to CWD
  pollutes the KerSor session tree. Enforced by `stray-files-static`.
- **`meta` stays a pure literal** — the Workflow tool extracts it statically.

### 4. Do the minimum, keep the method free

Consolidation removes duplication *below* the method (helpers, contract), never
standardizes the method itself. A workflow's phase spine and optimization logic
stay free-form — that is the point of having many workflows. Do not "standardize"
control flow, and do not mix unrelated refactoring into a workflow change.

## Versioning & changelog (REQUIRED on every change)

AKW follows [Semantic Versioning](https://semver.org/) and keeps a
[Keep a Changelog](https://keepachangelog.com/)-style changelog. **Every change
that touches a workflow, the substrate/templates/tools, the manifest/catalog, or
the docs MUST record a changelog entry.** No silent changes.

### 1. Record the change

Add a bullet under the top `## [Unreleased]` section of **both**:

- `CHANGELOG.md` (English)
- `CHANGELOG.zh-CN.md` (中文)

Keep the two languages in sync. Group under `Added` / `Changed` / `Fixed` /
`Removed` / `Deprecated` / `Security`. Say what changed, why a consumer cares, and
cite the file(s) touched.

### 2. The version lives in `VERSION`

A single semver string in `VERSION` (e.g. `0.1.0`). Bump it only when **cutting a
release**, by the largest applicable rule:

| Bump | When | AKW examples |
|------|------|--------------|
| **MAJOR** `X.0.0` | Backward-**incompatible** change to a workflow's contract — existing dispatches break. | Remove/rename a workflow or a required arg; change a workflow's return-envelope / `meta` contract incompatibly; change the dispatch/args convention KerSor relies on. |
| **MINOR** `0.X.0` | Backward-**compatible** new capability. | Add a new workflow; add an optional arg/phase; a cross-cutting capability (e.g. genome self-report); a backward-compatible substrate/template upgrade. |
| **PATCH** `0.0.X` | Backward-**compatible** fix only. | Bug fix in a workflow, prompt tweak that keeps the contract, doc fix, codemod fix. |

When unsure between two levels, pick the higher one.

### 3. Release steps

1. Pick `X.Y.Z` by the table above.
2. In both changelogs, rename `## [Unreleased]` → `## [X.Y.Z] - YYYY-MM-DD` and add
   a fresh empty `## [Unreleased]` above it.
3. Set `VERSION` to `X.Y.Z`.
4. Commit, then tag: `git tag vX.Y.Z`.

A change not yet released still gets its `## [Unreleased]` entry immediately — the
release step only renames and bumps.

### 4. Keep the workflow count in sync

When a workflow is added or removed, update `badges/workflows.json` `message` to
the new total and note it in the changelog.

## Definition of done

Before considering any workflow change complete:

- The workflow parses. Workflow bodies use top-level `return`/`await` (legal in
  the Claude Code Workflow runtime, not in a plain module), so syntax-check by
  wrapping in an async function:
  `node -e 'const fs=require("fs");let s=fs.readFileSync(F,"utf8");const i=[];s=s.replace(/^[ \t]*import\s.*$/gm,m=>{i.push(m);return""}).replace(/^([ \t]*)export\s+const\s+meta/m,"$1const meta");fs.writeFileSync("/tmp/c.mjs",i.join("\n")+"\nasync function w(){\n"+s+"\n}\n")' && node --check /tmp/c.mjs`
- `meta` stays a pure literal (the Workflow tool extracts it statically).
- **The guards that police your change pass.** Run the relevant guard(s) with the
  glob form (the runner does not recurse a bare directory):
  `node --test '_meta/tools/test/**/*.test.js'` for everything, or target the ones
  your change touches, e.g.
  `node --test _meta/tools/test/backend-axis-ssot-guard.test.js`. If a
  `*-ssot-*guard` fails, do NOT edit the workflow copy to match — edit the
  `_meta/scaffolding/` SSOT and run the codemod named in the failure message
  (see rule 1). (Note: the full suite has some pre-existing unrelated failures;
  the bar for your PR is that *the guards covering your change* are green and you
  introduce no new failure.)
- **No `WORKFLOW_SUITABILITY` / `assertWorkflowSuitability()`** in the `.js`
  (rule 2); eligibility is in `manifest.yaml` `routing.accepts`.
- Changelog updated in **both** languages (and `VERSION` bumped if releasing).

## 中文备忘

添加或修改 kernel optimization workflow 时,遵循 [Agent.md](Agent.md)。**运行时与一致性硬规则见上文
"Workflow code: the non-negotiable rules"**:共享 helper 只在 `_meta/scaffolding/` 单一来源,
sentinel 之间的代码是生成的,改动要改 SSOT 再跑 `node scripts/patch-<X>.js --refresh`,
禁止手工编辑内联副本;资格判断走 manifest `routing.accepts`,不要写 `WORKFLOW_SUITABILITY` /
`assertWorkflowSuitability`;禁止 `import` / `Date.now` / `Math.random`,`agent()` 必须用
`agentRetry` 包裹,substrate 用 `--artifact/--problem/--out`,写文件到 `args.exp_dir`。版本与
changelog 规范:遵循 SemVer,版本号写在 `VERSION`,每次改动都要在 `CHANGELOG.md` 与
`CHANGELOG.zh-CN.md` 的 `## [Unreleased]` 下同步记录(双语)。
