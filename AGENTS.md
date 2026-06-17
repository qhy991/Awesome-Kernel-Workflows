# AGENTS.md — working rules for agents in this repo

Instructions for any AI agent (Claude Code, Codex, …) changing
Awesome-Kernel-Workflows (AKW). Human contributors should follow them too.

When adding or modifying a kernel-optimization workflow, follow
[Agent.md](Agent.md) — the workflow maintenance guide. This file adds the
**versioning & changelog policy** (the common `AGENTS.md` filename serves as the
entry point).

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
- Changelog updated in **both** languages (and `VERSION` bumped if releasing).

## 中文备忘

添加或修改 kernel optimization workflow 时,遵循 [Agent.md](Agent.md)。版本与
changelog 规范见上:遵循 SemVer,版本号写在 `VERSION`,每次改动都要在
`CHANGELOG.md` 与 `CHANGELOG.zh-CN.md` 的 `## [Unreleased]` 下同步记录(双语)。
