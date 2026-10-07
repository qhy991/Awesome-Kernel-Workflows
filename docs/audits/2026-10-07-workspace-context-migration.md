# File-backed workflow context migration

## Implemented scope

This follow-up migrates STARK, AdaExplore, ReGraphT, KernelFoundryDx, KernelAgent, cuPilot, AKO4X, KernelBlaster, KernelSkill, Xe-Forge, GemmPTX and HarnessEngineering. The previous five-method task-mode repairs remain in place. All35 workflows receive the same pure helper definitions via the existing typed-args generator; unrelated methods do not acquire additional agent calls.

Candidate code is retained in its existing runtime record for compatibility, but repeated source context uses its declared file. STARK and ReGraphT evaluator activations persist exact source via a CPU-only command before evaluation; AdaExplore reuses its existing per-step files and fixes the Sol path alias. Refined KernelAgent candidates get distinct round paths. Model-producing activations in cuPilot/KernelSkill persist their final output as part of the same activation. Failure diagnostics and graph examples are retained as complete files, not discarded substrings. Reward/selection scalars stay in prompt indices; unknown values do not become zero. Search counts, feature pools and update rules are unchanged.

The source persistence helper streams a JSON string through stdin (not a giant argv field), preserving UTF8, quotes, shell characters and newlines. It confines writes to exp_dir, refuses a symlink target, creates absent files exclusively, permits identical-content replay, and refuses a conflicting existing file. It never deletes the old candidate or chooses a similarly named file. A legacy record without an authoritative path retains its complete fallback; no path is invented. Instructions require explicit reporting of missing files. Source files, result JSON and graph files are not interchangeable semantic roles.

## Verification

Real filesystem tests execute the generated command with large Unicode/quoted/multiline inputs, identical replay, identity collision, outside-root and symlink negatives. Method-level tests run actual script bodies and execute the emitted source commands for STARK/AdaExplore/ReGraphT, then verify that downstream contexts reference the preserved files. cuPilot/Xe-Forge/KernelBlaster tests exercise the revised cross-iteration path handoff. Shared scaffold guards enforce generated helper identity.

Lean4.19.0 compiled `docs/formal/WorkspaceHandoff.lean` successfully with8 theorem proofs, no sorry or custom axiom. The abstract model proves collision rejection, same-content idempotence, exact fresh content, noninterference for other keys, reference identity, retained access to bound full artifacts, source-length-independent indices and distinct-key nonaliasing.

The Lean model assumes unique caller-owned keys and a functioning store; it is NOT a proof of the JavaScript/Python implementation, filesystem race freedom, model obedience, GPU correctness, latency, or WSR efficacy. Executable tests cover concrete persistence and prompt assembly. Native/GPU qualification of the migrated methods is still required before claiming production task success. Historical/frozen runs remain unchanged.

Full regression comparison against the exact prior2cb394d checkout retains pre-existing failures (including old golden prompt snapshots). Do not regenerate golden files from the changed workflow to manufacture a passing check. Updated transport tests assert complete-source file access and executable materialization rather than requiring inline source syntax.

## Remaining boundary

This removes the identified repeated-source/history assembly pattern, not the provider's context limit. A single newly generated source, task definition or method-specific strategy state can itself be large. Initial source materialization is necessary where no file exists. This change imposes no character cap and does not silently remove required task/skill inputs. Existing final compatibility fields may still contain full source; callers should prefer best_kernel_path/graph_path when constructing another prompt.

Verification receipt:104 focused persistence/method/task/scaffolding tests passed;35 Host parses passed;Lean4.19.0 accepted8 theorems. KernelSkill cache reuse was additionally checked: duplicate source content carries the original materialized path, not a newly invented round path. Full-suite known failures are recorded separately from focused success.

Final full suite:765 tests,695 passed,70 failed. Exact prior2cb394d baseline:755 tests,684 passed,71 failed. No new failing test names; the obsolete inline-parent assertion was replaced by an authoritative-path assertion. Existing golden/legacy failures were not hidden by regenerating snapshots.
