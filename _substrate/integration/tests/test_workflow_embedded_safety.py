#!/usr/bin/env python3
"""Bug-class guard for the integration-strategist rollout (step 1 of the rollout plan).

Static scan of every workflow .js. Catches the three structural bug-classes that the
Generalist review found, so the rollout to the other ~23 workflows cannot silently
re-introduce them. The check is heuristic (regex over source) — it is a SAFETY NET,
not a proof; but it catches the known shapes.

Rules (applied only to workflows that HAVE the relevant wiring):

1. PARALLEL EMBEDDED-EVAL RACE: a workflow that wires embedded eval
   (integrationEvalBlock / __embeddedEvalPlan / embedded_inplace / embedded_dispatch)
   AND evaluates candidates with `await parallel(` MUST gate embedded eval behind a
   serial path (IS_EMBEDDED / a serial for-loop over candidates) — embedded modes
   mutate a shared file (inplace) or share a project build (dispatch) and cannot run
   concurrently. (Generalist: IS_EMBEDDED serial loop ✓.)

2. EMBEDDED_INPLACE WITHOUT RESTORE SAFETY NET: a workflow with an embedded_inplace
   path MUST back up the original once (ORIGINAL_BACKUP / integ_original) and restore
   on exit. (Generalist: ORIGINAL_BACKUP + exit restore ✓.) embedded_dispatch is
   non-mutating (adapter unregister is the reversibility) so it is exempt.

3. PROFILE IGNORES PROFILING_DECISION: a workflow that resolves PROFILING_DECISION
   MUST branch the Profile on PROFILING_DECISION.method (not a hardcoded
   ncu_command-only path) — otherwise, when the strategist routes to perf_heuristic,
   the Profile still tries ncu (A-O1). (Generalist: branches on method + native→perf
   downgrade ✓.)

Currently only Generalist has the full wiring; it passes. The 4 workflows with the
inlined __embeddedEvalPlan substrate (ARGUS/CUDAAgent/FACT/GPUForecasters) use
embedded_dispatch with no parallel candidate eval, so they are safe and pass. This
test is the gate the rollout must keep green as each new workflow is wired.
"""
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # _substrate/integration/tests/ -> AKW repo root

EMBEDDED_EVAL = re.compile(
    r'integrationEvalBlock|__embeddedEvalPlan|embeddedInplace|embedded_inplace|embeddedDispatch'
)
INPLACE = re.compile(r'embedded_inplace|embeddedInplace')
PARALLEL_EVAL = re.compile(r'await parallel\(')
SERIAL_GATE = re.compile(
    r'IS_EMBEDDED|method\s*!==\s*[\'"]standalone[\'"]|'
    r'for\s*\(\s*(?:let|const)\s+i\s*=\s*0\s*;\s*i\s*<\s*plans'
)
ORIGINAL_BACKUP = re.compile(r'ORIGINAL_BACKUP|integ_original')
RESTORE = re.compile(
    r'restore pristine|exit[- ]restore|revertToBackup|'
    r'cp\s+-a\s+.*backup.*(?:kernelPath|KERNEL_PATH)|ALWAYS restore'
)
PROFILING_DECISION_DECL = re.compile(r'let\s+PROFILING_DECISION|PROFILING_DECISION\s*=\s*\{')
PROFILING_DECISION_BRANCH = re.compile(r'PROFILING_DECISION\.method')


def _workflow_files():
    out = []
    for d in sorted(REPO_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith('_') or d.name in ('docs', 'paper', 'ppt', 'scripts', 'outputs', 'badges'):
            continue
        for js in sorted(d.glob("*.js")):
            out.append(js)
    return out


def _check_source(src, wf):
    """Return list of (workflow, rule, detail) for safety-rule violations in one source."""
    f = []
    has_emb = bool(EMBEDDED_EVAL.search(src))
    has_inplace = bool(INPLACE.search(src))
    has_parallel = bool(PARALLEL_EVAL.search(src))
    has_serial = bool(SERIAL_GATE.search(src))
    has_backup = bool(ORIGINAL_BACKUP.search(src))
    has_restore = bool(RESTORE.search(src))
    has_pd_decl = bool(PROFILING_DECISION_DECL.search(src))
    has_pd_branch = bool(PROFILING_DECISION_BRANCH.search(src))

    # Rule 1: parallel + embedded eval requires a serial gate.
    if has_emb and has_parallel and not has_serial:
        f.append((wf, "parallel-embedded-race",
                  "embedded eval wired + `await parallel(` candidate eval but no serial gate "
                  "(IS_EMBEDDED / serial for-loop). Embedded modes cannot run concurrently."))

    # Rule 2: embedded_inplace requires original-backup + restore.
    if has_inplace and not (has_backup and has_restore):
        missing = []
        if not has_backup: missing.append("ORIGINAL_BACKUP (single original backup)")
        if not has_restore: missing.append("exit/forced restore")
        f.append((wf, "inplace-no-restore",
                  f"embedded_inplace path without {' and '.join(missing)}."))

    # Rule 3: PROFILING_DECISION resolved must be branched on in Profile.
    if has_pd_decl and not has_pd_branch:
        f.append((wf, "profile-ignores-strategist",
                  "PROFILING_DECISION resolved but never branched on (.method) — Profile "
                  "ignores the strategist (A-O1)."))
    return f


def _findings():
    """Return list of (workflow, rule, detail) for any safety-rule violation."""
    f = []
    for js in _workflow_files():
        f.extend(_check_source(js.read_text(errors='replace'), js.parent.name))
    return f


class WorkflowEmbeddedSafetyGuard(unittest.TestCase):
    def test_no_embedded_safety_violations(self):
        findings = _findings()
        if findings:
            detail = "\n".join(f"  - {wf}: [{rule}] {detail}" for wf, rule, detail in findings)
            self.fail(
                f"{len(findings)} embedded/profiling safety violation(s) found across workflows "
                f"(these are the Generalist bug-classes — fix before relying on the embedded path):\n{detail}"
            )

    def test_guard_self_consistency_generalist_passes(self):
        # The reference (Generalist) MUST pass — it has the full wiring + all safety nets.
        gen = [f for f in _findings() if f[0] == 'Generalist']
        self.assertEqual(gen, [], f"Generalist (the rollout reference) must pass the guard: {gen}")

    def test_guard_actually_detects_violations(self):
        # Prove the guard is not toothless: a synthetic source with each bug-class MUST be flagged.
        race = "const integ = integrationEvalBlock(...)\nawait parallel(plans.map(...))\n"  # no serial gate
        self.assertTrue(any(r == 'parallel-embedded-race' for _, r, _ in _check_source(race, 'synth')))
        inplace = "embedded_inplace eval block\n"  # no ORIGINAL_BACKUP, no restore
        self.assertTrue(any(r == 'inplace-no-restore' for _, r, _ in _check_source(inplace, 'synth')))
        pd = "let PROFILING_DECISION = {method:'native_profiler'}\n"  # resolved but never branched
        self.assertTrue(any(r == 'profile-ignores-strategist' for _, r, _ in _check_source(pd, 'synth')))
        # And a clean Generalist-shaped source passes all three.
        clean = ("integrationEvalBlock IS_EMBEDDED for (let i = 0; i < plans.length; i++) {} "
                 "embedded_inplace ORIGINAL_BACKUP restore pristine "
                 "let PROFILING_DECISION = {} PROFILING_DECISION.method === 'native_profiler'")
        self.assertEqual(_check_source(clean, 'synth'), [])


if __name__ == "__main__":
    unittest.main()
