# Solver Best-of-Breed Audit (26 solvers)

Evidence base for [`SOLVER-SDK.md`](./SOLVER-SDK.md). Each solver's `.js` was read
in full (not just headers). Complements
[`../docs/workflow-fidelity-audit-2026-06-03.md`](../docs/workflow-fidelity-audit-2026-06-03.md)
(which judges *source fidelity*); this doc judges *reusable strength* — what a
best-of-breed solver should steal, and whether the mechanism is executable or
prompt-only.

## Core finding: 7 orthogonal axes + 1 exclusive axis

The 26 solvers are not 26 parallel methods; they are points on a small number of
axes. "集各家所长" = pick the strongest *executable* component on each axis.

| # | Axis | Composable? | Best-of-breed | Borrow this | Fidelity |
|---|------|-------------|---------------|-------------|----------|
| 1 | Diagnosis (measurement → direction) | ✅ stack | **ARGUS** / KEET / cuPilot | tag-function assertions giving concrete thread/element **counterexamples**; KEET hypothesis-first | ARGUS needs real checker; KEET prompt-only ⚠️ |
| 2 | Persistent memory (cross-run compounding) | ✅ | **KernelBlaster** | state-keyed cross-run DB, confidence blended with measured speedup, decays | real ✅ |
| 3 | Failure memory (avoid re-treading) | ✅ | **AKO4X** + AdaExplore | dead-ends with **WHY + revalidate_if** (not blind ban); TRAPS.md; "You cannot…" rules scored by occurrence | real ✅ |
| 4 | Anti-cheat / validity gate | ✅ | **KernelSkill** + CUDAAgent + AKO4X | deterministic machine_check → allowed_methods; robust reward (beat eager AND compile); pre-commit Expected before bench | deterministic ✅ |
| 5 | Cost control (save GPU) | ✅ | **GPUForecasters** + KernelBand + CutlassGEMM | hardware masking (skip saturated-resource strategies); ceiling detection → library fallback; abstention skips GPU | real ✅; forecaster heavy ⚠️ |
| 6 | Generation quality (plan→code fidelity) | ✅ | **STARK** + cuPilot + KernelAgent | grounded anchors `<<<IMPROVE BEGINS/ENDS>>>`; strategy as intermediate semantic layer; temperature-diverse parallel seeds | semi prompt-mediated |
| 7 | Attribution (did the change actually help) | ✅ | **FACT** | leave-one-out **ablation** measuring each pattern's real delta before crediting | real ✅ |
| 8 | **Search topology** | ❌ **exclusive — pick ONE** | (depends on kernel family) | beam / MCTS / evolutionary / bandit / graph — **mutually exclusive**; heterogeneity belongs to KerSor's portfolio, not inside one solver | n/a |

The axis-8 insight is load-bearing: stacking topologies in one solver is a category
error. It is also why the KerSor portfolio + transfer object exist.

---

## Per-solver digests (grounded in full `.js` reads)

Schema: core_loop · authoritative_feedback (executable vs prompt) · distinctive_strength · memory_state · borrow_this · fidelity_risk.

### Iterative / self-improving

**AccelOpt** — Setup(NCU baseline)→Plan→Execute(multi-sample)→Evaluate(NCU)→Learn(threshold-filtered slow/fast pairs)→Iterate · **feedback:** NCU metrics, executable · **strength:** dense NCU-informed planning, not generic advice · **memory:** experienceMemory + lastIterNewPatterns · **borrow:** threshold-filtered experience sampling (only add patterns with speedup>threshold) + random pool for diversity · **fidelity:** executable (real ncu; static fallback).

**AKO4X** — Setup→Round-Init(cross-round reflection)→Iterate(smoke→bench→log)→Archive(pre-archive gates)→Retrospect→Report · **feedback:** benchmark score after smoke-test correctness gate; optional NCU · **strength:** two-layer (round+inner) loop, pre-commit Expected hypothesis before bench, TRAPS.md silent-bug accrual · **memory:** experienceMemory (narrow+broad WHEN), deadEnds (WHY priors), traps · **borrow:** dead-ends as re-verifiable expectation priors (WHY, not prohibition) · **fidelity:** executable; pre-archive gates catch silent-skip / library delegation.

**KDA** — Inspect→Plan(draft+executable plan)→Implement(one candidate)→Validate→Decide(promote/revise/reject)→Report · **feedback:** validation+evaluation commands, measured latency, promotion gate (speedup≥1.2× + pass) · **strength:** explicit promotion criteria; draft-first; parent-linked forensic lineage · **memory:** candidates.jsonl (id, parent_id, status, metrics, reason) · **borrow:** draft-before-code protocol (rank directions by value/risk first) · **fidelity:** executable; static fallback.

**CUDAAgent** — Setup→Profile→Implement(kernel+bindings+model_new.py)→Verify(compile+correct+bench)→Refine→Report · **feedback:** compile/correct/speedup, executable; robust reward r∈{-1,2,3} · **strength:** full pybind11+model_new.py integration; reward balances correctness vs perf · **memory:** history(turn/action/outcome/speedup/error) · **borrow:** robust reward schedule (r=3 beats eager AND compile, r=2 eager only) — prevents single-baseline overfit · **fidelity:** executable; inference-time adaptation (no RL training).

**ARGUS** — Setup→Plan(ICRL + data-flow invariants)→Select→Lower(tag functions)→Validate(invariant check+tests+profile)→Learn · **feedback:** invariant violations (concrete counterexamples) + compile + tests + TFLOPS · **strength:** dense feedback via tag-function assertions instead of sparse pass/fail · **memory:** optimizationHistory, invariantViolationLog, candidateBeam(top-3) · **borrow:** `tag(A)==tag(B)` assertions failing with concrete thread/element counterexamples · **fidelity:** executable ONLY if invariant_check_command provided; else "missing_invariant_evidence" ⚠️.

**CUDALLM** — Setup→FeatureCatalog→GenerateTests→SelectFeatures(explore/exploit)→GenerateKernel→Evaluate→Reinforce→Report · **feedback:** compile/correct/speedup via eval_command JSON; no credit without measured evidence · **strength:** feature-level scoring; flags reward-hacking (hardcoded shapes, skipped compute, torch fallback) · **memory:** featureScores (per-feature attempts/compiled/correct/reward/best) · **borrow:** feature-level reinforcement (unimplemented features get no credit; unsafe flagged upfront) · **fidelity:** executable; workflow adaptation (no training).

### Tree / MCTS / graph

**KSearch** — Setup→init world model→(Select frontier→Generate/Improve→Evaluate→Refine/Backtrack)*→Report · **feedback:** real compile+correct+bench drives refine/backtrack · **strength:** co-evolving decision-tree world model guiding action selection + adaptive backtrack · **memory:** JSON tree (open/solved/failed nodes, scores, solutions) · **borrow:** dual stagnation detection (no-improvement streak + not-beating-parent streak) · **fidelity:** tree real; refine/backtrack prompts heuristic.

**AdaExplore** — Setup→MCTS root→(Select UCB1→Expand large/small→Evaluate→Backprop→AdaptMemory)*→Report · **feedback:** real evaluator; memory updates only from evaluated failure logs · **strength:** failure-driven "You cannot…" skill memory scored by occurrence + diversity-preserving MCTS · **memory:** skill rules {rule,score}; MCTS visits/reward · **borrow:** max/avg reward blend (REWARD_ALPHA) for UCB1 + failure-log→rule synthesis · **fidelity:** evaluator executable; skill extraction heuristic.

**STARK** — Setup→ref kernel→(ε-greedy Select→Plan w/ anchors OR Debug→Code→Evaluate→Update tree)*→Report · **feedback:** real compile/correct/perf; debug path when failing · **strength:** grounded instruction anchors bridging plan→code; role-specific context windows · **memory:** tree nodes (parent_id, code, plan, anchors, runtime, correct) + leaderboard · **borrow:** `<<<IMPROVE BEGINS/ENDS>>>` anchors tying generation to planner intent · **fidelity:** eval real; anchor realization prompt-mediated.

**ReGraphT** — Setup→BuildGraph→(Select path via MCGS→Generate from method sequence+examples→Evaluate→UpdateGraph)*→Report · **feedback:** evaluator JSON (compiled/correct/speedup); reward backprop only on real measurement · **strength:** CUDA reasoning graph + example-conditioned generation + Monte Carlo Graph Search · **memory:** graph nodes (method, visits, reward, examples), edge priors · **borrow:** graph-node examples as generation conditioning + edge priors as method-suitability · **fidelity:** eval real; graph construction heuristic.

### Evolutionary / bandit / search

**KernelFoundry** — Setup→Select→Vary→Evaluate→Insert→Evolve-Prompts (MAP-Elites) · **feedback:** compile+correct+bench, executable · **strength:** MAP-Elites 4×4×4 behavioral descriptors + gradient-informed parent selection (∇F+∇R+∇E) · **memory:** 4×4×4 elite archive + transition history + evolved meta-prompts · **borrow:** gradient-informed selection combining 3 orthogonal signals for coverage+convergence · **fidelity:** descriptors classified by static pattern match (not profiler); meta-prompt edits LLM-instructed ⚠️.

**KernelFoundryDx** (distinct method, shares only the name) — Setup→Init(expert-RAG seeds+anti-cheat)→Evolve(per-island role mutation)→Evaluate(lightweight compile+run)→Diagnose(failure/bottleneck class)→Evolve-Pop(island update + hint reinforce)→Report · **feedback:** real compile+execute (no ncu); anti-cheat gates init AND evolution · **strength:** multi-island role specialization (fusion/memory/param-tuning/restructuring) + diagnosis-keyed hint library reinforced by measured speedup · **memory:** island populations + elite archives + shared hint library (trigger, bottleneck_class, success_count, avg_speedup) · **borrow:** diagnosis-driven hint library reinforced/down-weighted by measured outcomes · **fidelity:** bottleneck class inferred coarsely from runtime stats (not profiler) ⚠️; hints grounded in measured speedups.

**cuPilot** — Setup(gen + roofline class)→Epochs[Strategize(SCE crossover at strategy level)→Translate→Revise(compile→func→NCU→fix)→Evolve(tournament+elitism+alignment)] · **feedback:** nvcc + function check + NCU (SM/mem/L2), executable, requires NCU · **strength:** strategy as intermediate semantic representation decoupling crossover from synthesis; roofline-guided prompting · **memory:** population (kernel, strategy, fitness, hwUtil) + growing strategy pool + roofline class · **borrow:** roofline positioning as semantic guide (compute-bound→SM throughput; memory-bound→bandwidth/L2) · **fidelity:** roofline+NCU executable; strategy→code translation LLM-guided.

**KernelBand** — Setup→{Cluster(K-means on φ)→Select(masked UCB over (cluster,strategy))→Generate→Evaluate→Update}×T · **feedback:** real compile+execute + φ(k)=[T̄,n_reg,n_smem,d_block,η_occ]; hardware mask from NCU saturation; reward=(T−T')/T · **strength:** MAB with hardware-aware pruning — mask removes (cluster,strategy) when target resource >75% saturated · **memory:** candidate pool w/ φ + hw signatures + clusters; bandit stats {N,μ̂}; masked pool · **borrow:** hardware-aware action masking (strategy only where its target resource is below saturation) · **fidelity:** all signals real; Lipschitz clustering is an unvalidated theoretical assumption.

### Pipeline / FSM / multi-agent

**KEET** — Source Inspection(hypotheses from code)→Profile Inspection→Aggregation→Review(confirm/refute vs data) · **feedback:** NCU data; verdicts prompt-mediated · **strength:** hypothesis-first (predict from source BEFORE profile) to prevent confirmation bias · **memory:** algorithm_summary, performance_hypotheses, profileAnalyses, hypothesis_verdicts · **borrow:** hypothesis-first methodology template (predict upfront, verify in separate phase) · **fidelity:** prompt-mediated throughout ⚠️ (explanation tool, not optimizer).

**KernelAgent** — Setup→Route(static complexity → direct vs pipeline)→Generate(parallel seeds, temp variation)→Verify(sandboxed)→Refine→Compose(stitch subgraphs)→Report · **feedback:** sandboxed test harness (exit code), disallowed-pattern checks · **strength:** parallel diversity seeds + strict verify-before-refine + auto-route single vs multi-subgraph · **memory:** candidates, verifiedKernels, refinementHistory, subgraphs · **borrow:** parallel seed + iterative repair with aggressive policy enforcement · **fidelity:** executable (real Python validation).

**TritorX** — Setup→per-operator: Generate→Lint→Compile-Test(JIT+OpInfo)→[fail: Debug→Lint]→Success/Failure · **feedback:** custom Triton linter + JIT compile + OpInfo multi-dtype (20k+ tests) · **strength:** FSM with in-context error feedback; linter prevents PyTorch-fallback cheating; extreme coverage · **memory:** currentKernel, operatorsPassed/Failed, allResults · **borrow:** custom linter + FSM loop distilling compiler/runtime errors into feedback prompts · **fidelity:** executable when LINT+COMPILE+TEST cmds present.

**Astra** — Setup→PrepareTests→ProfileBaseline→[N×: Plan→Code→Evaluate→Record]→PostProcess→Report · **feedback:** JSON contract (compiled/correct/speedup/runtime_ms); accept only from executable harness · **strength:** evidence-driven planning grounded in measured bottlenecks + production-safe constraints (preserve export, prefer loop/memory opts) · **memory:** currentBestCode, runLog, baselineProfile, lessons · **borrow:** lexicographic score (compile×correct×speedup) + lessons accumulation forcing learn-from-measured · **fidelity:** executable; measured=false if harness missing.

### Newer additions

**CutlassGEMM** — NCU-guided GEMM tuning with ceiling detection + cuBLAS hybrid · Analyze→Baseline(4-way dispatch)→one-shot NCU→tile tuning(actionable M only)→ceiling gate→hybrid fallback→MFU report · **feedback:** SOL-ExecBench + NCU, executable · **borrow:** ceiling detection — when M<threshold shows flat latency, switch to cuBLAS instead of tuning (applies to any overhead-bound kernel) · **fidelity:** real.

**FACT** — compositional synthesis: discover patterns from exemplars → realize as CUTLASS templates → compose → ablate · Setup→Pattern discovery→Realization(templates+constraints)→Composition(greedy, dependency-aware)→Ablation(leave-one-out)→Evaluation→Report · **feedback:** CUTLASS compile+correct + GFLOPS; ablation isolates each pattern's measured contribution · **borrow:** discovery→realization→composition→ablation; separates mechanism from measured payoff · **fidelity:** real.

**GPUForecasters** — learned speedup forecasting + abstention + PUCT search · Setup→Train forecasters(sample+fit+abstention calibration)→PUCT(forecaster priors; execute when abstain)→Refinement→Validation→Report · **feedback:** real GPU exec + forecaster MAE + abstention calibration · **borrow:** abstention-guided PUCT — trust forecaster when confident (skip GPU), execute when uncertain → cuts tree-search cost by orders of magnitude · **fidelity:** mixed (forecaster predicts ⚠️; final validation real).

**KernelBlaster** — continual CUDA opt via memory-augmented in-context RL (MAIC-RL) · Setup(load/seed DB)→rollout: ProfileState(NCU→class)→Retrieve(rank from DB)→Plan→Execute→Evaluate(re-profile cycles)→Reward(update DB)→Iterate(policy-update over replay buffer) · **feedback:** NCU elapsed cycles; reward = improvement/100 + accuracy_bonus ± penalty · **borrow:** state-keyed cross-run DB (confidence_score + usage_count + measured actual_speedup, decay/growth) — compounds across rollouts AND kernels · **fidelity:** real (NCU-grounded; DB mutations backed by measurement).

**KernelSkill** — multi-agent dual-memory (long-term skill library + short-term trajectory) · Setup→Seed(3 candidates, best valid)→round: Evaluate(ncu+nsys)→Repair(if invalid, chained memory) OR Optimize(feature extractor→deterministic gate→retrieve skill→plan→optimize)→backfill→Report · **feedback:** PyTorch reference latency + ncu/nsys + correctness within tol · **borrow:** two-layer skill library with **deterministic gate** (field_mapping→derived→headroom tier→decision table→allowed_methods); non-binding llm_assist only AFTER gating · **fidelity:** real/deterministic gate.

**StitchCUDA** — three-agent (Planner-Coder-Verifier) + adaptive replanning · Setup→attempt: (Replan if triggered)→Plan→Code→Verify(compile+correct+perf)→track history; replan on stagnation/failures · **feedback:** nvcc + correctness + GFLOPS · **borrow:** replanning triggers (2 consecutive failures OR 3-iter CV<5% stagnation) → generate alternative_approach, reset counters · **fidelity:** real.

**Xe-Forge** — Intel XPU (Triton/SYCL) multi-stage CoVeR · Setup→initial impl→cycle: Analyze(profile+bottleneck)→Plan(top-3)→Optimize→Verify→Refine(continue if >5%) · **feedback:** Intel XPU profiler (VTune/onemkl) GFLOPS/BW/EU-occupancy + correctness · **borrow:** CoVeR cycle + early-termination gate (<5% improvement or resource limit → stop) keeping iteration count low · **fidelity:** real.

---

## Cross-cutting patterns

1. **Executable feedback is the dividing line.** Strict solvers own a real evidence
   source (NCU, benchmark, simulator, linter+OpInfo); weak ones encode the
   load-bearing signal as a prompt (KEET verdicts, KernelFoundry descriptors,
   coarse bottleneck classification). Borrow a mechanism **only with its evidence owner.**
2. **Persistent, state-keyed memory is the rarest and highest-value component**
   (only KernelBlaster does it well) — the main source of cross-kernel compounding.
3. **Deterministic gating beats LLM discretion** (KernelSkill gate, AKO4X pre-archive
   gates, anti-cheat validators) — the same lesson KerSor applies at the routing layer.
4. **Cost-control gates are underused** but cheap and high-leverage (ceiling,
   masking, abstention, stagnation/replanning).
5. **Search topology is the one true differentiator** — keep it per-solver; let the
   portfolio carry heterogeneity.
