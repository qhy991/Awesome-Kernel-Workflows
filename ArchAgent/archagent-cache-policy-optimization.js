export const meta = {
  name: 'archagent-cache-policy-optimization',
  description: 'Evolutionary search for cache replacement policies using AlphaEvolve',
  whenToUse: 'Use for discovering novel cache replacement policies through evolutionary search',
  phases: [
    { title: 'Setup', detail: 'Initialize ChampSim environment and evolutionary parameters' },
    { title: 'Initialize Population', detail: 'Generate initial candidate policies' },
    { title: 'Short Evaluation', detail: 'Quick fitness evaluation on subset of traces' },
    { title: 'Evolution', detail: 'Apply mutations and crossover operations' },
    { title: 'Island Migration', detail: 'Exchange elite candidates between islands' },
    { title: 'Long Evaluation', detail: 'Comprehensive evaluation of best candidates' },
    { title: 'Validation', detail: 'Final validation and anti-cheating checks' },
    { title: 'Report', detail: 'Generate optimization report' },
  ],
};

// --- BEGIN genome-report (auto-inserted by scripts/patch-genome-report.js) ---
// Self-reported, work-plane (forgeable) stage trace for observability + the
// recombiner. NOT a trust anchor — see _meta/genome-trajectory-schema.md.
async function __genomeReport(phaseName, wfName) {
  try {
    const __dir = (typeof args !== 'undefined' && args && args.exp_dir) ? args.exp_dir : '.'
    await agent(
      'Append exactly one line to ' + __dir + '/genome.jsonl (create it if missing; use a shell append: printf %s\\n ... >> file). ' +
      'The line must be this JSON on ONE line: {"workflow":"' + wfName + '","phase":"' + phaseName + '","ts":"<UTC>","status":"entered"}. ' +
      'Produce <UTC> by running: date -u +%Y-%m-%dT%H:%M:%SZ . Do nothing else; modify no other file. Echo the exact line you appended.',
      { label: 'genome:' + phaseName, phase: phaseName }
    )
  } catch (__e) { /* observability must never break the workflow */ }
}
// --- END genome-report ---

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cpp'],
  supported_problem_types: ['cache-policy-search'],
  problem_types: ['CPU cache replacement policy search', 'ChampSim-style cache policy evolution'],
  reason: 'ArchAgent optimizes C++ cache replacement policies with simulator IPC feedback; it is not a GPU kernel workflow.',
}

function normalizeSuitabilityValue(value) {
  const raw = String(value || '').trim().toLowerCase().replace(/_/g, '-')
  const aliases = {
    'c++': 'cpp',
    cxx: 'cpp',
    cplusplus: 'cpp',
    cute: 'cute-dsl',
    hip: 'rocm',
    'intel-xpu': 'xpu',
    optimize: 'kernel-optimization',
    optimization: 'kernel-optimization',
    generate: 'kernel-generation',
    generation: 'kernel-generation',
    explain: 'performance-explanation',
    explanation: 'performance-explanation',
  }
  return aliases[raw] || raw
}

function supportsSuitabilityValue(supported, requested) {
  return supported.includes(requested) || supported.some(value => value.endsWith(`-${requested}`))
}

function assertWorkflowSuitability() {
  const requestedLanguage = normalizeSuitabilityValue(args.language)
  if (requestedLanguage && requestedLanguage !== 'auto') {
    const supported = WORKFLOW_SUITABILITY.supported_languages.map(normalizeSuitabilityValue)
    if (!supported.includes(requestedLanguage)) {
      throw new Error(
        `${meta.name} is not suitable for language="${args.language}". ` +
        `Supported languages/backends: ${WORKFLOW_SUITABILITY.supported_languages.join(', ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }

  const requestedProblemType = normalizeSuitabilityValue(args.problem_type)
  if (requestedProblemType && requestedProblemType !== 'auto') {
    const supportedProblemTypes = (WORKFLOW_SUITABILITY.supported_problem_types || []).map(normalizeSuitabilityValue)
    if (supportedProblemTypes.length && !supportsSuitabilityValue(supportedProblemTypes, requestedProblemType)) {
      throw new Error(
        `${meta.name} is not suitable for problem_type="${args.problem_type}". ` +
        `Supported problem types: ${WORKFLOW_SUITABILITY.supported_problem_types.join(', ')}. ` +
        `Typical use cases: ${WORKFLOW_SUITABILITY.problem_types.join('; ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }
}

assertWorkflowSuitability()

// ArchAgent: Evolutionary search for cache replacement policies
// Based on arXiv:2602.22425 (Columbia University)
// Implements AlphaEvolve with island model + MAP-Elites diversity maintenance

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup'); await __genomeReport('Setup', meta.name);

  const setupResult = await agent(
    `Set up ArchAgent evolutionary search environment:

1. Validate ChampSim simulator availability
2. Prepare trace workloads (SPEC CPU2017, GAP)
3. Configure evolutionary parameters:
   - Population size per island
   - Number of islands
   - Mutation rate
   - Crossover rate
   - Elite retention
   - State budget (48KB max)
4. Set up LRU baseline for IPC comparison
5. Configure anti-cheating constraints:
   - Lines of code penalty
   - State budget enforcement
   - Write-bypassing detection

Return JSON:
{
  "champsim_available": true/false,
  "trace_workloads": ["workload1", "workload2", ...],
  "num_islands": <int>,
  "population_per_island": <int>,
  "max_generations": <int>,
  "mutation_rate": <float>,
  "crossover_rate": <float>,
  "elite_retention": <int>,
  "state_budget_kb": 48,
  "short_eval_traces": <int>,
  "long_eval_traces": <int>,
  "lru_baseline_ipc": <float>
}`,
    {
      label: 'Setup ArchAgent',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          champsim_available: { type: 'boolean' },
          trace_workloads: { type: 'array', items: { type: 'string' } },
          num_islands: { type: 'integer' },
          population_per_island: { type: 'integer' },
          max_generations: { type: 'integer' },
          mutation_rate: { type: 'number' },
          crossover_rate: { type: 'number' },
          elite_retention: { type: 'integer' },
          state_budget_kb: { type: 'integer' },
          short_eval_traces: { type: 'integer' },
          long_eval_traces: { type: 'integer' },
          lru_baseline_ipc: { type: 'number' },
        },
        required: ['champsim_available', 'trace_workloads', 'num_islands', 'population_per_island'],
      },
    }
  );

  if (!setupResult || !setupResult.champsim_available) {
    log('ChampSim not available or setup failed');
    return { success: false, reason: 'champsim_unavailable' };
  }

  log(`Evolution config: ${setupResult.num_islands} islands × ${setupResult.population_per_island} candidates, ${setupResult.max_generations || 50} generations`);

  // Initialize island populations
  const numIslands = setupResult.num_islands;
  const populationPerIsland = setupResult.population_per_island;
  const maxGenerations = setupResult.max_generations || 50;
  const eliteRetention = setupResult.elite_retention || 2;

  // Track evolution history
  const islands = [];
  const evolutionHistory = [];
  let globalBestCandidate = null;

  // ============================================================================
  // Phase 2: Initialize Population
  // ============================================================================
  phase('Initialize Population'); await __genomeReport('Initialize Population', meta.name);

  log(`Generating initial population for ${numIslands} islands...`);

  const initResults = await parallel(
    Array.from({ length: numIslands }, (_, islandIdx) =>
      () => agent(
        `Generate initial population for island ${islandIdx}:

Population size: ${populationPerIsland}
State budget: ${setupResult.state_budget_kb}KB
Target: L2C cache replacement policy (ChampSim C++ code)

Generate diverse initial candidates using seeding strategies:
1. Template-based (LRU variants, RRIP, DRRIP, SHiP)
2. Random initialization
3. Heuristic-based (recency + frequency hybrids)

Each candidate should:
- Be valid C++ code for ChampSim LLC replacement
- Implement required functions: initialize_replacement, find_victim, update_replacement
- Stay within state budget
- Be diverse from other candidates

Return JSON:
{
  "island_id": ${islandIdx},
  "candidates": [
    {
      "id": "i${islandIdx}_c0",
      "code": "C++ code",
      "description": "brief description",
      "seeding_strategy": "template|random|heuristic"
    },
    ...
  ]
}`,
        {
          label: `Init island ${islandIdx}`,
          phase: 'Initialize Population',
          schema: {
            type: 'object',
            properties: {
              island_id: { type: 'integer' },
              candidates: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    id: { type: 'string' },
                    code: { type: 'string' },
                    description: { type: 'string' },
                    seeding_strategy: { type: 'string' },
                  },
                  required: ['id', 'code', 'description'],
                },
              },
            },
            required: ['island_id', 'candidates'],
          },
        }
      )
    )
  );

  const validInitResults = initResults.filter(Boolean);
  if (validInitResults.length === 0) {
    log('Failed to initialize any islands');
    return { success: false, reason: 'initialization_failed' };
  }

  // Initialize island populations
  for (const initResult of validInitResults) {
    islands.push({
      id: initResult.island_id,
      population: initResult.candidates,
      generation: 0,
      bestFitness: -Infinity,
      bestCandidate: null,
    });
  }

  log(`Initialized ${islands.length} islands with ${islands.reduce((sum, isl) => sum + isl.population.length, 0)} total candidates`);

  // ============================================================================
  // Main Evolution Loop
  // ============================================================================

  for (let generation = 0; generation < maxGenerations; generation++) {
    log(`\n=== Generation ${generation + 1}/${maxGenerations} ===`);

    // ==========================================================================
    // Phase 3: Short Evaluation
    // ==========================================================================
    phase('Short Evaluation'); await __genomeReport('Short Evaluation', meta.name);

    log(`Evaluating all candidates on ${setupResult.short_eval_traces || 5} traces...`);

    // Flatten all candidates from all islands
    const allCandidates = islands.flatMap(island =>
      island.population.map(cand => ({ ...cand, island_id: island.id }))
    );

    const evalResults = await parallel(
      allCandidates.slice(0, 20).map(cand => // Limit parallelism to avoid overwhelming
        () => agent(
          `Evaluate cache replacement policy candidate:

Candidate ID: ${cand.id}
Island: ${cand.island_id}
Description: ${cand.description}

Code:
\`\`\`cpp
${cand.code.substring(0, 2000)}${cand.code.length > 2000 ? '\n... (truncated)' : ''}
\`\`\`

Evaluate on ChampSim:
1. Compile the replacement policy
2. Run on ${setupResult.short_eval_traces || 5} representative traces
3. Measure IPC (instructions per cycle)
4. Calculate geometric mean speedup vs LRU baseline (${setupResult.lru_baseline_ipc})
5. Check constraints:
   - State budget ≤ ${setupResult.state_budget_kb}KB
   - No write-bypassing cheats
   - Lines of code penalty (LOC factor)

Fitness function:
  base_fitness = geomean_ipc_speedup
  loc_penalty = max(0, (LOC - 100) / 100) * 0.1
  fitness = base_fitness - loc_penalty

Return JSON:
{
  "candidate_id": "${cand.id}",
  "compiled": true/false,
  "compile_errors": ["error1", ...],
  "ipc_speedup": <float or null>,
  "state_usage_kb": <float>,
  "lines_of_code": <int>,
  "fitness": <float or null>,
  "cheating_detected": true/false,
  "cheating_reason": "reason if detected"
}`,
          {
            label: `Eval ${cand.id}`,
            phase: 'Short Evaluation',
            schema: {
              type: 'object',
              properties: {
                candidate_id: { type: 'string' },
                compiled: { type: 'boolean' },
                compile_errors: { type: 'array', items: { type: 'string' } },
                ipc_speedup: { type: ['number', 'null'] },
                state_usage_kb: { type: 'number' },
                lines_of_code: { type: 'integer' },
                fitness: { type: ['number', 'null'] },
                cheating_detected: { type: 'boolean' },
                cheating_reason: { type: 'string' },
              },
              required: ['candidate_id', 'compiled', 'fitness', 'cheating_detected'],
            },
          }
        )
      )
    );

    // Process remaining candidates in batches
    const remainingCandidates = allCandidates.slice(20);
    for (let batchStart = 0; batchStart < remainingCandidates.length; batchStart += 20) {
      const batch = remainingCandidates.slice(batchStart, batchStart + 20);
      const batchResults = await parallel(
        batch.map(cand =>
          () => agent(
            `Evaluate cache replacement policy candidate ${cand.id} on ChampSim (short eval).

Code: ${cand.code.substring(0, 1000)}...

Return fitness, IPC speedup, constraints check.`,
            {
              label: `Eval ${cand.id}`,
              phase: 'Short Evaluation',
              schema: {
                type: 'object',
                properties: {
                  candidate_id: { type: 'string' },
                  compiled: { type: 'boolean' },
                  fitness: { type: ['number', 'null'] },
                  ipc_speedup: { type: ['number', 'null'] },
                  cheating_detected: { type: 'boolean' },
                },
                required: ['candidate_id', 'compiled', 'fitness'],
              },
            }
          )
        )
      );
      evalResults.push(...batchResults.filter(Boolean));
    }

    // Update island populations with fitness scores
    for (const island of islands) {
      for (const candidate of island.population) {
        const evalResult = evalResults.find(r => r && r.candidate_id === candidate.id);
        if (evalResult) {
          candidate.fitness = evalResult.fitness;
          candidate.ipc_speedup = evalResult.ipc_speedup;
          candidate.compiled = evalResult.compiled;
          candidate.cheating_detected = evalResult.cheating_detected;
        }
      }

      // Update island best
      const validCandidates = island.population.filter(c => c.fitness !== null && !c.cheating_detected);
      if (validCandidates.length > 0) {
        const islandBest = validCandidates.reduce((best, cand) =>
          cand.fitness > best.fitness ? cand : best
        );
        if (islandBest.fitness > island.bestFitness) {
          island.bestFitness = islandBest.fitness;
          island.bestCandidate = islandBest;
        }
      }
    }

    // Update global best
    for (const island of islands) {
      if (island.bestCandidate && (!globalBestCandidate || island.bestFitness > globalBestCandidate.fitness)) {
        globalBestCandidate = { ...island.bestCandidate, generation };
      }
    }

    log(`Generation ${generation + 1} best fitness: ${globalBestCandidate?.fitness?.toFixed(3) || 'N/A'}`);

    evolutionHistory.push({
      generation: generation + 1,
      best_fitness: globalBestCandidate?.fitness,
      best_speedup: globalBestCandidate?.ipc_speedup,
      island_best_fitnesses: islands.map(isl => isl.bestFitness),
    });

    // ==========================================================================
    // Phase 4: Evolution (Mutation & Crossover)
    // ==========================================================================
    phase('Evolution'); await __genomeReport('Evolution', meta.name);

    log('Applying evolutionary operators...');

    const evolutionResults = await parallel(
      islands.map(island =>
        () => agent(
          `Evolve population for island ${island.id} (generation ${generation + 1}):

Current population size: ${island.population.length}
Best fitness: ${island.bestFitness.toFixed(3)}
Elite retention: ${eliteRetention}

Evolutionary operators:
1. Selection: Tournament selection (k=3)
2. Mutation rate: ${setupResult.mutation_rate || 0.3}
   - Point mutations (change operators, constants)
   - Structural mutations (add/remove state, change algorithm)
   - LLM-guided mutations (multiple models: Gemini 2.5 Flash/Pro)
3. Crossover rate: ${setupResult.crossover_rate || 0.2}
   - Uniform crossover
   - Single-point crossover
4. Elite retention: Keep top ${eliteRetention} candidates unchanged

Current elites:
${island.population
  .filter(c => c.fitness !== null)
  .sort((a, b) => b.fitness - a.fitness)
  .slice(0, eliteRetention)
  .map(c => `  ${c.id}: fitness=${c.fitness.toFixed(3)}`)
  .join('\n')}

Generate next generation:
- Retain elites
- Apply selection, crossover, mutation to produce offspring
- Ensure diversity (avoid duplicates)

Return JSON:
{
  "island_id": ${island.id},
  "generation": ${generation + 1},
  "next_population": [
    {
      "id": "i${island.id}_g${generation + 1}_c0",
      "code": "C++ code",
      "description": "brief description",
      "parent_ids": ["parent1", "parent2"] or null,
      "operator": "elite|mutation|crossover"
    },
    ...
  ]
}`,
          {
            label: `Evolve island ${island.id}`,
            phase: 'Evolution',
            schema: {
              type: 'object',
              properties: {
                island_id: { type: 'integer' },
                generation: { type: 'integer' },
                next_population: {
                  type: 'array',
                  items: {
                    type: 'object',
                    properties: {
                      id: { type: 'string' },
                      code: { type: 'string' },
                      description: { type: 'string' },
                      parent_ids: { type: ['array', 'null'] },
                      operator: { type: 'string' },
                    },
                    required: ['id', 'code', 'description', 'operator'],
                  },
                },
              },
              required: ['island_id', 'next_population'],
            },
          }
        )
      )
    );

    // Update island populations
    for (const evolResult of evolutionResults.filter(Boolean)) {
      const island = islands.find(isl => isl.id === evolResult.island_id);
      if (island) {
        island.population = evolResult.next_population;
        island.generation = evolResult.generation;
      }
    }

    // ==========================================================================
    // Phase 5: Island Migration
    // ==========================================================================
    if (generation > 0 && generation % 5 === 0) {
      phase('Island Migration'); await __genomeReport('Island Migration', meta.name);

      log('Migrating elite candidates between islands...');

      // Collect elites from each island
      const migrationPool = islands.map(island => ({
        island_id: island.id,
        elite: island.bestCandidate,
      })).filter(entry => entry.elite);

      // Perform ring migration (island i sends elite to island (i+1) % numIslands)
      for (let i = 0; i < islands.length; i++) {
        const sourceIsland = islands[i];
        const targetIsland = islands[(i + 1) % islands.length];

        if (sourceIsland.bestCandidate) {
          // Clone elite to target island (with new ID)
          const migratedCandidate = {
            ...sourceIsland.bestCandidate,
            id: `i${targetIsland.id}_g${generation + 1}_migrated_from_i${sourceIsland.id}`,
          };

          // Replace worst candidate in target island
          const worstIdx = targetIsland.population.reduce(
            (worstIdx, cand, idx) =>
              (cand.fitness || -Infinity) < (targetIsland.population[worstIdx].fitness || -Infinity) ? idx : worstIdx,
            0
          );
          targetIsland.population[worstIdx] = migratedCandidate;
        }
      }

      log('Migration complete');
    }
  }

  // ============================================================================
  // Phase 6: Long Evaluation
  // ============================================================================
  phase('Long Evaluation'); await __genomeReport('Long Evaluation', meta.name);

  log('Performing comprehensive evaluation on top candidates...');

  // Select top 5 candidates across all islands
  const allFinalCandidates = islands.flatMap(isl => isl.population);
  const topCandidates = allFinalCandidates
    .filter(c => c.fitness !== null && !c.cheating_detected)
    .sort((a, b) => b.fitness - a.fitness)
    .slice(0, 5);

  const longEvalResults = await parallel(
    topCandidates.map(cand =>
      () => agent(
        `Long evaluation of top candidate ${cand.id}:

Code:
\`\`\`cpp
${cand.code}
\`\`\`

Perform comprehensive evaluation:
1. Run on full trace suite (${setupResult.long_eval_traces || 20} traces)
2. Measure per-trace IPC speedup vs LRU
3. Calculate geometric mean speedup
4. Verify constraints (state budget, no cheating)
5. Profile characteristics (access patterns, eviction behavior)

Return JSON:
{
  "candidate_id": "${cand.id}",
  "long_ipc_speedup": <float>,
  "per_trace_speedups": [<float>, ...],
  "geomean_speedup": <float>,
  "state_usage_kb": <float>,
  "verification_passed": true/false,
  "characteristics": "brief description of behavior"
}`,
        {
          label: `Long eval ${cand.id}`,
          phase: 'Long Evaluation',
          schema: {
            type: 'object',
            properties: {
              candidate_id: { type: 'string' },
              long_ipc_speedup: { type: 'number' },
              per_trace_speedups: { type: 'array', items: { type: 'number' } },
              geomean_speedup: { type: 'number' },
              state_usage_kb: { type: 'number' },
              verification_passed: { type: 'boolean' },
              characteristics: { type: 'string' },
            },
            required: ['candidate_id', 'geomean_speedup', 'verification_passed'],
          },
        }
      )
    )
  );

  const validLongEvals = longEvalResults.filter(Boolean);
  if (validLongEvals.length === 0) {
    log('Long evaluation failed for all candidates');
    return { success: false, reason: 'long_eval_failed' };
  }

  // Select best from long evaluation
  const finalBest = validLongEvals
    .filter(r => r.verification_passed)
    .reduce((best, curr) =>
      curr.geomean_speedup > best.geomean_speedup ? curr : best
    );

  log(`Final best candidate: ${finalBest.candidate_id} with ${finalBest.geomean_speedup.toFixed(3)}x speedup`);

  // ============================================================================
  // Phase 7: Validation
  // ============================================================================
  phase('Validation'); await __genomeReport('Validation', meta.name);

  const validationResult = await agent(
    `Perform final validation and anti-cheating checks:

Best candidate: ${finalBest.candidate_id}
Geomean speedup: ${finalBest.geomean_speedup.toFixed(3)}x

Anti-cheating checks:
1. Manual code review for hardcoded patterns
2. Verify no write-bypassing (detecting dirty blocks to avoid writebacks)
3. Check state budget compliance (${setupResult.state_budget_kb}KB)
4. Verify generalization (not overfitted to specific traces)
5. Test on held-out trace set

Return JSON:
{
  "candidate_id": "${finalBest.candidate_id}",
  "validation_passed": true/false,
  "cheating_checks": {
    "hardcoded_patterns": true/false,
    "write_bypassing": true/false,
    "state_budget_ok": true/false,
    "generalization_ok": true/false
  },
  "held_out_speedup": <float>,
  "final_verdict": "approved|rejected",
  "notes": "validation notes"
}`,
    {
      label: 'Final validation',
      phase: 'Validation',
      schema: {
        type: 'object',
        properties: {
          candidate_id: { type: 'string' },
          validation_passed: { type: 'boolean' },
          cheating_checks: { type: 'object' },
          held_out_speedup: { type: 'number' },
          final_verdict: { type: 'string' },
          notes: { type: 'string' },
        },
        required: ['candidate_id', 'validation_passed', 'final_verdict'],
      },
    }
  );

  if (!validationResult || validationResult.final_verdict !== 'approved') {
    log('Final validation failed');
    return {
      success: false,
      reason: 'validation_failed',
      notes: validationResult?.notes,
    };
  }

  // ============================================================================
  // Phase 8: Report
  // ============================================================================
  phase('Report'); await __genomeReport('Report', meta.name);

  const report = await agent(
    `Generate ArchAgent evolutionary search report:

Search summary:
- Islands: ${numIslands}
- Generations: ${maxGenerations}
- Total candidates evaluated: ~${maxGenerations * numIslands * populationPerIsland}
- Best candidate: ${finalBest.candidate_id}
- Final speedup: ${finalBest.geomean_speedup.toFixed(3)}x vs LRU baseline

Evolution history:
${evolutionHistory.slice(0, 10).map(h => `  Gen ${h.generation}: best=${h.best_fitness?.toFixed(3) || 'N/A'}`).join('\n')}
  ... (${evolutionHistory.length} total generations)

Generate report with:
1. Executive summary
2. Evolution trajectory visualization
3. Island diversity analysis
4. Best policy analysis and characteristics
5. Comparison with baselines (LRU, RRIP, SHiP)
6. Anti-cheating verification results

Return JSON:
{
  "summary": "brief summary",
  "best_speedup": ${finalBest.geomean_speedup},
  "generations": ${maxGenerations},
  "total_evaluations": <int>,
  "report_path": "path/to/report.md"
}`,
    {
      label: 'Generate report',
      phase: 'Report',
      schema: {
        type: 'object',
        properties: {
          summary: { type: 'string' },
          best_speedup: { type: 'number' },
          generations: { type: 'integer' },
          total_evaluations: { type: 'integer' },
          report_path: { type: 'string' },
        },
        required: ['summary', 'best_speedup', 'generations'],
      },
    }
  );

  // ============================================================================
  // Return final results
  // ============================================================================

  return {
    success: true,
    method: 'ArchAgent',
    approach: 'AlphaEvolve with island model',
    simulator: 'ChampSim',
    islands: numIslands,
    generations: maxGenerations,
    population_per_island: populationPerIsland,
    best_candidate: finalBest.candidate_id,
    baseline: 'LRU',
    speedup: finalBest.geomean_speedup,
    held_out_speedup: validationResult.held_out_speedup,
    state_usage_kb: finalBest.state_usage_kb,
    validation_passed: validationResult.validation_passed,
    report: report?.report_path,
    summary: report?.summary,
  };
}

// Execute the workflow
return await main();
