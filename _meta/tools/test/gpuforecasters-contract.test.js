'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'GPUForecasters/gpuforecasters-kernel-optimization.js')

const marker = 'GPUFORECASTERS_CONTRACT_MARKER'

const baseArgs = {
  language: 'cuda',
  problem_type: 'cuda-kernel-optimization',
  note: `${marker}: user note carries required validation details`,
  kernel_path: `/tmp/${marker}/baseline.cu`,
  problem_definition: 'Optimize a CUDA row-wise reduction kernel.',
  op_description: 'row-wise reduction',
  target_gpu: 'H100',
  test_command: `python /tmp/${marker}/test.py {kernel_path} {result_path}`,
  benchmark_command: `python /tmp/${marker}/bench.py {kernel_path} {result_path}`,
  baseline_latency_ms: 0.42,
  baseline: `${marker}: baseline is the reference CUDA implementation`,
  iterations: 2,
  curriculum_size: 3,
  exp_dir: `/tmp/${marker}/exp`,
}

const returns = {
  'Setup GPUForecasters': {
    kernel_name: 'row_reduce',
    optimization_space: { parameters: ['block_size'], search_space_size: 4 },
    forecaster_models: ['ordinal-forecaster'],
    training_budget: 3,
    puct_exploration_constant: 2,
    puct_simulation_budget: 2,
    tree_depth_limit: 2,
    baseline_perf: 0.42,
    backend: 'local',
    target_gpu: 'H100',
  },
  'Train forecasters': {
    training_samples: 3,
    training_dataset_size: 3,
    trained_models: [
      {
        model_name: 'ordinal-forecaster',
        architecture: 'Transformer',
        train_mae: 0.1,
        val_mae: 0.1,
        abstention_rate: 0.2,
        abstention_strategy: 'native',
      },
    ],
    best_training_speedup: 1.1,
    best_training_config: 'block_size=256',
  },
  'Calibrate forecasters': {
    calibrated_models: [
      {
        model_name: 'ordinal-forecaster',
        calibrated_threshold: 0.3,
        calibrated_mae: 0.1,
        calibrated_abstention_rate: 0.2,
        coverage: 0.8,
      },
    ],
    ensemble_strategy: 'unanimous',
    ensemble_mae: 0.1,
    ensemble_abstention_rate: 0.2,
    ensemble_coverage: 0.8,
  },
  'PUCT search': {
    simulations: 2,
    total_executions: 1,
    abstention_saved_executions: 1,
    tree_nodes_explored: 2,
    tree_max_depth: 1,
    best_config: 'block_size=256',
    best_speedup: 1.2,
    search_trajectory: [{ step: 1, speedup: 1.2, executed: true }],
  },
  'Refine candidates': {
    refinement_candidates: 1,
    refinement_executions: 1,
    best_refined_config: 'block_size=512',
    best_refined_speedup: 1.3,
    improvement_over_puct: 0.1,
    ablation_insights: 'larger block improved occupancy',
  },
  'Validate best config': {
    config: 'block_size=512',
    validation_runs: 3,
    mean_speedup: 1.3,
    std_speedup: 0.02,
    min_speedup: 1.28,
    max_speedup: 1.32,
    correctness_passed: true,
    hardware_utilization: {},
    validation_passed: true,
  },
  'Generate report': {
    summary: 'ok',
    best_speedup: 1.3,
    total_executions: 5,
    executions_saved: 1,
    search_efficiency: 0.26,
    report_path: '/tmp/gpuforecasters-report.md',
  },
}

test('threads user note and evaluator contracts into GPUForecasters prompts', async () => {
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: baseArgs, agentReturns: returns })
  assert.deepEqual(calls.map(c => c.label), [
    'Setup GPUForecasters',
    'Train forecasters',
    'Calibrate forecasters',
    'PUCT search',
    'Refine candidates',
    'Validate best config',
    'Generate report',
  ])

  for (const call of calls) {
    assert.match(call.prompt, new RegExp(marker), `${call.label} prompt must include the user note/baseline marker`)
    assert.match(call.prompt, new RegExp('test_command: ' + 'python .*test\\.py \\{kernel_path\\} \\{result_path\\}'), `${call.label} prompt must include test_command`)
    assert.match(call.prompt, new RegExp('benchmark_command: ' + 'python .*bench\\.py \\{kernel_path\\} \\{result_path\\}'), `${call.label} prompt must include benchmark_command`)
    assert.match(call.prompt, /baseline_latency_ms: 0\.42/, `${call.label} prompt must include baseline latency`)
  }
})
