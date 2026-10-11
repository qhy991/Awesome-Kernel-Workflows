# CUDA-LLM FSR Workflow

Feature Search and Reinforcement workflow adapted from:

- Paper: [CUDA-LLM: LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092)
- Workflow file: [`cudallm-fsr-kernel-generation.js`](cudallm-fsr-kernel-generation.js)

**English** · [简体中文](README.zh-CN.md)

---

## What it does

`cudallm-fsr-kernel-generation` generates CUDA kernels by explicitly searching CUDA optimization features and reinforcing feature choices with measured evidence.

The loop is:

1. Read the task spec and reference implementation
2. Build a CUDA feature catalog
3. Generate diverse correctness and boundary tests
4. Select compatible features
5. Generate a CUDA candidate
6. Compile, correctness-test, and benchmark it
7. Reinforce feature scores from compile/correctness/speedup reward
8. Report the best kernel and feature reward table

This is not model training. It is an agent-executable FSR loop suitable for Claude Code workflows.

---

## Arguments

Primary arguments accepted by this workflow:

- `problem_path`: CUDA generation task specification
- `reference_code_path`: optional reference implementation
- `benchmark_command`: evaluator command using `{kernel_path}` and `{result_path}`
- `target_gpu`: target GPU, default `H100`
- `iterations`: feature-search iterations, default `8`
- `feature_budget`: maximum features per candidate, default `4`
- `samples_per_feature_set`: candidate samples per iteration, default `2`
- `rtol` / `atol`: correctness tolerances
- `exp_dir`: output directory
- `termination_file`: optional non-empty cooperative-stop request file
- `deadline_epoch`: optional positive Unix epoch deadline

Native `task_result_command` runs write `exp_dir/checkpoint.json` after each
candidate's evaluation and reinforcement, including typed failures. Legacy runs
enable the same safe points when either stop argument is supplied. A stop returns
the already reduced feature scores, bound best candidate, complete candidate-file
index, `candidates_completed`, and the number of **fully** completed iterations;
the optional final-report agent is skipped. No in-flight build or evaluation is
cancelled, and the original FSR reward rules and declared search budget remain intact.

The checkpoint stores original arguments, feature catalog/scores, test definitions,
candidate source/result paths and source hashes. Complete source and diagnostics
remain in those original files. It is an audit snapshot for recovery through the
**same native Workflow activation**, with its existing replay identity and budget.
There is no independent checkpoint loader or automatic resume into a new activation.
Copying a checkpoint does not grant extra candidate slots or qualify a kernel.

Evaluator JSON should contain:

```json
{
  "compiled": true,
  "correct": true,
  "speedup": 1.23,
  "latency_ms": 0.12,
  "baseline_latency_ms": 0.15,
  "error_message": "",
  "passed_tests": 128,
  "total_tests": 128
}
```

---

## Example invocation

```javascript
Workflow({
  name: 'cudallm-fsr-kernel-generation',
  args: {
    problem_path: '/path/to/task.md',
    reference_code_path: '/path/to/reference.py',
    benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
    target_gpu: 'H100',
    iterations: 8,
    feature_budget: 4,
    samples_per_feature_set: 2,
    rtol: 0.01,
    atol: 0.01,
    exp_dir: '/tmp/cudallm_fsr_exp',
  },
})
```

---

## Outputs

The workflow returns:

- `best_speedup`
- `best_latency_ms`
- `best_kernel_code`
- `best_candidate_id`
- `feature_scores`
- `candidates`
- `report`

---

## Citation

```bibtex
@article{chen2025cudallm,
  title={CUDA-LLM: LLMs Can Write Efficient CUDA Kernels},
  author={Chen, Wentao and Zhu, Jiace and Fan, Qi and Ma, Yehan and Zou, An},
  journal={arXiv preprint arXiv:2506.09092},
  year={2025},
  url={https://arxiv.org/abs/2506.09092}
}
```

### Native task self-test qualification

Optional `task_result_command` and `task_workload_count` enable the frozen-task Python/CuTe path. Snapshot explicitly selected files, execute full official tests through the broker, and reduce matching raw task/terminal receipts. Only complete correct measured results may score; NCU is diagnostic. Preserve finite native topology/defaults. CUDAAgent/AccelOpt compare to their independently generated initial implementation; KernelFoundry/KSearch/CUDALLM report official-reference scores. Return the tested snapshot rather than rewritten display code. Audit native agent JSON relays against raw files and independent final acceptance; they are not process-Host evaluate receipts or formal-ablation authority.
