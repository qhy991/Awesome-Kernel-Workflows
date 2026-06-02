# K-Search Workflow

World-model-guided tree search workflow for GPU kernel optimization, adapted from the K-Search paper:

- Paper: [K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model](https://arxiv.org/abs/2602.19128)
- Workflow file: [`ksearch-kernel-optimization.js`](ksearch-kernel-optimization.js)

**English** · [简体中文](README.zh-CN.md)

---

## What it does

`ksearch-kernel-optimization` is designed for kernels with a large and highly-coupled design space, where single-path iterative tuning is often not enough.

It keeps a co-evolving **decision tree (world model)** and runs repeated cycles:

1. Setup
2. Initialize world model tree
3. Select best frontier action
4. Generate / debug / improve kernel code
5. Evaluate correctness and performance
6. Refine tree on success or backtrack on failure
7. Report final trajectory and best solution

---

## Recommended use cases

- Complex operators (e.g., MLA attention, MoE routing, fused kernels)
- Multiple orthogonal design axes (tiling, memory hierarchy, parallel decomposition, algorithm variants)
- Frequent local minima where backtracking and branch exploration matter

---

## Arguments

Primary arguments accepted by this workflow:

- `kernel_spec_path`: path to problem spec (required)
- `op_description`: operation description
- `language`: `triton | cuda | python`
- `target_gpu`: GPU target, e.g., `H100`
- `max_cycles`: max search cycles (default `10`)
- `attempts_per_cycle`: generate/improve rounds per cycle (default `5`)
- `stagnation_window`: non-improving window to stop a cycle (default `3`)
- `max_difficulty`: max action difficulty to select first (default `4`)
- `bench_command`: benchmark command
- `baseline_code_path`: optional baseline kernel path
- `rtol` / `atol`: correctness tolerance
- `exp_dir`: output directory

For exact behavior, see the source in [`ksearch-kernel-optimization.js`](ksearch-kernel-optimization.js).

---

## Example invocation

```javascript
Workflow({
  name: 'ksearch-kernel-optimization',
  args: {
    kernel_spec_path: '/path/to/spec.yaml',
    op_description: 'MLA decode attention kernel',
    language: 'triton',
    target_gpu: 'H100',
    max_cycles: 10,
    attempts_per_cycle: 5,
    stagnation_window: 3,
    max_difficulty: 4,
    bench_command: 'python eval.py --kernel',
    baseline_code_path: '/path/to/baseline.py',
    rtol: 0.01,
    atol: 0.01,
    exp_dir: '/tmp/ksearch_exp',
  },
})
```

---

## Outputs

The workflow returns:

- `best_metric`
- `best_solution_code`
- `cycles_completed`
- `solutions_evaluated`
- `decision_tree` (final world model state)
- `solution_lineage` (top solution trace)
- `report`
- `baseline_metric`
- `speedup_over_baseline`

---

## Citation

If you use this workflow in research or engineering reports, please cite the original K-Search paper:

```bibtex
@article{cao2026ksearch,
  title={K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model},
  author={Cao, Shiyi and Mao, Ziming and Gonzalez, Joseph E. and Stoica, Ion},
  journal={arXiv preprint arXiv:2602.19128},
  year={2026},
  url={https://arxiv.org/abs/2602.19128}
}
```
