# ReGraphT Workflow

Reasoning-graph-guided CUDA optimization workflow adapted from:

- Paper: [From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph](https://arxiv.org/abs/2510.19873)
- Workflow file: [`regrapht-kernel-optimization.js`](regrapht-kernel-optimization.js)

**English** · [简体中文](README.zh-CN.md)

---

## What it does

`regrapht-kernel-optimization` turns the ReGraphT method into an agent-executable optimization loop.

It does not train a small model. Instead, it uses the paper's training-free inference idea:

1. Read the CUDA task and evaluator contract
2. Build or load a CUDA Reasoning Graph from optimization trajectories
3. Select an optimization-method path with Monte Carlo Graph Search
4. Generate a CUDA candidate using selected methods and examples
5. Compile, correctness-test, and benchmark the candidate
6. Update graph rewards and examples from measured evidence
7. Report the best kernel and remaining graph paths

---

## Recommended use cases

- You have LLM optimization traces and want to reuse them as structured CUDA expertise
- You want local/smaller coding models to follow multi-step CUDA reasoning paths
- You need graph-guided retrieval instead of plain nearest-neighbor code RAG
- You want failures to update the method graph rather than disappear into logs

---

## Arguments

Primary arguments accepted by this workflow:

- `kernel_path`: path to source CUDA/C++ code or task spec
- `benchmark_command`: command that writes evaluator JSON using `{kernel_path}` and `{result_path}`
- `op_description`: operation description
- `trace_corpus_path`: optional JSONL corpus of optimization trajectories
- `graph_path`: optional persisted CUDA Reasoning Graph
- `baseline_command`: optional baseline evaluator command
- `iterations`: candidate attempts, default `20`
- `rollouts_per_select`: conceptual MCGS rollouts per selection, default `12`
- `exploration_weight`: UCT exploration weight, default `1.4`
- `max_path_length`: max graph path length, default `4`
- `target_gpu`: target GPU, default `H100`
- `exp_dir`: experiment output directory

Evaluator JSON should contain:

```json
{
  "compiled": true,
  "correct": true,
  "speedup": 1.23,
  "kernel_time_ms": 0.12,
  "baseline_time_ms": 0.15,
  "error_message": "",
  "error_type": ""
}
```

---

## Example invocation

```javascript
Workflow({
  name: 'regrapht-kernel-optimization',
  args: {
    kernel_path: '/path/to/source.cu',
    op_description: 'Sequential stencil kernel to CUDA',
    benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
    trace_corpus_path: '/path/to/llm_optimization_traces.jsonl',
    graph_path: '/path/to/regraph.json',
    iterations: 20,
    rollouts_per_select: 12,
    exploration_weight: 1.4,
    max_path_length: 4,
    target_gpu: 'H100',
    exp_dir: '/tmp/regrapht_exp',
  },
})
```

---

## Outputs

The workflow returns:

- `best_speedup`
- `best_kernel_code`
- `best_candidate_id`
- `evaluated_candidates`
- `correct_candidates`
- `selected_paths`
- `graph`
- `graph_stats`
- `report`

---

## Citation

```bibtex
@article{gong2025regrapht,
  title={From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph},
  author={Gong, Junfeng and Wei, Zhiyi and Chen, Junying and Liu, Cheng and Li, Huawei},
  journal={arXiv preprint arXiv:2510.19873},
  year={2025},
  url={https://arxiv.org/abs/2510.19873}
}
```
