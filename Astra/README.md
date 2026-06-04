# Astra Workflow

Multi-agent CUDA kernel optimization workflow adapted from:

- Paper: [Astra: A Multi-Agent System for GPU Kernel Performance Optimization](https://arxiv.org/abs/2509.07506)
- Repository: [Anjiang-Wei/Astra](https://github.com/Anjiang-Wei/Astra)
- Workflow file: [`astra-kernel-optimization.js`](astra-kernel-optimization.js)

**English** · [简体中文](README.zh-CN.md)

---

## What it does

`astra-kernel-optimization` optimizes an existing CUDA kernel through specialized agent roles:

1. Setup reads the initial CUDA kernel and callable/export contract
2. Testing Agent prepares correctness and shape coverage
3. Profiling Agent measures baseline latency and bottlenecks
4. Planning Agent proposes one evidence-grounded optimization
5. Coding Agent edits the current best kernel
6. Testing + Profiling validate the candidate with real evaluator evidence
7. Record carries measured lessons into the next iteration
8. PostProcess prepares reintegration notes

The workflow is aimed at production-style CUDA kernels, especially SGLang-like kernels, where the starting point is an existing CUDA/PyBind implementation rather than a PyTorch module specification.

---

## Arguments

Primary arguments accepted by this workflow:

- `kernel_path`: path to initial CUDA kernel file
- `compare_kind`: comparison mode tag such as `generic`, `rmsnorm`, `silu`, or `mergestate`
- `baseline_module`: Python module containing the baseline function, default `sgl_kernel`
- `baseline_func`: baseline function name
- `generated_export_func`: export function expected in the generated PyBind module
- `test_command`: correctness command using `{kernel_path}` and `{result_path}`
- `benchmark_command`: benchmark command using `{kernel_path}` and `{result_path}`
- `iterations`: optimization iteration iterations, default `5`
- `target_gpu`: target GPU, default `H100`
- `exp_dir`: experiment output directory

Evaluator JSON should contain:

```json
{
  "compiled": true,
  "correct": true,
  "speedup": 1.23,
  "runtime_ms": 0.12,
  "baseline_runtime_ms": 0.15,
  "error_message": "",
  "profile_summary": ""
}
```

---

## Example invocation

```javascript
Workflow({
  name: 'astra-kernel-optimization',
  args: {
    kernel_path: '/path/to/rms_v1.cu',
    compare_kind: 'rmsnorm',
    baseline_module: 'sgl_kernel',
    baseline_func: 'fused_add_rmsnorm',
    generated_export_func: 'sgl_fused_add_rmsnorm',
    test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
    benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
    iterations: 5,
    target_gpu: 'H100',
    exp_dir: '/tmp/astra_exp',
  },
})
```

---

## Outputs

The workflow returns:

- `best_speedup`
- `best_runtime_ms`
- `baseline_runtime_ms`
- `best_kernel_code`
- `run_log`
- `lessons`
- `post_process`
- `report`

---

## Citation

```bibtex
@inproceedings{wei2025astra,
  title={Astra: A Multi-Agent System for GPU Kernel Performance Optimization},
  author={Wei, Anjiang and Sun, Tianran and Seenichamy, Yogesh and Song, Hang and Ouyang, Anne and Mirhoseini, Azalia and Wang, Ke and Aiken, Alex},
  booktitle={NeurIPS 2025 Fourth Workshop on Deep Learning for Code},
  year={2025}
}
```
