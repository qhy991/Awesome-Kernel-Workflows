# KernelBlaster Workflow

Memory-augmented in-context RL workflow for CUDA kernel optimization, adapted from the KernelBlaster paper:

- Paper: [KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning](https://arxiv.org/abs/2602.14293)
- Code: [github.com/NVlabs/KernelBlaster](https://github.com/NVlabs/KernelBlaster) (Apache-2.0)
- Workflow file: [`kernelblaster-kernel-optimization.js`](kernelblaster-kernel-optimization.js)

**English** · [简体中文](README.zh-CN.md)

---

## What it does

KernelBlaster reframes CUDA kernel optimization as **memory-augmented in-context RL (MAIC-RL)**. Instead of guessing optimizations or forgetting what worked on previous kernels, it maintains a **persistent, state-keyed knowledge base** and reuses it across kernels and runs.

Each **rollout** is a multi-step trajectory:

1. **Profile + classify** the current kernel into a hardware performance *state* (memory / compute / latency bound) using NCU.
2. **Retrieve** the best-known optimization for that state from the knowledge base (scored by confidence × predicted speedup, penalizing already-used techniques).
3. **Apply** the optimization (strategy-guided prompt).
4. **Evaluate** — compile, verify correctness, NCU-profile to get **Elapsed Cycles**.
5. **Reward + update** — compute the reward, append a trajectory step, and update the database entry's measured payoff and confidence.

Periodically a **policy-update cycle** analyzes the replay buffer and adjusts database confidence. Because the knowledge base is written back to disk, experience accumulates **across invocations** — the paper's core novelty.

```
                 persistent optimization_database.json  (cross-task memory)
                          │ load                                ▲ write-back
                          ▼                                      │
  Setup ─▶ ┌──────────────── rollout (× rl_iterations) ──────────────┐ ─▶ Learn ─▶ Report
           │  Retrieve ─▶ Execute ─▶ Evaluate ─▶ Reward+Update  (× rollout_steps) │
           │     ▲                                         │                       │
           │     └──────────── new state ─────────────────┘                       │
           └─────────────────────────────────────────────────────────────────────┘
```

---

## The performance-state taxonomy

KernelBlaster keys its knowledge base on three hardware states (faithful to `data/kernelblaster/optimization_database.json`):

| State | Primary bottleneck | Representative techniques |
|-------|--------------------|---------------------------|
| `memory_bandwidth_limited` | memory_bound | vectorized access (float4/half2), coalescing, shared-memory tiling, AoS→SoA layout |
| `compute_throughput_limited` | compute_bound | tensor cores, instruction-mix balancing, functional-unit utilization, algorithmic change |
| `latency_occupancy_limited` | latency_bound | increase occupancy, more parallelism, reduce synchronization, thread coarsening |

Each technique entry tracks `confidence_score`, `usage_count`, `predicted_improvement`, and `actual_speedup` — these are the values the workflow updates as it measures real results.

---

## The reward function

Verbatim from the paper's `calculate_reward`:

```
actual_improvement% = (prev_cycles − new_cycles) / prev_cycles × 100
reward = actual_improvement/100
       + accuracy_bonus      # +0.2 if 0.8 ≤ actual/predicted ≤ 1.2, else −0.1·|accuracy−1|
       + penalty             # −0.5 if the variant is not faster
```

A rollout stops early on severe degradation (`actual_improvement < −20%`) or when no unused optimization remains for the current state.

---

## Arguments

| Arg | Default | Meaning |
|-----|---------|---------|
| `kernel_path` | (required) | Target CUDA kernel (`init.cu`) |
| `op_description` | `'CUDA kernel'` | Human-readable op description |
| `optimization_db_path` | `''` | Persistent knowledge base JSON. Loaded at Setup, written back after the run — **cross-task memory** |
| `harness_build_cmd` | `''` | Build command (KernelBench-CUDA `driver.cpp` + `init.cu`) |
| `harness_run_args` | `''` | Runtime args for the harness binary |
| `kernel_name_regex` | `''` | Regex for `ncu -k` |
| `ncu_binary` | `'ncu'` | Nsight Compute binary |
| `exp_dir` | `'/tmp/kernelblaster_exp'` | NCU reports / experiment outputs |
| `rl_iterations` | `3` | Number of rollouts (trajectories) |
| `rollout_steps` | `4` | Max optimization iterations per rollout |
| `update_frequency` | `3` | Run a policy-update cycle every N trajectories |
| `buffer_size` | `100` | Replay-buffer capacity |
| `target_gpu` | `'L40S'` | Target GPU for optimization hints |
| `test_command` / `benchmark_command` | `''` | Fallbacks when NCU is unavailable |
| `run_timestamp_iso` | `'unknown'` | ISO timestamp for report headers |

---

## Example invocation

```javascript
Workflow({
  name: 'kernelblaster-kernel-optimization',
  args: {
    kernel_path: '/path/to/kernelbench-cuda/level1/001_Square_matrix_multiplication/init.cu',
    op_description: 'Square matrix multiplication',
    optimization_db_path: '/path/to/optimization_database.json',
    harness_build_cmd: '<user-provided harness build command>',
    kernel_name_regex: 'matmul_kernel',
    ncu_binary: '<user-provided ncu binary path>',
    exp_dir: '/tmp/kernelblaster_exp',
    rl_iterations: 3,
    rollout_steps: 4,
    target_gpu: 'L40S',
  },
})
```

Run on several kernels in sequence with the **same** `optimization_db_path` to accumulate cross-task experience.

---

## Outputs

- `baseline_cycles`, `best_cycles`, `overall_speedup`
- `rollouts_completed`, `trajectory_count`
- `optimization_db` (final knowledge base; also written to `optimization_db_path`)
- `best_kernel_code`
- `report`

---

## Prerequisites

- NVIDIA GPU + CUDA toolchain (`nvcc`)
- **NVIDIA Nsight Compute (`ncu`)** — KernelBlaster is Elapsed-Cycles-driven; without `ncu`, pass `test_command` / `benchmark_command` for fallbacks
- KernelBench-CUDA-style task layout (`init.cu` + `driver.cpp`) recommended

---

## Citation

```bibtex
@article{dong2026kernelblaster,
  title={KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning},
  author={Dong, Kris Shengjun and Modi, Sahil and Nikiforov, Dima and Damani, Sana and Lin, Edward and Hari, Siva Kumar Sastry and Kozyrakis, Christos},
  journal={arXiv preprint arXiv:2602.14293},
  year={2026},
  url={https://arxiv.org/abs/2602.14293}
}
```
