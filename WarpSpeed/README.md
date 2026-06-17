# WarpSpeed — Rewindable Parallel Kernel-Optimization Search

A Claude Code Dynamic Workflow that runs a sustained, experiment-driven search
over CUDA kernel optimizations on a single multi-GPU node. It is built around
five ideas:

1. **Checkpoint tree, not a patch loop.** Every accepted kernel state is a git
   commit annotated with the structural-assumption tags baked into it (e.g.
   `persistent_kernel`, `fp8_reg_accum`). The search tree lives in SQLite.
2. **Experiments, not edits.** Every candidate states one falsifiable
   hypothesis with a MANDATORY predicted gain and mechanism before any code is
   written; prediction gaps are tracked and fed back to the planner.
3. **Rewind discards code, never knowledge.** BitLessons are append-only and
   survive rewinds; a rewind is a frontier-pointer move in the DB (git history
   is untouched, the subtree is retired, a replay experiment re-derives the
   validated downstream strategies on the rewound base).
4. **The GPU is a mutex-protected resource.** Nothing touches a device except
   through `infra/gpu_run` (machine-global flocks, per-role device policy,
   timeout kill, GPU-minute accounting). Two-tier benchmarking: interleaved
   A/B screening on pool devices (device-unbiased relative speedup) routes
   decision-relevant candidates to a locked-clock canonical device.
5. **Cross-model review.** Every candidate is independently re-validated from
   a fresh clone and reviewed by a DIFFERENT model (codex CLI) for race/
   barrier/fence correctness and intent compliance, with at most 3
   implementor↔reviewer iterations.

## Decoupling contract (where things live)

This directory contains **only code and templates** — it is never written to
at run time. The target kernel project is `project_dir` (any git repo on
disk). All run state derives from the project:

```
<project_dir>/                     # YOUR kernel repo (anywhere on disk)
├── harness/                       # project-owned correctness ground truth
│   ├── correctness.py             #   (scaffolded once from harness-template/,
│   ├── problem_shapes.json        #    completed by a human, then chmod a-w)
│   └── tolerances.json
└── .warpspeed/                    # all run state (gitignored; state_dir to move)
    ├── config.json                # materialized by tools/render_config.py at Init
    ├── search.sqlite              # checkpoint tree + experiments + ledger
    ├── bitlessons.jsonl|.md       # append-only knowledge (survives rewinds)
    ├── worktrees/ review/ builds/ # per-experiment git worktrees, review clones,
    ├── ncu_cache/ logs/ results/  # parent-binary cache, NCU cache, report
/tmp/warpspeed/locks/gpu_N         # MACHINE-GLOBAL device locks (cross-project safe)
```

## Prerequisites (on the GPU node)

- exclusive access to a multi-GPU NVIDIA node (default policy: devices 0–5
  screening pool, 6 locked-clock bench, 7 NCU profiling; override via config)
- git ≥ 2.20, Python ≥ 3.8 (stdlib only), the project's own build toolchain
- `nvidia-smi`, NCU (`ncu`), `compute-sanitizer` on PATH; clock locking wants
  root or a sudo rule (otherwise tier-2 runs at default clocks with a warning
  and the actual clocks are recorded in every confirm result)
- the cross-model reviewer CLI (default `codex exec`) installed and
  authenticated — Init preflight hard-fails before any GPU spend if missing
- a completed correctness harness in the project (first run scaffolds
  `harness-template/` into the project and exits with instructions; a human
  fills in the reference/tolerances once, verifies by hand, and re-runs)

Benchmark-binary contract: the build output must self-time and print
`LAT_US=<float>` lines under `--reps/--warmup/--shape` flags — see
[harness-template/README.md](harness-template/README.md).

## Usage

```js
Workflow({
  scriptPath: 'WarpSpeed/warpspeed-kernel-search.js',
  args: {
    project_dir: '/abs/path/to/kernel-repo',
    build_command: './build.sh',
    binary_path: 'bin/kernel_bench',
    kernel_paths: 'src/kernel.cu,src/kernel_traits.cuh',
    target_gpu: 'H100',
    iterations: 20,
    budget_gpu_minutes: 480,
  },
})
```

### Parameters

| name | required | default | meaning |
|---|---|---|---|
| `project_dir` | yes | — | absolute path to the kernel git repo |
| `build_command` | yes | — | project build, run verbatim inside each worktree |
| `binary_path` | yes | — | repo-relative path of the built self-timing binary |
| `kernel_paths` | yes | — | comma-separated editable sources = reviewer diff allowlist |
| `target_gpu` | yes | — | H100 / B200 / sm90 / sm100 (selects hardware-facts + wiki scope) |
| `warpspeed_dir` | no | `WarpSpeed` | path to this directory (resolved absolute at Init) |
| `state_dir` | no | `<project>/.warpspeed` | run-state home |
| `harness_dir` | no | `<project>/harness` | project harness location |
| `iterations` | no | 20 | maximum search rounds |
| `budget_gpu_minutes` | no | — | GPU-minute ceiling (tracked per gpu_run invocation) |
| `target_latency_us` | no | — | stop when the confirmed best reaches this |
| `parallel_agents` | no | 4 | experiment fan-out per round |
| `reviewer_cmd` | no | `codex exec` | cross-model reviewer CLI |
| `bench_shape` | no | `default` | shape used for screen/confirm/profile |
| `exp_dir` | no | — | KerSor session run dir (`genome.jsonl` + report mirror); search state stays in `state_dir` |
| `op_description` | no | — | free-text task context for planner + implementor |
| `single_spec_json` | no | — | M2 mode: run exactly one hand-written spec, then report |
| `config_overrides` | no | — | JSON merged into the materialized config (clocks, reps, devices, …) |

Outputs: `{ok, stop_reason, rounds_run, baseline_latency_us, best_latency_us,
best_commit, speedup, lessons_total, report_path, state_dir, history}` plus a
human report at `<state>/results/report.md`. Watch a live run with the wsdb
status subcommand against `<state>/search.sqlite`.

## How a round works

```
Plan      recover orphans -> snapshot tree -> allocation (exploit/explore/wildcard,
          exploit-shifted as the GPU-minute ledger depletes) -> planner emits specs
          (soft dedup by strategy-set hash; queued replay/ablation specs first)
Generate  per spec: git worktree from parent checkpoint; implementor edits ONLY
          kernel_paths, proves correctness via the fixed harness invocation, commits
          (even on failure - evidence); reviewer re-validates from a FRESH clone
          (build + harness + sanitizers) and relays the codex verdict; <=3 iterations
Screen    interleaved A/B vs parent binary on one pool device (relative, unbiased)
Confirm   only when decision-relevant: new-best claim, or |delta| < CONFIRM_MARGIN
Profile   curated NCU sections (checkpoints cached once, ever) + analyst diagnosis
          - including correct_slower candidates (the best lesson material)
Record    per-candidate rows; round barrier: hard dedup (latency within noise AND
          equal NCU fingerprint), checkpoint promotion, blocked counting
Postmortem blocked >= K rounds -> blame agent -> ONE ablation experiment ->
          confirmed: rewind frontier + retire subtree + queue replay
          refuted: keep subtree, record the refutation as a lesson
Maintain  worktree cleanup, lesson compaction every COMPACT_EVERY rounds
```

Significance is a single rule: a candidate counts as faster only when its
relative speedup exceeds `max(MIN_GAIN_PCT, 2 x noise)` where noise =
max(within-device screening std, cross-device sigma from one-time
calibration). No superstition lessons below that line.

## Testing

```
WarpSpeed/tests/mock/run_mock_tests.sh         # T1-T13, no GPUs needed (macOS/Linux)
node WarpSpeed/tests/workflow/dryrun.test.mjs  # orchestrator logic under the vm harness
WarpSpeed/tests/node/sync_to_node.sh H100-lsh  # rsync to the GPU node
ssh H100-lsh 'warpspeed-accept/WarpSpeed/tests/node/run_node_tests.sh'   # real-GPU acceptance
```

The mock suite exercises the REAL scripts end-to-end with PATH-injected fake
GPU tools and a deterministic mock kernel project whose source header drives
its measured latency — the full search loop (screening deltas, significance,
dedup, rewind) runs on a laptop.

---

# WarpSpeed — 可回退的并行内核优化搜索（中文）

在单台多 GPU 节点上做持续的、实验驱动的 CUDA 内核优化搜索：

- **检查点树**：每个被接受的内核状态是一个带"结构假设标签"的 git 提交，搜索树存于 SQLite；
- **实验制**：每个候选先声明一句可证伪的假设 + 必填的预期收益与机制，预测偏差回流给规划器；
- **回退只丢代码、不丢知识**：BitLessons 仅追加、跨回退留存；回退 = 前沿指针在数据库里后移并退役子树，git 历史不动，随后排队 replay 实验在回退基座上重做已验证的下游策略；
- **GPU 是互斥资源**：一切设备访问必须经过 `infra/gpu_run`（机器级 flock、按角色分配设备、超时强杀、GPU 分钟记账）；两级基准：池设备上交错 A/B 得到与设备无关的相对加速，再按需路由到锁频基准卡确认绝对时延；
- **跨模型评审**：每个候选由另一模型（codex CLI）从全新 clone 独立复验正确性、审查竞态/屏障/fence 与"是否真的实现了所述假设"，至多 3 轮往返。

解耦约定：本目录只放代码与模板；目标内核工程由 `project_dir` 指定（任意路径的 git 仓库），全部运行状态在 `<project>/.warpspeed/`；设备锁在机器级 `/tmp/warpspeed/locks`，因此多个项目并发搜索也不会撞卡。

首次运行会把 harness 模板脚手架到工程里并退出，由人完成参考实现与容差（只此一次，之后目录被设为只读）；之后再启动即进入搜索。本地无 GPU 时可用 `tests/mock/run_mock_tests.sh` 跑通全部 13 项验收；真实 GPU 验收见 `tests/node/`。
