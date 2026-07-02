# GemmPTX：从 PTX/SASS 证据出发的 GEMM 优化

[English](README.md) · **简体中文**

GemmPTX 是一个 GEMM 专用 workflow，用来优化**已有 CUDA/CuTe/CUTLASS GEMM kernel**。它从指令层往上走，模仿人类专家的流程：

```
Hardware Census -> GEMM Signature -> Baseline Evidence
-> Instruction Plan -> Implement -> Disassemble Verify
-> Profile -> Decide -> Report
```

核心契约很硬：候选不能只因为跑得快就声称用了 `mma.sync`、`wgmma.mma_async`、TMA 或 `tcgen05`。只有 PTX/SASS 证据证明目标指令路径真的出现，才算该指令假设成立。编译正确但没有 lower 到预期指令的候选会记录为 `hypothesis_not_realized`。

## 适用范围

适合使用 GemmPTX 的情况：

- 已有 CUDA、CuTe、CUTLASS 或 C++ GEMM-like kernel。
- 能提供编译、正确性、benchmark 和反汇编命令。
- 想优化指令选择和 lowering 行为：`mma.sync`、`wgmma.mma_async`、`cp.async.bulk.tensor` / TMA、mbarrier pipeline，或 `tcgen05` / TMEM。

不适合的情况：

- 你需要通用 compute-bound optimizer。它**不是通用 compute-bound optimizer**。
- 任务是 softmax、reduction、layernorm、stencil、elementwise、FFT、sampling，而不是 GEMM/matmul。
- 无法提供 `disassemble_command`；没有反汇编就无法验证 PTX/SASS 路径。
- 目标是 SOL-ExecBench 的 CUTLASS 运行时 dispatch threshold 调优；这种情况应使用 [CutlassGEMM](../CutlassGEMM/)。

## 必需证据

| 参数 | 必需 | 契约 |
|---|---:|---|
| `kernel_path` | 是 | 已有 GEMM kernel 源码。workflow 会把候选写到 `exp_dir`，不会覆盖原文件。 |
| `compile_command` | 是 | 使用 `{candidate_path}` 或 `{kernel_path}` 以及 `{result_path}` 的命令；应报告 `compiled` 和 artifact 路径。 |
| `test_command` | 是 | 正确性命令；应报告 `correct`。 |
| `benchmark_command` | 是 | 性能命令；应报告 `latency_ms`、`throughput` 或 `speedup`。 |
| `disassemble_command` | 是 | PTX/SASS 证据命令；应报告 artifact 路径、observed instructions、registers/thread、SMEM、local memory、spill。 |
| `hardware_probe_command` | 否 | 运行时 GPU 事实。缺失时只能使用静态/兜底事实，并标记为未实测。 |
| `profile_command` / `ncu_command` | 否 | 可选 NCU/native profile 指标，用于机制诊断。 |

命令可使用这些占位符：`{kernel_path}`、`{candidate_path}`、`{artifact_path}`、`{result_path}`、`{exp_dir}`、`{target_gpu}`。

## Workflow-local skill

GemmPTX 带有一个本地专家 skill：`GemmPTX/skills/gemmptx-instruction-evidence/SKILL.md`。workflow 把它声明为可选 skill binding：`gemmptx-instruction-evidence`；规划、实现和反汇编验证 prompt 会要求 agent 在可用时读取它。

这个 skill 存放精简的规则包：架构到指令路径映射、GEMM triage、PTX/SASS regex 证据门，以及 WGMMA/TMA/`tcgen05` 常见失败模式。KerSor 或其他 runner 可以注入这个 skill，但权威来源跟随 AKW 里的 workflow。

## 示例

```javascript
Workflow({name: 'gemmptx-gemm-optimization', args: {
  kernel_path: '/abs/project/src/gemm.cu',
  problem_definition: 'bf16 GEMM: C[M,N] = A[M,K] @ B[K,N], fp32 accumulation',
  language: 'cuda',
  target_gpu: 'H100',
  exp_dir: '/tmp/gemmptx-run',
  iterations: 3,

  hardware_probe_command: '/abs/tools/probe_gpu.py --json > {result_path}',
  compile_command: '/abs/project/tools/build_candidate.sh {candidate_path} {artifact_path} > {result_path}',
  test_command: '/abs/project/tools/test_candidate.sh {candidate_path} > {result_path}',
  benchmark_command: '/abs/project/tools/bench_candidate.sh {candidate_path} > {result_path}',
  disassemble_command: '/abs/project/tools/disasm_candidate.sh {artifact_path} --ptx --sass > {result_path}',
  profile_command: '/abs/project/tools/profile_candidate.sh {artifact_path} > {result_path}',
}})
```

## 候选状态

| 状态 | 含义 |
|---|---|
| `compile_error` | 候选无法编译。 |
| `incorrect` | 候选能编译但正确性失败。 |
| `hypothesis_not_realized` | 候选正确，但 PTX/SASS 没有出现预期指令 regex。 |
| `rejected` | 候选正确且指令验证通过，但性能提升不足。 |
| `accepted` | 候选正确、指令验证通过，且实测超过当前最佳。 |

## Fidelity 边界

这是 AKW 原创的工程 workflow，不是某篇论文的严格复现。它的关键机制是证据闭环：硬件事实和 GEMM signature 生成指令假设，compile/test/disassembly 验证假设，benchmark/profile 决定接受或拒绝。

它最适合 tensor-core GEMM。对于更广义的 compute-bound 任务，可以复用 evidence schema，但需要换成对应算子自己的 rule pack。
