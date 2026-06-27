# AscendC 内核优化

[English](README.md) · **简体中文**

面向昇腾 910B NPU 的**权威 Ascend / AscendC 内核优化 workflow**。它是
Ascend 原生的（不是把 CUDA workflow 换个标签）：使用 **AscendC** 语言，用
**msprof**（CANN）做 profiling，并把编译 + 正确性 + NPU 计时统一委托给
MultiKernelBench 的 `ascendc_direct_launch` runner（通过 substrate 的
[ascend backend](../_substrate/backends/ascend/)）。

> 解决 [issue #16](https://github.com/qhy991/Awesome-Kernel-Workflows/issues/16)：
> 此前 catalog 没有可用的 Ascend workflow，所有 Ascend 任务都在**workflow 选型
> 阶段（第 1 轮）STALL**，必须演化出一个仅在当前 session 有效的变体，且无法跨
> session 复用。本 workflow 就是那个经过验证的变体，已被提升为 upstream 并泛化。

## 来源 / 保真度

派生自 session 本地变体 `generalist-ascend-optimization`，在 910b-exp 的
`20260622-160108` / `20260623-{194914,190425,191018}` 等 session 中演化并验证
（见 `AKW-Exp/910b-exp/KEY-ISSUES.md §B`）。已去除与具体任务（batched matmul）
耦合的提示，保留权威的 `ascendc_direct_launch` 提交契约与 substrate ascend
backend 驱动。保真度边界：**substrate_reference** —— 真正的测量信号由
`eval_single_runner.py` 拥有，而非 prompt。

## 工作流程

```
Setup      → 读参考算子 + ascend backend manifest；断言 language=ascendc
Generate   → 生成 BREADTH 个 AscendC 候选，写成 ascendc_direct_launch JSON
             （kernel.cpp + pybind11.cpp + ModelNew.py），逐文件用 Bash 写入
Evaluate   → substrate ascend run.sh：bisheng+cmake 编译 + 正确性 + NPU 计时
Optimize   → 每一轮：msprof 当前最优 → 选一个 AscendC 改进点
             （Cube/MTE 流水、FRACTAL_NZ 布局、tiling 贴合 L1/L0、dispatch 折叠）
             实现 → 评测一次 → RETURN（turn 边界）
Report     → 最优 kernel + 收敛状态
```

**循环控制**：达到 `target_speedup`（`converged`）/ 停滞（连续 2 轮 < 2% 改进
→ `stalled`）/ token 预算耗尽（`budget_exhausted`）即停止。

## 证据契约

编译 + 正确性 + 计时耦合在 MultiKernelBench 的 `eval_single_runner.py` 内（由
`_substrate/backends/ascend/run.sh` 调用）。结果 JSON 含
`{compiled, correct, candidate_latency_ms, eager_latency_ms,
compile_latency_ms, claimed_speedup}`。`correct:false` 时 speedup 下限为 1.0。
**绝不伪造**指标 —— 评测失败就如实上报 `compiled=false` / `correct=false` 及
真实错误。

## 鲁棒性脚手架（issue #17）

- **`agentRetry` + null 守卫**：每个 `agent()` 调用都包裹在有限次（5 次）重试
  中，每个被解引用的结果都做空值守卫，瞬态 API 429 或 agent 跳过不再让整个
  run 崩溃（这是 session `20260622-161357` 中最高杠杆的修复）。
- **turn 边界**：每个实现 turn 写文件 → 跑一次评测 → 读结果 → **返回**（避免
  出现 130 分钟的失控 turn）。
- **逐文件 Bash 写入**：AscendC 源文件逐个写入，而不是塞进一个 JSON 整包
  （整包路径会截断 / 超时 —— KEY-ISSUES §3 第 9 行）。
- **禁止操纵 harness**：禁止用 allocator / free-pool scrubbing 让 reference
  匹配（KEY-ISSUES §3 第 3 行）。

## 调用

```javascript
Workflow({ name: 'ascendc-kernel-optimization', args: {
  kernel_path: '/path/to/reference_kernel.cpp',   // 或用 problem_definition 生成
  op_description: '昇腾 910B2 上的 flash attention',
  substrate_dir: '/path/to/Awesome-Kernel-Workflows/_substrate',
  backend_dir:   '/path/to/Awesome-Kernel-Workflows/_substrate/backends/ascend',
  mkb_root:      process.env.MULTIKERNELBENCH_ROOT,
  op:            'flash_attention',
  exp_dir:       '/path/to/experiment/output',
  iterations: 3, breadth: 2, target_speedup: 1.5,
}})
```

依赖：bisheng（AscendC 编译器）、`torch_npu`、MultiKernelBench 评测 harness，
以及昇腾 NPU（msprof/CANN）。在没有 NPU 的主机上，substrate ascend 脚本会诚实
降级（exit 4 / `ok:false`），绝不伪造结果。

## 不适用的场景

不适用于 CUDA / Triton / ROCm / Metal 目标 —— 它们有各自的 profiler
（ncu / rocprof / metal-capture）与编译链。这些场景请在 catalog 中选择匹配的
workflow。
