# KernelSkill 工作流

基于双层记忆（dual-level memory）的多智能体 CUDA 内核优化工作流，改编自 KernelSkill 论文：

- 论文：[KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization](https://arxiv.org/abs/2603.10085)
- 参考实现：[github.com/0satan0/KernelMem](https://github.com/0satan0/KernelMem)（CC-BY-4.0）
- 工作流文件：[`kernelskill-kernel-optimization.js`](kernelskill-kernel-optimization.js)

[English](README.md) · **简体中文**

---

## 它做什么

KernelSkill 从 PyTorch 参考实现生成自定义 CUDA 内核，并在多轮中持续优化。其核心思想是：**用显式、外部化的"专家优化技能"取代 LLM 内部不透明的隐式启发式**——让"应用哪种优化"这一决策变得知识驱动且可审计。

这通过双层记忆实现：

- **长期记忆**——可跨任务复用的**技能库**，包含两部分：
  - **确定性决策策略**（`field_mapping`、`derived_fields`、headroom 分层、瓶颈优先级规则，以及映射到 `allowed_methods` 的决策表），从 profiling + 代码证据中*筛选*可行方法；
  - **方法知识**（原理 + 实现要点），仅在门控*之后*使用，使所选动作可解释。
- **短期记忆**——每个任务的**优化历史**（尝试过的方法 + 实测结果）与**修复链**记忆（避免循环修复）。

---

## 循环流程

1. **Setup**——读取 PyTorch 参考实现，测量 Torch eager 基线。
2. **Seed**——Generator 生成若干"正确性优先"的候选内核；Reviewer 评估；选出最佳有效种子作为工作内核。
3. **精炼轮次**（论文用 15 轮）——每轮是**双分支**控制流：
   - **修复分支**（内核无效）：`Diagnoser → Repairer`，依据修复链记忆，避免在相同错误变体间反复横跳。
   - **优化分支**（内核有效）：`特征提取 → 确定性门控(allowed_methods) → 检索方法知识 → Planner → Optimizer`，Planner 以短期优化记忆为条件。
4. **Report**——最佳内核、加速比轨迹、已应用技能、修复历史。

选择度量为 **speedup = 参考 (Torch eager) 延迟 / 测试延迟**；`ncu` 的内存/SM/占用率指标与 `nsys` 的内核启动次数驱动*选择哪种方法*。

---

## 推荐场景

- KernelBench 风格的算子/模型任务（Level 1–3），PyTorch → 自定义 CUDA。
- 当你希望方法选择**显式且可审计**（门控筛选的 `allowed_methods`），而非依赖模型的隐式判断时。
- 需要稳定多轮精炼（避免循环修复）的场景。

---

## 参数

主要参数：

- `reference_path`：PyTorch 参考任务路径（必填）
- `op_description`：操作描述
- `target_gpu`：目标 GPU（默认 `A100-80GB`）
- `rounds`：精炼轮数（默认 `15`）
- `seed_candidates`：前置生成的种子内核数（默认 `3`）
- `skill_library_path`：长期技能库 YAML/JSON 路径，**覆盖**内置库
- `bench_command`：编译 + benchmark 命令
- `ncu_binary` / `nsys_binary`：Nsight 工具路径
- `tol`：相对参考的最大误差容差
- `warmup` / `repeat`：benchmark 迭代次数
- `exp_dir`：输出目录
- `run_timestamp_iso`：报告元数据时间戳

内置的**技能库**（长期记忆）忠实于上游仓库的 `memorybank/bottleneck_headroom_kernelstructure.yaml`（NCU `field_mapping`、`derived_fields`、headroom 分层、决策表 → `allowed_methods`）。通过 `skill_library_path` 指向你自己的文件可跨任务扩展或替换。

---

## 调用示例

```javascript
Workflow({
  name: 'kernelskill-kernel-optimization',
  args: {
    reference_path: '/path/to/KernelBench/level1/19_ReLU.py',
    op_description: 'ReLU activation over a large tensor',
    target_gpu: 'A100-80GB',
    rounds: 15,
    seed_candidates: 3,
    bench_command: 'python utils/compile_and_run.py --kernel',
    tol: 0.01,
    exp_dir: '/tmp/kernelskill_exp',
  },
})
```

---

## 输出

- `baseline_latency_ms`——Torch eager 基线
- `best_speedup`——相对 eager 的最佳加速比
- `best_kernel_code`——最佳有效内核源码
- `rounds_completed`
- `skills_applied`——每轮方法 + 加速比
- `speedup_trajectory`——每轮后的最佳加速比
- `opt_memory`——短期优化记忆
- `report`——最终技术报告

---

## 引用

```bibtex
@article{sun2026kernelskill,
  title={KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization},
  author={Sun, Qitong and Han, Jun and Li, Tianlin and Tang, Zhe and Chen, Sheng and Yang, Fei and Liu, Aishan and Liu, Xianglong and Liu, Yang},
  journal={arXiv preprint arXiv:2603.10085},
  year={2026},
  url={https://arxiv.org/abs/2603.10085}
}
```
