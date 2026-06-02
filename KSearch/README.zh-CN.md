# K-Search Workflow

面向 GPU kernel 优化的世界模型树搜索工作流，基于 K-Search 论文方法改写为 Claude Code workflow：

- 论文：[K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model](https://arxiv.org/abs/2602.19128)
- Workflow 文件：[`ksearch-kernel-optimization.js`](ksearch-kernel-optimization.js)

[English](README.md) · **简体中文**

---

## 它在做什么

`ksearch-kernel-optimization` 适用于设计空间大、决策耦合强的 kernel 优化场景。相比单路径迭代，它通过维护一个持续演化的 **决策树（world model）** 来系统探索策略分支。

核心循环如下：

1. Setup（读取问题与基线）
2. Initialize（初始化世界模型决策树）
3. Select（选择最高价值前沿 action）
4. Generate / Debug / Improve（生成或迭代代码）
5. Evaluate（正确性与性能评测）
6. Refine / Backtrack（成功则扩展、失败则回退）
7. Report（输出搜索轨迹与最优解）

---

## 推荐使用场景

- 复杂算子（如 MLA attention、MoE routing、融合算子）
- 存在多个正交设计维度（tile、memory hierarchy、并行拆分、算法变体）
- 需要显式回溯与分支探索，避免陷入局部最优

---

## 参数说明

该 workflow 的主要参数：

- `kernel_spec_path`：问题定义文件路径（必填）
- `op_description`：算子描述
- `language`：`triton | cuda | python`
- `target_gpu`：目标 GPU（如 `H100`）
- `max_cycles`：最大搜索轮数（默认 `10`）
- `attempts_per_cycle`：每轮生成/改进次数（默认 `5`）
- `stagnation_window`：连续无改进提前结束窗口（默认 `3`）
- `max_difficulty`：优先探索的最大 action 难度（默认 `4`）
- `bench_command`：性能评测命令
- `baseline_code_path`：可选，基线 kernel 路径
- `rtol` / `atol`：正确性容差
- `exp_dir`：实验输出目录

详细执行细节请直接查看 [`ksearch-kernel-optimization.js`](ksearch-kernel-optimization.js)。

---

## 调用示例

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

## 输出结果

工作流返回以下关键结果：

- `best_metric`
- `best_solution_code`
- `cycles_completed`
- `solutions_evaluated`
- `decision_tree`（最终世界模型状态）
- `solution_lineage`（优解路径）
- `report`
- `baseline_metric`
- `speedup_over_baseline`

---

## 引用

如果你在论文、技术报告或对外分享中使用了该 workflow，请引用 K-Search 原论文：

```bibtex
@article{cao2026ksearch,
  title={K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model},
  author={Cao, Shiyi and Mao, Ziming and Gonzalez, Joseph E. and Stoica, Ion},
  journal={arXiv preprint arXiv:2602.19128},
  year={2026},
  url={https://arxiv.org/abs/2602.19128}
}
```
