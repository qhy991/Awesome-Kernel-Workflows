# CUDA-LLM FSR Workflow

基于 Feature Search and Reinforcement 的 CUDA kernel 生成工作流，改编自：

- 论文：[CUDA-LLM: LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092)
- Workflow 文件：[`cudallm-fsr-kernel-generation.js`](cudallm-fsr-kernel-generation.js)

[English](README.md) · **简体中文**

---

## 它做什么

`cudallm-fsr-kernel-generation` 通过显式搜索 CUDA 优化特征生成 kernel，并用真实测量结果强化 feature 选择。

循环如下：

1. 读取 task spec 与 reference implementation
2. 构建 CUDA feature catalog
3. 生成多样正确性和边界测试
4. 选择兼容 feature 组合
5. 生成 CUDA candidate
6. 编译、正确性测试、benchmark
7. 用 compile/correctness/speedup reward 更新 feature scores
8. 输出最佳 kernel 和 feature reward table

这不是模型训练，而是适合 Claude Code Workflow 的 agent-executable FSR 循环。

---

## 参数

主要参数：

- `problem_path`：CUDA 生成任务说明路径
- `reference_code_path`：可选 reference implementation
- `benchmark_command`：评测命令，使用 `{kernel_path}` 和 `{result_path}`
- `target_gpu`：目标 GPU，默认 `H100`
- `iterations`：feature search 迭代轮数，默认 `8`
- `feature_budget`：每个 candidate 最多选择的 feature 数，默认 `4`
- `samples_per_feature_set`：每轮候选样本数，默认 `2`
- `rtol` / `atol`：正确性容差
- `exp_dir`：输出目录

Evaluator JSON 建议格式：

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

## 调用示例

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

## 输出

Workflow 返回：

- `best_speedup`
- `best_latency_ms`
- `best_kernel_code`
- `best_candidate_id`
- `feature_scores`
- `candidates`
- `report`

---

## 引用

```bibtex
@article{chen2025cudallm,
  title={CUDA-LLM: LLMs Can Write Efficient CUDA Kernels},
  author={Chen, Wentao and Zhu, Jiace and Fan, Qi and Ma, Yehan and Zou, An},
  journal={arXiv preprint arXiv:2506.09092},
  year={2025},
  url={https://arxiv.org/abs/2506.09092}
}
```

### Native 任务自测资格

可选 `task_result_command`、`task_workload_count` 启用冻结任务 Python/CuTe 路径。快照明确选择的文件，经 broker 执行完整官方测试并归约匹配的原始任务/终态记录。只有完整正确且实测的结果可评分；NCU 仅诊断。保留有限原生拓扑及默认值。CUDAAgent/AccelOpt 比较本轮自行生成初始实现；KernelFoundry/KSearch/CUDALLM 报告 official reference 分数。返回实测快照，不重写显示源码。Native agent JSON 传输须核对原始文件及外部终验，不冒充 process Host evaluate 回执或正式消融权威。
