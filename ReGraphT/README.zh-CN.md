# ReGraphT Workflow

基于 CUDA Reasoning Graph 的图搜索优化工作流，改编自：

- 论文：[From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph](https://arxiv.org/abs/2510.19873)
- Workflow 文件：[`regrapht-kernel-optimization.js`](regrapht-kernel-optimization.js)

[English](README.md) · **简体中文**

---

## 它做什么

`regrapht-kernel-optimization` 将 ReGraphT 方法落地为可由 agent 执行的 CUDA 优化循环。

它不训练小模型，而是采用论文中的 training-free 推理思路：

1. 读取 CUDA 任务与 evaluator contract
2. 从优化轨迹构建或加载 CUDA Reasoning Graph
3. 用 Monte Carlo Graph Search 选择优化方法路径
4. 基于选中的方法与示例生成 CUDA candidate
5. 编译、正确性测试、benchmark
6. 用真实测量结果更新 graph reward 与示例
7. 输出最佳 kernel 与后续可探索路径

---

## 推荐场景

- 你已有 LLM CUDA 优化轨迹，希望把它们复用为结构化优化经验
- 希望本地/小模型沿着多步 CUDA reasoning 路径生成代码
- 不想只做普通代码 RAG，而是希望按优化方法图进行检索和搜索
- 希望失败样本回写到方法图中，而不是只留在日志里

---

## 参数

主要参数：

- `source_code_path`：CUDA/C++ 源码或任务 spec 路径
- `eval_command`：评测命令，需支持 `{kernel_path}` 和 `{result_path}` 占位符
- `op_description`：算子/任务描述
- `trace_corpus_path`：可选，LLM 优化轨迹 JSONL
- `graph_path`：可选，持久化 CUDA Reasoning Graph
- `baseline_command`：可选，baseline 评测命令
- `budget`：candidate 尝试次数，默认 `20`
- `rollouts_per_select`：每次选择的 MCGS rollout 数，默认 `12`
- `exploration_weight`：UCT 探索权重，默认 `1.4`
- `max_path_length`：最大 graph 路径长度，默认 `4`
- `target_gpu`：目标 GPU，默认 `H100`
- `exp_dir`：实验输出目录

Evaluator JSON 建议格式：

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

## 调用示例

```javascript
Workflow({
  name: 'regrapht-kernel-optimization',
  args: {
    source_code_path: '/path/to/source.cu',
    op_description: 'Sequential stencil kernel to CUDA',
    eval_command: 'python eval.py --kernel {kernel_path} --json {result_path}',
    trace_corpus_path: '/path/to/llm_optimization_traces.jsonl',
    graph_path: '/path/to/regraph.json',
    budget: 20,
    rollouts_per_select: 12,
    exploration_weight: 1.4,
    max_path_length: 4,
    target_gpu: 'H100',
    exp_dir: '/tmp/regrapht_exp',
  },
})
```

---

## 输出

Workflow 返回：

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

## 引用

```bibtex
@article{gong2025regrapht,
  title={From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph},
  author={Gong, Junfeng and Wei, Zhiyi and Chen, Junying and Liu, Cheng and Li, Huawei},
  journal={arXiv preprint arXiv:2510.19873},
  year={2025},
  url={https://arxiv.org/abs/2510.19873}
}
```
