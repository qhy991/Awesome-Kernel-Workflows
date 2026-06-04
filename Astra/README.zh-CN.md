# Astra Workflow

多 Agent CUDA kernel 优化工作流，改编自：

- 论文：[Astra: A Multi-Agent System for GPU Kernel Performance Optimization](https://arxiv.org/abs/2509.07506)
- 仓库：[Anjiang-Wei/Astra](https://github.com/Anjiang-Wei/Astra)
- Workflow 文件：[`astra-kernel-optimization.js`](astra-kernel-optimization.js)

[English](README.md) · **简体中文**

---

## 它做什么

`astra-kernel-optimization` 通过专门 agent 角色优化已有 CUDA kernel：

1. Setup 读取初始 CUDA kernel 和导出/调用 contract
2. Testing Agent 准备正确性测试和 shape 覆盖
3. Profiling Agent 测量 baseline 延迟和瓶颈
4. Planning Agent 基于证据提出单一优化方向
5. Coding Agent 修改当前最佳 kernel
6. Testing + Profiling 用真实 evaluator 结果验证候选
7. Record 将测量到的经验传给下一轮
8. PostProcess 生成回接生产代码的说明

该 workflow 面向生产风格 CUDA kernel，尤其适合 SGLang 类已有 CUDA/PyBind 实现；它不是从 PyTorch module spec 重新生成 kernel。

---

## 参数

主要参数：

- `kernel_path`：初始 CUDA kernel 文件路径
- `compare_kind`：比较模式，例如 `generic`、`rmsnorm`、`silu`、`mergestate`
- `baseline_module`：包含 baseline 函数的 Python 模块，默认 `sgl_kernel`
- `baseline_func`：baseline 函数名
- `generated_export_func`：生成 PyBind 模块中期望导出的函数名
- `test_command`：正确性测试命令，使用 `{kernel_path}` 和 `{result_path}`
- `benchmark_command`：性能测试命令，使用 `{kernel_path}` 和 `{result_path}`
- `iterations`：优化轮数，默认 `5`
- `target_gpu`：目标 GPU，默认 `H100`
- `exp_dir`：实验输出目录

Evaluator JSON 建议格式：

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

## 调用示例

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

## 输出

Workflow 返回：

- `best_speedup`
- `best_runtime_ms`
- `baseline_runtime_ms`
- `best_kernel_code`
- `run_log`
- `lessons`
- `post_process`
- `report`

---

## 引用

```bibtex
@inproceedings{wei2025astra,
  title={Astra: A Multi-Agent System for GPU Kernel Performance Optimization},
  author={Wei, Anjiang and Sun, Tianran and Seenichamy, Yogesh and Song, Hang and Ouyang, Anne and Mirhoseini, Azalia and Wang, Ke and Aiken, Alex},
  booktitle={NeurIPS 2025 Fourth Workshop on Deep Learning for Code},
  year={2025}
}
```
