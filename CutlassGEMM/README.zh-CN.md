# CutlassGEMM：基于 CUTLASS 的多配置调度 GEMM 内核优化

[English](README.md) · **简体中文**

使用 **NVIDIA CUTLASS** 设备级 API 实现的迭代式 GEMM 内核优化工作流，结合**多配置运行时调度**和 **StreamK 负载均衡**。面向 SOL-ExecBench 中 M 维度可变的 GEMM 问题。

## 概述

CutlassGEMM 弥补了静态 CUTLASS 模板实例化与 cuBLAS 逐问题自动调优之间的差距。cuBLAS 在运行时为每个 (M, N, K) 组合动态选择 tile 配置，而单一 CUTLASS 配置无法在所有 M 值上达到最优。本工作流通过以下方式解决这一问题：

1. **分析**问题的 M 范围分布，识别不同的性能区间
2. **生成**多个 CUTLASS `GemmUniversal` 实例化，每个使用不同的 tile 配置
3. **评测**每个工作负载的性能（通过 SOL-ExecBench）
4. **迭代调优**调度阈值和 tile 参数

### 核心创新：M 自适应多配置调度

```
M >= 512:  数据并行 GemmUniversal，大 tile（256x128x32），swizzle=8
           → SM 完全占满，调度器开销最小

M = 128-511: StreamK GemmUniversal，同样的大 tile
           → 当 CTA 数量不能被 SM 数整除时平衡负载

M < 128:  StreamK GemmUniversal，较小 tile（128x128x32），更多流水线级
           → 减少边缘浪费计算，更深的流水线隐藏延迟
```

### 性能结果（A800，问题 011_gemm_n28672_k4096，Llama 3.1 8B MLP）

| M 范围 | 相对 cuBLAS 加速比 | 策略 |
|--------|-------------------|------|
| M=128-256 | **1.05-1.09x** | StreamK + 大 tile |
| M=512-2053 | 0.95-1.06x | 数据并行 |
| M=8192 | 0.97x | 数据并行 |
| M<128 | 0.83-1.00x | StreamK + 小 tile |
| **全部 43 个工作负载** | **正确性通过** | |

## 使用方法

```javascript
Workflow({name: 'cutlass-gemm-optimization', args: {
  problem_dir: '/path/to/SOL-ExecBench/data/benchmark/.../problem_name',
  cutlass_dir: '/path/to/cutlass',
  sol_execbench_dir: '/path/to/SOL-ExecBench',
  output_dir: '/tmp/cutlass_gemm_opt',
  iterations: 3,
  gpu_arch: 'sm_80',
}})
```

### 直接评估（不使用工作流）

```bash
CUTLASS_DIR=/path/to/cutlass uv run sol-execbench \
  /path/to/problem_dir \
  --solution solution_example.json \
  --no-json -v
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `problem_dir` | (必需) | SOL-ExecBench 问题目录路径 |
| `cutlass_dir` | `/usr/local/cutlass` | CUTLASS 源码树路径（需含 `include/`） |
| `sol_execbench_dir` | (自动检测) | SOL-ExecBench 安装根目录 |
| `output_dir` | `/tmp/cutlass_gemm_opt` | 输出 solution 文件的目录 |
| `iterations` | `3` | 调优循环迭代次数 |
| `gpu_arch` | `sm_80` | 目标 GPU 计算能力 |

## 前置依赖

| 依赖 | 用途 |
|------|------|
| **Claude Code** | 工作流运行时 |
| **NVIDIA GPU** | sm_80+（A100/A800/H100） |
| **CUTLASS 3.x/4.x** | 模板库头文件 |
| **SOL-ExecBench** | 评估框架 |
| **PyTorch + CUDA** | 通过 cpp_extension 编译 |
| **uv** | Python 包管理器 |

## CUTLASS 布局映射（关键）

PyTorch 张量使用**行主序**存储。CUTLASS 的 `RowMajor` 和 `ColumnMajor` 必须根据运算映射正确设置：

| 运算 | A 形状 | B 形状 | CUTLASS LayoutA | CUTLASS LayoutB |
|------|--------|--------|-----------------|-----------------|
| C = A @ B | [M, K] | [K, N] | RowMajor | RowMajor |
| C = A @ B.T | [M, K] | [N, K] | RowMajor | **ColumnMajor** |
| C = A.T @ B | [K, M] | [K, N] | ColumnMajor | RowMajor |

## 优化调节项

| 调节项 | 选项 | 影响 |
|--------|------|------|
| ThreadblockShape | 256x128x32, 128x256x32, 128x128x32 | 每 SM 的 CTA 数、寄存器压力 |
| WarpShape | 64x64x32, 32x64x32 | Warp 级并行度 |
| 流水线级数 | 3-10 | 延迟隐藏 vs 共享内存占用 |
| Swizzle 策略 | Identity(1,2,4,8), StreamK | L2 局部性 vs 负载均衡 |
| Split-K | 1, 2, 4, 8 | 小 M × 大 K 时的并行度 |
| 累加器精度 | fp32（必需） | 正确性约束 |

## 与其他方法的对比

| 方面 | CutlassGEMM | cuBLAS | Triton 自动调优 |
|------|-------------|--------|-----------------|
| 调优方式 | 多配置 + 调度阈值 | 每次调用启发式选择 | JIT 网格搜索 |
| 运行时开销 | 零（静态调度） | ~5us/调用 | 首次 JIT 编译 |
| 灵活性 | 完全控制 tile/swizzle/流水线 | 黑盒 | 受限于 Triton IR |
| 峰值性能 | 0.95-1.09x cuBLAS | 1.0x（基线） | 0.8-1.0x cuBLAS |

## 参考文献

- [CUTLASS: Fast Linear Algebra in CUDA C++](https://github.com/NVIDIA/cutlass)
- [StreamK: Stream-K Parallel Decomposition for GEMM](https://arxiv.org/abs/2301.03598)
- [SOL-ExecBench: GPU Kernel Evaluation Framework](https://github.com/NVIDIA/SOL-ExecBench)
