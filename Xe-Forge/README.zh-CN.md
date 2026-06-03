# Xe-Forge: Intel XPU 多阶段 CoVeR 优化

**论文**: [Xe-Forge: LLM-driven Triton kernel optimization for Intel XPUs](https://github.com/intel/Xe-Forge)  
**发表**: GitHub Project (2024)

## 概述

Xe-Forge 实现了面向 Intel XPU（Arc Pro, Ponte Vecchio）的 **多阶段 CoVeR（Code-Verify-Refine）优化流程**。工作流通过 11 个硬序优化阶段进行，每个阶段运行嵌套的 CoVeR 循环，追踪最佳结果并设置 ≥2% 的改进门槛。

## 核心洞察

**多阶段递进与硬依赖关系**：早期阶段（ALGORITHMIC、DTYPE_FIX）必须在后期阶段（DEVICE_SPECIFIC、AUTOTUNING）之前完成。每个阶段运行 CoVeR 循环：生成候选 → 验证正确性和性能 → 追踪最佳。只有 ≥2% 的改进才能推进。这种结构防止过早优化，确保在设备特定调优之前建立坚实基础。

## 循环拓扑

```
11 阶段顺序流水线，带嵌套 CoVeR 循环：

阶段 1 (ALGORITHMIC):
  CoVeR 循环（最多 10 次迭代）：
    生成 → 验证 → 优化
    追踪最佳（≥2% 门槛）
  → 将最佳提升为阶段 2 基线

阶段 2 (DISCOVERY):
  CoVeR 循环...
  → 将最佳提升为阶段 3 基线

...

阶段 11 (VECTORIZATION):
  CoVeR 循环...
  → 最终最佳内核
```

**阶段顺序**（硬依赖）：
1. ALGORITHMIC（算法层）
2. DISCOVERY（探索）
3. DTYPE_FIX（数据类型修正）
4. FUSION（算子融合）
5. MEMORY_ACCESS（内存访问）
6. BLOCK_POINTERS（块指针）
7. PERSISTENT_KERNEL（持久化内核）
8. DEVICE_SPECIFIC（设备特定）
9. AUTOTUNING（自动调优）
10. POLISHING（代码打磨）
11. VECTORIZATION（向量化）

## 优化阶段

1. **ALGORITHMIC**: 高层算法选择、数据流优化
2. **DISCOVERY**: 广泛探索优化空间
3. **DTYPE_FIX**: 数据类型选择（fp32/fp16/bf16）、混合精度
4. **FUSION**: 算子融合、epilogue 融合
5. **MEMORY_ACCESS**: 内存合并、bank 冲突消除
6. **BLOCK_POINTERS**: Triton 块指针优化
7. **PERSISTENT_KERNEL**: 流式工作负载的持久化内核模式
8. **DEVICE_SPECIFIC**: Intel XPU 特定优化（XMX、EU）
9. **AUTOTUNING**: tile 大小、block 维度、启动参数
10. **POLISHING**: 最终代码清理、小幅调整
11. **VECTORIZATION**: SIMD 优化、向量指令使用

## 硬件目标

- **Intel Arc Pro**: A40, A50, A60
- **Intel Data Center GPU Max**: Ponte Vecchio
- **特性**: XMX（Xe 矩阵扩展）、EU（执行单元）、HBM

## 反馈信号

- **speedup**（越高越好）：相对基线的加速比
- **correctness**（门槛）：数值误差阈值
- **improvement_gate**（≥2%）：接受候选的最小改进
- **eu_occupancy**: 执行单元利用率
- **xmx_utilization**: Xe 矩阵扩展使用率

## 典型结果

- **加速比**: 1.5x - 5x（相对朴素 Triton 基线）
- **每阶段迭代数**: 5-10
- **总运行时间**: 30-90 分钟（取决于内核复杂度）

## 使用示例

```javascript
// 在 Claude Code 中：
/workflow xe-forge-kernel-optimization

// 工作流将：
// 1. 初始化 Intel XPU + Triton 环境
// 2. 加载基线 Triton 内核
// 3. 对于每个阶段（ALGORITHMIC → VECTORIZATION）：
//    a. 运行 CoVeR 循环：生成 → 验证 → 优化
//    b. 追踪阶段最佳（≥2% 改进门槛）
//    c. 将最佳提升为下一阶段基线
// 4. 返回优化内核及逐阶段细分
```

## 关键参数

- **max_iterations_per_stage**: 每阶段最大 CoVeR 循环迭代数（默认: 10）
- **improvement_threshold**: 接受候选的最小改进（默认: 2.0%）
- **correctness_tolerance**: 数值误差阈值（默认: 1e-5 绝对, 1e-3 相对）
- **enable_vtune**: 启用 VTune 性能分析（默认: false，开销大）

## 注意事项

- **硬阶段顺序**: 早期阶段为后期优化奠定基础
- **CoVeR 循环**: 生成 → 验证 → 优化（不只是生成 → 接受）
- **最佳追踪**: 保留所有迭代中的最佳，而非仅最后一个
- **Intel XPU 特定**: 利用 XMX、EU 架构、HBM 带宽
- **Triton 作为 IR**: 比 CUDA 更高层，编译为 Intel GPU 的 SPIR-V
- **VTune 集成**: 可选的详细瓶颈分析

## 相关工作流

- **KernelBlaster**: 基于 MAIC-RL 的知识库方法
- **KernelFoundryDx**: 诊断驱动的多岛进化（Triton）
- **GPU Forecasters**: 带学习加速预测的 PUCT 搜索
