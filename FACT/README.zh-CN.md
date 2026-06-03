# FACT: 组合式内核合成框架

**论文**: [FACT: Compositional kernel synthesis framework](https://github.com/Project-FACT/FACT)  
**发表**: GitHub Project (2024)

## 概述

FACT 实现了 **三阶段组合式合成流水线**：模式发现 → 模式实现 → 模式组合。工作流从示例内核中学习优化模式，将其实现为索引在模式注册表中的 CUTLASS 代码转换，并通过消融验证将其组合为优化内核。

## 核心洞察

**模式注册表与组合式合成**：FACT 不采用单体优化，而是将过程分解为发现可重用模式（tiling、memory、compute、fusion、layout），将其索引为 `T(rule_type, dtype, architecture)`，并系统地组合它们。消融研究验证每个模式的贡献。这实现了**迁移学习**（在一个内核上发现的模式可应用于其他内核）和**可解释性**（理解哪些模式重要）。

## 循环拓扑

```
三阶段顺序流水线：

发现（DISCOVERY）:
  分析示例内核
  提取优化模式
  → 模式候选（REUSE, ADAPT, NEW）

实现（REALIZATION）:
  对于每个模式：
    生成 CUTLASS 代码模板
    定义转换参数
    指定适用性约束
    创建依赖图
  → 已实现的模式注册表

组合（COMPOSITION）:
  选择组合策略（贪心、依赖感知、基于搜索）
  生成模式组合
  按约束过滤
  调优参数
  编译和评估
  → 组合内核候选

消融（ABLATION）:
  选择 top-K 内核
  创建消融变体（留一法、留组法）
  测量性能影响
  → 模式贡献分析

报告（REPORT）:
  最佳内核 + 模式分解
```

## 组件

### 阶段 1: 模式发现

**目标**: 从示例内核中提取优化模式

**策略**:
- **REUSE**: 现有模式直接适用
- **ADAPT**: 现有模式需修改
- **NEW**: 从头创建新模式

**模式类型**:
- **Tiling**: 块/线程 tiling 策略
- **Memory**: 内存层次优化（共享内存、寄存器）
- **Compute**: 计算模式（Tensor Core 使用、指令调度）
- **Fusion**: 算子融合（epilogue、多阶段）
- **Layout**: 数据布局转换

### 阶段 2: 模式实现

**目标**: 将模式实现为具体的 CUTLASS 代码转换

**输出**: 模式注册表 `T(rule_type, dtype, architecture)`

**CUTLASS 层**:
- **Tile**: 块级 tiling 策略
- **Kernel**: warp 级计算模式
- **Grid**: 线程块配置

**验证**:
- 语法正确性
- 类型安全
- 资源约束（共享内存、寄存器）

### 阶段 3: 模式组合

**目标**: 将模式组合为优化内核

**组合策略**:
1. **贪心**: 迭代添加高影响模式
2. **依赖感知**: 尊重模式依赖关系
3. **基于搜索**: 系统探索模式空间

**约束**:
- 资源限制（共享内存、寄存器）
- 模式兼容性（依赖、冲突）
- 架构特定约束

### 阶段 4: 消融

**目标**: 验证模式贡献

**消融配置**: 2^N + 1（N 个模式 + 基线）

**方法**:
- **留一法**: 单独移除每个模式
- **留组法**: 移除相关模式组
- **性能影响**: 测量加速比损失

## 硬件目标

- **NVIDIA A100, H100**
- **架构**: sm_80, sm_89, sm_90
- **特性**: Tensor Cores, HBM2e/HBM3, CUDA Compute Capability 8.0+

## 反馈信号

- **speedup**（越高越好）：相对 PyTorch 基线的加速比
- **gflops**（越高越好）：持续吞吐量
- **correctness**（门槛）：数值正确性
- **tensor_core_utilization**（越高越好）：Tensor Core 使用率 %
- **memory_bandwidth_utilization**（越高越好）：HBM 带宽使用率 %

## 典型结果

- **加速比**: 2x - 10x（相对 PyTorch 基线）
- **发现模式数**: 10-20
- **组合内核数**: 20-50
- **运行时间**: 60-120 分钟（取决于模式数量）

## 使用示例

```javascript
// 在 Claude Code 中：
/workflow fact-kernel-optimization

// 工作流将：
// 1. 分析示例内核（如 CUTLASS 示例、PyTorch 实现）
// 2. 发现优化模式（tiling、memory、compute、fusion、layout）
// 3. 将模式实现为 CUTLASS 代码转换
// 4. 将模式组合为内核候选
// 5. 运行消融研究以验证贡献
// 6. 返回最佳内核及模式分解
```

## 关键参数

- **exemplar_kernels**: 示例实现路径（CUTLASS 示例、PyTorch）
- **composition_strategy**: "greedy" | "dependency-aware" | "search-based"
- **max_patterns**: 组合的最大模式数（默认: 10）
- **ablation_top_k**: 消融的 top 内核数（默认: 5）
- **cutlass_version**: CUTLASS 版本（默认: "3.x"）

## 注意事项

- **模式注册表是关键**: `T(rule_type, dtype, architecture)` 索引转换
- **示例驱动**: 从高性能参考中学习
- **组合式**: 从简单模式构建复杂优化
- **消融验证**: 确保模式有意义地贡献
- **CUTLASS 特定**: 利用 CUTLASS 3.x API（Tile/Kernel/Grid）
- **监督检查点**: 可选的人工参与审批门
- **迁移学习**: 在一个内核上发现的模式可应用于其他内核

## 相关工作流

- **StitchCUDA**: 三智能体编排的 CUDA 内核合成
- **KernelFoundryDx**: 诊断驱动的多岛进化（Triton）
- **Xe-Forge**: Intel XPU 多阶段 CoVeR 优化
