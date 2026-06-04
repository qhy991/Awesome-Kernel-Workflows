# StitchCUDA: 三智能体编排的 CUDA 内核合成

**论文**: [StitchCUDA: Three-agent orchestration for CUDA kernel synthesis](https://arxiv.org/abs/2603.02637)  
**发表**: arXiv preprint (2024)

## 概述

StitchCUDA 实现了 **三智能体编排框架**用于 CUDA 内核合成：**规划器 → 编码器 → 验证器**，并带自适应重规划。规划器生成战略优化计划，编码器将其实现为 CUDA 内核，验证器检查正确性和性能。失败模式触发带有替代策略的重规划。

## 核心洞察

**关注点分离与自适应重规划**：通过将优化分为战略规划（规划器）、实现（编码器）和验证（验证器），每个智能体专注于其领域。自适应重规划从失败模式中学习：连续编译失败 → 针对编译问题重规划；连续正确性失败 → 针对算法正确性重规划；性能停滞 → 用替代优化方法重规划。这种结构避免陷入局部最优。

## 循环拓扑

```
顺序三智能体流水线，带自适应重规划：

规划器（PLANNER）:
  生成优化计划
  (策略、步骤、线程配置)
  → 计划（JSON）

编码器（CODER）:
  将计划实现为 CUDA 内核
  → CUDA 内核代码

验证器（VERIFIER）:
  编译检查（nvcc）
  正确性检查（数值）
  性能检查（基准测试）
  → 验证结果

自适应重规划（条件触发）:
  如果 2 次连续编译失败
    → 重规划，关注编译问题
  如果 2 次连续正确性失败
    → 重规划，关注算法正确性
  如果 3 次迭代性能停滞（CV < 5%）
    → 用替代优化方法重规划

重复最多 max_attempts 次（默认: 20）
追踪所有尝试中的最佳内核
```

## 组件

### 智能体 1: 规划器

**角色**: 战略优化规划

**职责**:
- 高层优化方法（内存受限、计算受限、平衡）
- 分解为实现步骤
- 指定线程/块配置
- 识别关键优化

**输出**: JSON 计划，包含:
- `strategy`: "memory-bound" | "compute-bound" | "balanced"
- `iterations`: 实现步骤数组
- `threading_config`: `{block_size, grid_size, threads_per_block}`
- `key_optimizations`: 优化名称数组（如 "coalescing", "shared_memory", "register_blocking"）

### 智能体 2: 编码器

**角色**: CUDA 内核代码生成

**职责**:
- 遵循计划生成完整 CUDA 内核
- 实现计划中的所有步骤
- 应用关键优化（合并、共享内存、寄存器分块、循环展开等）
- PyTorch `load_inline` 兼容格式

**输出**: 完整 CUDA 内核代码 + 主机启动代码

### 智能体 3: 验证器

**角色**: 正确性和性能验证

**职责**:
- **编译检查**: nvcc 编译、语法验证、资源使用
- **正确性检查**: 相对参考实现的数值精度
- **性能检查**: 基准测试执行时间，使用 nsys/ncu 分析（可选）
- **KernelBench 评估**: 标准化评估框架

**输出**: 验证结果:
- `compilation_success`: 布尔值
- `correctness_passed`: 布尔值
- `speedup`: 浮点数（相对基线）
- `performance_gflops`: 浮点数
- `errors`: 错误消息数组

### 自适应重规划

**触发条件**:

1. **N 次连续编译失败** (N=2)
   - **动作**: 重规划，关注编译问题
   - **策略**: 简化代码、避免复杂模板、检查资源限制

2. **N 次连续正确性失败** (N=2)
   - **动作**: 重规划，关注算法正确性
   - **策略**: 审查算法、添加验证检查、简化逻辑

3. **性能停滞** (M=3 次迭代，CV < 5%)
   - **动作**: 用替代优化方法重规划
   - **策略**: 从内存受限切换到计算受限（或反之），探索不同的线程配置

**重规划流程**:
1. 诊断失败根本原因
2. 生成替代方法
3. 避免先前的失败模式
4. 重规划后重置失败计数器

## 硬件目标

- **NVIDIA GPU**: CUDA 兼容（sm_80, sm_89, sm_90）
- **架构**: NVIDIA A100, L40S, H100
- **特性**: CUDA Compute Capability 8.0+, 共享内存, 寄存器, Tensor Cores（可选）

## 反馈信号

- **compilation_success**（门槛）：内核必须无错误编译
- **correctness_passed**（门槛）：相对参考的数值正确性
- **speedup**（越高越好）：相对基线的加速比（阈值: 1.0x）
- **performance_gflops**（越高越好）：持续吞吐量

## 典型结果

- **加速比**: 1.5x - 5x（相对基线）
- **尝试次数**: 10-20（取决于 max_attempts）
- **运行时间**: 20-60 分钟（取决于 max_attempts）

## 使用示例

```javascript
// 在 Claude Code 中：
/workflow stitchcuda-kernel-optimization

// 工作流将：
// 1. 初始化 CUDA + KernelBench 环境
// 2. 对于每次尝试（最多 max_attempts）：
//    a. 规划器生成优化计划
//    b. 编码器生成 CUDA 内核
//    c. 验证器检查编译、正确性、性能
//    d. 如果失败触发重规划：调整策略
// 3. 追踪所有尝试中的最佳内核
// 4. 返回最佳内核及报告
```

## 关键参数

- **max_attempts**: 最大合成尝试次数（默认: 20）
- **compile_failure_threshold**: 触发重规划的连续编译失败次数（默认: 2）
- **correctness_failure_threshold**: 触发重规划的连续正确性失败次数（默认: 2）
- **stagnation_threshold**: 性能停滞的迭代次数（默认: 3）
- **stagnation_cv_threshold**: 停滞检测的变异系数（默认: 5%）
- **early_termination_speedup**: 触发早期终止的加速比（默认: 2.0x）

## 注意事项

- **三智能体专业化**: 战略规划、代码生成、验证
- **自适应重规划**: 从失败模式中学习
- **快速迭代**: PyTorch `load_inline` 实现快速编译-测试循环
- **KernelBench 集成**: 标准化评估框架
- **失败计数器**: 追踪连续编译/正确性失败
- **停滞检测**: 变异系数 < 5%
- **早期终止**: 达到出色性能时停止（≥2x 加速比）

## 相关工作流

- **FACT**: 带模式发现的组合式内核合成
- **KernelBlaster**: 基于 MAIC-RL 的知识库方法
- **GPU Forecasters**: 带学习加速预测的 PUCT 搜索
