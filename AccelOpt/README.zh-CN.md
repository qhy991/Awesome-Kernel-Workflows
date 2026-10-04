# AccelOpt 内核优化工作流

[English](README.md) · **简体中文**

基于 Nsight Compute（NCU）的自改进 CUDA 内核优化工作流，实现 [AccelOpt](https://arxiv.org/abs/2511.15915) 方法。

## 核心循环

`Plan → Execute → Profile → Summarize → Accumulate Experience → Repeat`

## 适用场景

- 需要基于真实 profiling 数据做优化（避免“拍脑袋”）
- 希望保留候选者池（beam），而不是只保留一个最优解
- 希望从慢/快样本对中沉淀可复用经验模式

## 入口文件

- 工作流脚本：`accelopt-kernel-optimization.js`
- 英文完整文档：`README.md`

## 论文

- [AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization](https://arxiv.org/abs/2511.15915)

### Native 任务自测资格

可选 `task_result_command`、`task_workload_count` 启用冻结任务 Python/CuTe 路径。快照明确选择的文件，经 broker 执行完整官方测试并归约匹配的原始任务/终态记录。只有完整正确且实测的结果可评分；NCU 仅诊断。保留有限原生拓扑及默认值。CUDAAgent/AccelOpt 比较本轮自行生成初始实现；KernelFoundry/KSearch/CUDALLM 报告 official reference 分数。返回实测快照，不重写显示源码。Native agent JSON 传输须核对原始文件及外部终验，不冒充 process Host evaluate 回执或正式消融权威。
