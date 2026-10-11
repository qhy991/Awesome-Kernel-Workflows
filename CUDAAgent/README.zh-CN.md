# CUDA Agent 工作流

[English](README.md) · **简体中文**

多轮技能化 CUDA 内核优化流程，实现 [CUDA Agent](https://arxiv.org/abs/2602.24286) 的推理时 agent 循环：profile → implement → verify → refine。

## 核心特点

原生 `task_result_command` 路径只有在全部声明 seed 都获得完整、source-bound 的 `candidate_failure` 后，才返回有限终态 `status`/`outcome_state = no_valid_candidate`、`stop_reason`/`termination_reason = seed_exhausted`。结果保留每个声明槽及完整原始结果，按实测汇总编译状态，最佳源码和性能指标为空；不再派发工作空间初始化、profiling、refinement 或替补 seed。证据缺失、unknown、权限和基础设施错误仍 HOLD；内核验收、模型身份与资源释放继续由外部证据判断。

- 多文件工作空间（`kernel.cu` / `kernel_binding.cpp` / `model_new.py`）
- 迭代式修复与优化，直到达到相对 `torch.compile` 的加速目标
- 采用离散奖励里程碑，降低 reward hacking 风险

## 入口文件

- 工作流脚本：`cuda-agent-kernel-optimization.js`
- 英文完整文档：`README.md`

## 论文

- [CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation](https://arxiv.org/abs/2602.24286)

### Native 任务自测资格

可选 `task_result_command`、`task_workload_count` 启用冻结任务 Python/CuTe 路径。快照明确选择的文件，经 broker 执行完整官方测试并归约匹配的原始任务/终态记录。只有完整正确且实测的结果可评分；NCU 仅诊断。保留有限原生拓扑及默认值。CUDAAgent/AccelOpt 比较本轮自行生成初始实现；KernelFoundry/KSearch/CUDALLM 报告 official reference 分数。返回实测快照，不重写显示源码。Native agent JSON 传输须核对原始文件及外部终验，不冒充 process Host evaluate 回执或正式消融权威。
