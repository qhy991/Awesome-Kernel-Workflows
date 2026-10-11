# AccelOpt 内核优化工作流

[English](README.md) · **简体中文**

基于 Nsight Compute（NCU）的自改进 CUDA 内核优化工作流，实现 [AccelOpt](https://arxiv.org/abs/2511.15915) 方法。

## 核心循环

原生 `task_result_command` 路径维护独立的失败证据索引：候选源文件路径与哈希、完整任务结果路径及编译/正确性类型化结论。下一轮 planner 和 implementer 必须读取原文件及其关联诊断；失败候选不进入性能 Learn 样本对，也不获得虚构的 slowdown 分数。原 beam、学习阈值、seed 数量与搜索预算保持不变。最优候选改变后，初始化 profile 明确标为历史证据，不能冒充新候选的实测指标。

原生路径在完整迭代结束后原子写入 `exp_dir/checkpoint.json`，保存 beam 和源绑定、初始测量、失败索引、经验、已完成轮数及原始参数。可选 `termination_file`（非空请求停止）或 `deadline_epoch`（Unix 秒）也可为旧路径启用此检查点。停止在本轮评测和学习完成后生效，返回实际 `iterations_completed`、`termination_reason`、`checkpoint_path` 并跳过额外报告 agent。这是同一个 native Workflow activation 的审计快照；恢复仍沿原 activation 的重放身份与预算，不提供独立 checkpoint loader，也不允许新 activation 通过复制检查点自动恢复或增加预算。

只有全部声明的原生 seed 都获得完整、source-bound 的 `candidate_failure`，才返回有限终态 `status`/`outcome_state = no_valid_candidate`、`stop_reason`/`termination_reason = seed_exhausted`。保留每个声明槽和完整 canonical 原始结果，不生成最佳内核或性能指标，不再派发 planner、profiler 或补充 seed。证据缺失、权限失败与 unknown 仍然 HOLD；该终态不代表内核合格或资源已释放。

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
