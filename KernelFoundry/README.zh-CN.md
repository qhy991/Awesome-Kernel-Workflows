# KernelFoundry 工作流

[English](README.md) · **简体中文**

基于 MAP-Elites 的质量-多样性进化搜索工作流，实现 [KernelFoundry](https://arxiv.org/abs/2603.12440) 方法。

## 核心机制

- 行为描述符网格维护多样化高质量候选（防止模式坍缩）
- 元提示词与候选内核共同进化（缓解上下文退化）
- 模板化参数搜索（算法搜索与参数调优解耦）

## 入口文件

- 工作流脚本：`kernelfoundry-kernel-optimization.js`
- 英文完整文档：`README.md`

## 论文

- [KernelFoundry: Hardware-aware evolutionary GPU kernel optimization](https://arxiv.org/abs/2603.12440)

### 2026-09-21 follow-up

SOL 路径的候选/测量/任务绑定由 Host evaluator 写入并计算哈希。只读 agent 不再负责创建绑定，也不覆盖已验证分数。

### Native 任务自测资格

可选 `task_result_command`、`task_workload_count` 启用冻结任务 Python/CuTe 路径。快照明确选择的文件，经 broker 执行完整官方测试并归约匹配的原始任务/终态记录。只有完整正确且实测的结果可评分；NCU 仅诊断。保留有限原生拓扑及默认值。CUDAAgent/AccelOpt 比较本轮自行生成初始实现；KernelFoundry/KSearch/CUDALLM 报告 official reference 分数。返回实测快照，不重写显示源码。Native agent JSON 传输须核对原始文件及外部终验，不冒充 process Host evaluate 回执或正式消融权威。
