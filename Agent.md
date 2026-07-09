# Agent 指南

本仓库把 GPU kernel 优化论文、开源系统和工业流程沉淀成 Claude Code workflow。添加或修改 workflow 时，第一优先级是保留原始方法的优化思想和决策机制，其次才是把它写成可执行 workflow。

## 核心原则

不要把论文里的关键机制改写成泛泛的 prompt，然后仍然声称这是严格复现。如果原方法依赖真实反馈信号，例如 NCU 指标、不变量违规、OpInfo 测试、多臂老虎机统计、图搜索奖励或 archive 更新，workflow 必须明确由哪个命令、JSON artifact 或确定性更新规则拥有这个信号。

## 必填分类

每个新增 workflow 都必须在 README catalog、方法学分类矩阵，以及推荐的 `manifest.yaml` 中声明方法类别。

除非刻意扩展 schema，否则使用以下主类别：

| 类别 | 适用场景 |
|------|----------|
| `iterative_self_improving` | 方法反复执行 plan/code/profile/learn 或等价的有状态迭代优化。 |
| `search_based` | 方法在搜索空间中采样、进化、剪枝或按 reward 选择候选。 |
| `tree_exploration` | 方法维护 tree、graph 或 world model，并进行节点/路径选择、回溯或 reward 传播。 |
| `single_pass_pipeline` | 方法基本是单遍分析或转换流水线。 |
| `multi_stage_refinement` | 方法由多个专门阶段组成，例如 generate、verify、refine、compose、postprocess。 |
| `tooling` | 条目是仓库基础设施，不是论文方法 workflow。 |

同时记录这些维度：

- 后端 / 语言：CUDA、Triton、ROCm、SYCL、DSL、CuTe、TileLang、C++。
- 搜索拓扑：iterative、beam、MCTS、UCB、MAB、MAP-Elites、epsilon-greedy tree、reasoning graph、FSM、pipeline。
- 反馈信号：profiler metric、throughput delta、correctness score、loss delta、custom invariant signal。
- 证据所有者：`ncu`、benchmark command、evaluator JSON、linter、compiler、test harness、invariant checker、archive updater。
- 状态 / 记忆：experience memory、skill memory、candidate beam、tree、graph、bandit table、archive、feature scores、strategy pool。
- Fidelity 边界：strict、faithful but simplified、inference-time adaptation、workflow adaptation、idea-preserving、high-risk partial。

## 新增 Workflow Checklist

1. 先读原论文或源仓库，再写 workflow。
2. 找出 load-bearing mechanism：这个方法到底凭什么区别于普通的 "generate and benchmark" 循环。
3. 明确 workflow 是严格复现、推理时适配，还是只保留思想的 sketch。
4. 创建 `<MethodName>/README.md` 和 `<MethodName>/<method-name>.js`。
5. 如果方法有非平凡循环，或希望通过 `_meta` 生成/校验，添加 `manifest.yaml`。
6. 同步更新 `README.md` 和 `README.zh-CN.md`：catalog 行、标签、核心循环、方法学分类矩阵。
7. 同步更新 README 的 workflow 适配矩阵，说明支持语言/后端、适合的问题类型，以及明显不适合的场景。
8. 如果新增了顶层 workflow，运行 `scripts/count-workflows.sh` 并提交更新后的 `badges/workflows.json`。
9. 如果修改了已有 fidelity 检查覆盖的 workflow，运行 `node scripts/check-fidelity-contracts.js`。
10. 如果 workflow 声称严格复现，当必要证据缺失时必须有 hard failure，而不是让 agent 自行脑补。

## Fidelity 要求

高保真 workflow 必须同时保留两件事：

- 决策循环：原方法中 planning、generation、selection、validation、learning、backtracking 的顺序和职责。
- 证据契约：原方法用来判断成功的同类真实测量信号。

当 scope 被缩小时，用精确标签说明：

- 如果省略训练、微调、RL 或数据集构建，写成 `inference-time adaptation` 或 `workflow adaptation`。
- 如果源系统 runtime 细节被压缩成 agent prompt，写成 `faithful but simplified`。
- 如果缺少核心 checker 或 profiler，写成 `idea-preserving` 或 `high-risk partial`。
- 除非 workflow 用具体 artifact 拥有原方法的反馈信号，否则不要声称 "implements the methodology"。

## 证据契约

优先使用具体 artifact，而不是自然语言说明。好的证据契约包括：

- `benchmark_command` 写出包含 `compiled`、`correct`、`speedup`、`latency_ms` 的 JSON。
- `ncu_command` 或 `ncu_binary` 在 `exp_dir` 下写出 profile report。
- `invariant_check_command` 写出 program point、thread/tile coordinates、assertion id 等不变量违规细节。
- `descriptor_result_path` 和 `archive_update_result_path` 记录 MAP-Elites descriptor 分类和 insert/update 决策。
- `roofline_result_path` 记录 compute/memory/middle-zone 分类及原始指标。
- `feature_vector_result_path` 记录 KernelBand feature vector 和 hardware mask 输入。
- `strict_harness` 或等价字段让 TritorX-style linter/OpInfo 声明可验证。

如果 workflow 只是要求 agent "analyze" 这些信号，但没有真实命令或 artifact，必须在 README 或 manifest 中把它写成限制。

## 参数命名规范

新增或修改 workflow 时，共通入口参数使用统一名称：`kernel_path`、`problem_definition`、`problem_path`、`language`、`target_gpu`、`compile_command`、`test_command`、`benchmark_command`、`iterations`、`seed_candidates`、`exp_dir`。

每个顶层 workflow 的资格判断(supported languages / problem types / backend)由 `manifest.yaml`
的 `routing.accepts:` 声明,并由 KerSor selector 强制执行(issue #24)。**不要**在 workflow `.js`
里写 `WORKFLOW_SUITABILITY` 常量或 `assertWorkflowSuitability()`——它们已废弃(仅剩 2 个待迁移的旧
workflow 还在用)。如果 workflow 需要从 args 解析 backend,只写一个不抛异常的 `resolveBackend()`
做归一化;拒绝逻辑归 selector 所有,workflow 不重复判定。不要从自然语言 `problem_definition` 中猜测
语言或问题类型;把资格声明写进 manifest `routing.accepts`。

如果只提供 `problem_definition` 或 `problem_path`，支持生成的 workflow 必须先生成并验证初始 kernel，再进入优化循环。证据契约参数如 `ncu_command`、`invariant_result_path`、`descriptor_result_path`、`archive_update_result_path` 不为统一命名而改写。

`compile_command`、`test_command`、`benchmark_command`、`profile_command`、`ncu_command`、`harness_build_cmd` 等执行命令必须由用户或问题定义显式提供。workflow JS 可以描述命令需要产出的 JSON/artifact 契约，但不能写死 `python ...`、`nvcc ...`、`bash ...`、`ncu --...` 这类默认命令；缺少命令时应标记 measured evidence 缺失，只做静态分析或保守估计。

## 文档标准

- 每个 workflow 目录应自包含：source link、required args、expected artifacts、example invocation。
- 英文和中文 README 的 taxonomy 必须一致。
- 描述 category、topology、feedback、correctness、return fields 时，优先使用 `docs/manifest-schema.yaml` 的词汇。
- 方法声明要保守。宁可写 "inspired by"，也不要夸大 fidelity。
- 添加 workflow 时不要混入无关重构。

## 完成前自查问题

新增 workflow PR 完成前，在 README 或 manifest 中回答这些问题：

1. 来源论文/仓库是什么？翻译的是哪一节或哪个组件？
2. 这个方法独特的优化思想是什么？
3. 权威反馈信号是什么？
4. 哪些状态会跨 attempt/round 保留？
5. 候选在什么条件下被 accept、reject、archive 或 backtrack？
6. 原方法中哪些部分被有意省略？
7. 哪些具体命令或 artifact 证明正确性和性能？
