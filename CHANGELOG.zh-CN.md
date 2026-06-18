# 更新日志（中文）

本文件记录 Awesome-Kernel-Workflows 的重要变更。格式遵循
[Keep a Changelog](https://keepachangelog.com/),版本号遵循
[语义化版本(SemVer)](https://semver.org/)。版本策略见 `AGENTS.md`。

## [Unreleased]

### 修复（Fixed）

- **Workflow 运行期 `meta` 引用崩溃。** 顶层 workflow 与旧模板现在使用
  body-scope 的 `WORKFLOW_NAME` 常量,不再在运行期读取导出的 `meta` 对象,避免
  Claude Code Workflow dispatch 因 `ReferenceError: meta is not defined` 失败。
  genome-report codemod 也会为新 patch 的 workflow 生成同样安全的常量,并新增回归
  测试防止重新引入运行期 `meta.*` 引用。
  (所有顶层 workflow JS 文件、`_templates/*.js`,
  `scripts/patch-genome-report.js`,
  `_meta/tools/test/runtime-meta-reference.test.js`)
- **Workflow args 字符串/对象漂移。** 所有顶层 workflow 与 workflow 模板现在都会在
  读取 `args` 前内联裸脚本安全的 `arg_guard` unwrap，因此 Workflow dispatch 传入
  JSON 字符串或 `key=value` 字符串时不再退化为空参数 round。`patch-arg-guard.js`
  现在生成内联 guard 而不是 static import，并新增回归测试保证新生成 workflow 也遵守
  同一契约。(所有顶层 workflow JS 文件、`_templates/*.js`,
  `_meta/templates/*.js`, `scripts/patch-arg-guard.js`,
  `_meta/tools/test/runtime-arg-guard.test.js`)

### 新增（Added）

- **WarpSpeed:对齐 AKW v0.2 genome 与 KerSor 派发。** 支持 `exp_dir` 写入
  `genome.jsonl` 并镜像报告;KerSor 参数别名(`compile_command`、`kernel_path`、
  `ggml_root`);阶段与 Screen/Confirm/Profile 的内联 genome 自报告;manifest 补全
  topology/inputs/fidelity 字段。
  (`WarpSpeed/warpspeed-kernel-search.js`, `WarpSpeed/manifest.yaml`)

## [Unreleased] - feat/proactive-knowledge-fetch

### 变更（Changed）

- **cuda-agent:重试时主动拉取知识(pilot)。** Implement doer 在重试时(history
  非空)现被要求:先用 KerSor 注入的 `## Knowledge Tools (on-demand)` 块里的检索命令
  (如 `query.py` 查 kernel 模式、`chub search` 查 API/Triton 文档)搜一下,读 1-2 页再
  实现——而不只消费轮次开始时预取的 `## Retrieved Context`。尽力而为(retrieval 关时无
  此块;永不阻塞)。把 workflow 从"被动消费注入上下文"变为"主动调用知识工具"。其余
  workflow 暂保持被动,待同样升级。
  (`CUDAAgent/cuda-agent-kernel-optimization.js`)

## [Unreleased]

## [0.2.1] - 2026-06-17

### 新增（Added）

- **真实 genome 示例** —— `_meta/genome-trajectory-schema.md` 加入一份真实的
  `run-N/genome.jsonl`(cuda-agent 优化 fused RMSNorm),展示每个 phase 的信息密度、
  每次迭代的 `candidate_id`、以及实测 `speedup`,并附健壮解析提示(跳过非 JSON 行)。

## [0.2.0] - 2026-06-17

### 新增（Added）

- **genome / trajectory 自报告契约** —— `_meta/genome-trajectory-schema.md` 定义了
  一个轻量、append-only 的 `${exp_dir}/genome.jsonl`,运行中的 workflow 据此实时
  可观测(阶段序列 + 每次迭代的结果),并明确信任边界(work-plane / 可伪造——
  用于观测与重组器,绝不作为 loop 完成判定的信任锚)。
- **Workflow 工具存储与可观测性参考** ——
  `_meta/workflow-tool-storage-and-observability.md` 记录 Claude Code `Workflow`
  工具如何存储状态、运行中从外部能观测到什么(自报告设计的事实依据),每条结论
  标注"已文档化 [D] / 推断 [I]"。
- **`scripts/patch-genome-report.js`** —— 幂等 codemod,在每个 `phase()` 注入通用的
  入口书记员;用于引导可观测性,并作为尚未升级 / 新生成 workflow 的兜底。

### 变更（Changed）

- **全部 30 个 workflow 现在每个 phase 由 doer 自写一行富 genome。** 每个 phase 的
  主 doer agent 在干完活后、作为最后一步,把一行带结果的 JSON append 到
  `${exp_dir}/genome.jsonl`——写在工作之后,因此携带真实结果
  (`technique` / `speedup` / `candidate_id` / `status: done|error`),且循环体内
  每次迭代各一行(per-iteration trajectory)。这取代了原先无信息量的入口书记员
  (每个 phase 付一整个 agent 只写 `"entered"`)。agentless 阶段与
  secondary/driver/passthrough 辅助 agent 不插桩;doer 的任务与返回 schema 不变;
  append 失败也绝不破坏 workflow。
- **`_tools/generate-workflow.js`** 注明 genome 书记员由 codemod 在生成后注入
  (新生成的 workflow 继承入口书记员,可事后升级为 doer 自写的富版本)。

## [0.1.0] - 2026-06-17

- **引入版本管理与 changelog。** `AGENTS.md` 现定义 SemVer + Keep-a-Changelog 规范,
  版本号写在 `VERSION`。此前的 workflow 库(已收录的 workflow + substrate/模板/工具)
  作为 `0.1.0` 基线,其更早的演进记录在 git 历史中。
