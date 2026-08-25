# 更新日志（中文）

本文件记录 Awesome-Kernel-Workflows 的重要变更。格式遵循
[Keep a Changelog](https://keepachangelog.com/),版本号遵循
[语义化版本(SemVer)](https://semver.org/)。版本策略见 `AGENTS.md`。

## [Unreleased]

### 新增（Added）

- **Harness Engineering workflow。** 新增基于 arXiv:2607.17979 的冻结契约、
  profile 驱动优化循环。编译/正确性、可选深度验证与计时仍由调用方命令拥有；
  候选只写入 `exp_dir`，只有正确且严格更快才晋升，并保留原始产物。
  (`HarnessEngineering/`)
- **Atrex 严格适配器。** 新增对 Atrex Kernel Agent 唯一受支持入口
  `orchestrator/optimize.py` 的 fail-closed adapter。V0/V1、clean-session Long
  Horizon episode、profiling、optimization dropout、完整 workload 验证、同
  allocation ABBA 与 squash promotion 仍由官方 supervisor 负责。(`Atrex/`)
- **研究支持的验证配置。** 根据 arXiv:2608.12700、arXiv:2607.16241 与
  arXiv:2607.27231 文档化 `contract-grade`、`kernelbench-verified`、
  `kernelgenbench` 三类证据义务；配置名不能替代项目可执行 verifier。
  (`_substrate/verification/README.md`)

### 修复（Fixed）

- **累计失败策略约束现在遵循 transfer object 的权威状态。** Workflow typed
  args 优先使用 KerSor 提供的累计 `failed_strategy_ids`，仅将逐轮 attempt
  evidence 保留为旧调用方回退；因此失败策略会持续被排除，直到后续
  validated win 将其覆盖。Typed-args codemod 也实现了已文档化的
  `--refresh` 模式，使所有生成的 workflow 投影都能从脚手架 SSOT 重新同步。
  (`_meta/scaffolding/typed-args.js`、`scripts/patch-typed-args.js`、workflow
  投影、`_meta/tools/test/`)

- **K-Search 的 legacy candidate 路径现在遵循已声明源码语言。** CUDA 生成会写入
  `cycle_<n>_a<m>.cu`，不再把 CUDA 源码放在 `.py` 后缀下；driver-backed 路径仍
  使用 driver 声明的扩展名。(`KSearch/ksearch-kernel-optimization.js`、
  `_meta/tools/test/ksearch-guard.test.js`)

- **macOS 上可移植的 `nsys` 测试回退。** Substrate driver-script 测试的隔离
  `PATH` 现在包含 `/bin`，使 `/usr/bin/env bash` launcher 可以解析 `bash`，
  同时不放宽生产命令路径。(`_substrate/tests/test_driver_scripts.py`)

- **SOL benchmark 机械步骤改由 workflow Host 执行。** 共享 substrate 会探测 Host
  的 `evaluate(sol-execbench-v1)` 原语；K-Search、CUDA-Agent、AdaExplore、
  KernelAgent 与 KernelFoundry 把候选精确源码交给同一个确定性打包/运行/解析
  transaction，仅把 prompt 操作路径保留为 legacy host 边缘适配。显式 SOL 模式
  也会跳过冗余的 integration-strategist 回合。该变更只修复开发态 workflow；
  是否发布仍由 KerSor fail-closed registry 的重新验收决定。
  (`_substrate/embedded/sol_execbench_eval.js`、核心五路 workflow、
  `_meta/tools/test/`)
  CUDA-Agent 也把显式 SOL baseline 视为合同事实，在候选生成前跳过通用的 LLM
  baseline profiler，并将旧 eager/compile baseline 字段报告为 null，而非占位值。

- **SOL 打包遵循候选源码类型。** 共享 packer 对含顶层 `run` 的 Python 候选原生
  输出 Triton 或 PyTorch module，只有 CUDA extension 要求 `PYBIND11_MODULE`。
  Embedded SOL evaluation 在 PACK/RUN/PARSE 前清除陈旧 transport/result 文件，
  失败阶段不会复用前一个候选的证据。KernelFoundry 也声明并消费规范化的
  `target_speedup`、language 与 generation 参数，同时保留有界 legacy speedup
  alias。(`_substrate/`、`KernelFoundry/`、九个 SOL workflow 投影)
- **由合同唯一控制 SOL 聚合。** 共享 sol-execbench 解析器现在直接从既有 session
  `contract.env` 读取 `aggregate_reduction`，在不新增重复 workflow 参数的前提下
  计算 `sum`、`mean` 或 `geomean`，并输出带归约来源的通用 `speedup` 字段。九个
  SOL workflow 投影均把合同传给解析器；KernelAgent 与 KernelFoundry 改为读取
  通用指标，不再假定几何均值。(`_substrate/`、九个 SOL workflow 投影、
  `_meta/tools/test/`)
- **K-Search 声明了并发容量但没有使用。** Workflow 现在通过 runtime
  `parallel()` 对互相独立的 seed 实现做有界 fan-out，保持稳定候选身份；已生成
  batch 仍串行评测，以保护 GPU timing 与 embedded project mutation，之后才从
  实测最强 seed 继续有依赖的 debug/improve 链。规范参数 `seed_candidates` 默认值为
  4，且受现有 `attempts_per_cycle` 总预算约束。(`KSearch/`、
  `_meta/tools/test/ksearch-parallel-candidates.test.js`)
- **Codex/SOL 候选源码传输与有界执行。** AdaExplore、CUDA-Agent 和
  KernelFoundry 现在会在生成与验证间传递完整 SOL 源码，把物化结果绑定到精确的
  producer call，严格串行执行打包/基准/解析，并取消可能在 workflow turn 超时后
  继续存活的 agent 重试。若一个已通过正确性的基准仅出现非正 reference latency，
  KernelFoundry 还允许对同一已打包候选进行一次纯测量重试。
  (`AdaExplore/`、`CUDAAgent/`、`KernelFoundry/`、`_meta/tools/test/`)
- **K-Search 可执行 frontier 保留。** 模型生成空树时会注入确定性的开放 frontier，
  后续空树更新会保留当前搜索状态，并显式处理 nullable exhaustion。Codex 运行因此
  能实际评估候选，而不会在 cycle zero 直接结束。
  (`KSearch/ksearch-kernel-optimization.js`、
  `_meta/tools/test/ksearch-guard.test.js`)
- **SOL CUDA 打包与单一验证所有者。** Candidate packer 现在接受没有
  `contract.env` 的 task-directory 运行，把通用 staging 文件名规范化为 `.cu`；
  KernelAgent 在权威 sol-execbench 验证后不再启动不兼容的 standalone driver。
  (`_substrate/integration/pack_sol_candidate.py`、`KernelAgent/`)
- **全目录 provenance/fidelity 发布门禁。** AdaExplore 与 KDA 现已在机器可读
  manifest 中声明论文/仓库来源；fidelity 检查还会在任一顶层 workflow manifest
  缺少非空来源记录或显式 fidelity boundary 时失败。由此把目录 32/32 的声明
  覆盖变成可复现的结构检查，但不把它冒充独立来源忠实度复核。
  (`{AdaExplore,KDA}/manifest.yaml`、`scripts/check-fidelity-contracts.js`)
- **KernelAgent 协作式 verification 安全点。** KernelAgent 现在在首次验证和
  每轮 refinement batch 之后物化并 checkpoint 当前最优实测 candidate，在这些
  自然边界检查 supervisor/deadline 控制；收到停止请求后不再派生新工作并正常
  返回。其 manifest 已声明运行时控制参数，通用 checkpoint 恢复路径也接受其
  全 workload geomean 证据。(`KernelAgent/`、
  `_meta/scaffolding/runtime-safe-point.js`)
- **核心五路 sol-execbench 准入闭环。** K-Search、AdaExplore、
  KernelAgent 与 KernelFoundry 现在声明并执行 CUDA-Agent 已使用的生产级
  `sol_execbench_solution` 契约，包括冻结的 CLI/任务/配置/环境参数，以及共享的
  打包 → 执行 → 全 workload 解析 substrate。E4 控制器因此只会在真实 5/5
  可执行组合上放行，不再把“目录存在”误判为运行时可行。
  (`KSearch/`、`AdaExplore/`、`KernelAgent/`、`KernelFoundry/`、
  `scripts/patch-sol-execbench-eval.js`)
- **严格的全 workload sol 正确性门控。** 共享 sol-execbench 解析器现在只有在
  所有发现的 workload 均通过时才输出 `PASS`，保留更高精度的 geomean，并可
  原子写出用于源码—测量绑定的规范 JSON。部分通过的候选不再可能作为正确实现
  进入 workflow archive。
  (`_substrate/integration/parse_sol_bench.py`、
  `_substrate/embedded/sol_execbench_eval.js`)

- **KernelFoundry 不可变 elite 绑定。** 每个实测 generation 只有在以 SHA-256
  绑定精确候选字节、原始 harness JSON、规范 geomean、完整 workload 计数和任务
  身份之后才能进入 archive。Checkpoint 会持久化完整的已绑定 archive，
  update JSONL 不再为空，并将胜出 binding 返回给 KerSor 的独立晋级 gate。
  (`KernelFoundry/kernelfoundry-kernel-optimization.js`)
- **保持字节不变的安全点晋级。** 共享 runtime safe point 现在可在校验预期
  SHA-256 后原子复制不可变的被测候选，而不是从 LLM 返回文本重建源码。
  KernelFoundry 已使用该路径，避免过期或跨任务源码与其他候选的测量错配。
  (`_meta/scaffolding/runtime-safe-point.js`)

## [0.12.1] - 2026-07-25

### 新增（Added）

- **共享运行时安全点脚手架。** KSearch、CUDAAgent、AdaExplore 与
  KernelFoundry 现在共用同一份 sandbox-safe helper，在各自自然的
  cycle/turn/step/generation 边界原子固化最强候选与 checkpoint，并读取注入的
  termination file/deadline。(`_meta/scaffolding/runtime-safe-point.js`)

### 修复（Fixed）

- **FI26 workflow 组支持协作终止。** KSearch 每 cycle 保存可续跑树状态，
  CUDAAgent 在完成验证的 turn 后保存，AdaExplore 在 MCTS step 后保存。
  supervisor 请求现在会正常返回 `termination_reason`、进度计数与
  `checkpoint_path`；提前停止后会跳过昂贵的最终报告/记忆阶段。

## [0.12.0] - 2026-07-25

### 新增（Added）

- **KernelFoundry 代际安全点。** 每个完整代际现在会原子保存最佳源码、原始
  harness result 指针、规范指标、目标状态和终止状态。受监督运行可以消费注入的
  termination file/deadline，并在该边界正常返回。

### 修复（Fixed）

- **KernelFoundry 目标收敛与指标忠实性。** 生产运行在满足可配置的目标耐心值/
  最小代数后停止（`stop_on_target=false` 仍保留固定预算研究模式）。候选和结果
  路径已确定化，机械归一化步骤只从 harness JSON 的 `geomean_speedup` 读取性能，
  不再允许用均值延迟比替换任务规范指标。目标截断后的 fitness 相同时，现由更高的
  规范 speedup 破平局，而不是保留第一个过线候选。
- **忠实的 sol-execbench 评估。** solution packer 现在输出受支持的
  `cuda_cpp` 语言枚举。五个 opt-in workflow 都会保留 benchmark 环境与可选的
  `--definition`，在 host probe 中声明 CLI，并在 standalone 分类之前显式选择
  `sol_execbench_solution`。当首选 harness 不可用时，strategist 会按失败关闭。
- **#71 AKO4X 空候选轮次。** 在 hypothesis dispatch 前初始化轮级评估计数，
  并聚合各 hypothesis 的结果，避免零候选路径引用未定义变量。
- **#104 worktree 隔离。** 当 KerSor 报告运行时工作区不是 Git 仓库时，AKO4X
  自动选择 `fresh-process`；依赖 Git 的 manifest 同时声明所需能力。

### 变更（Changed）

- **#72 sol-execbench 池。** 将 KDA 与 KernelBlaster 加入 opt-in 池，补齐
  solution contract、pack/run/parse 评估和运行时参数声明，池内现有五个 workflow。
- **#73 integration 词汇。** 将不受支持的 `external_harness`、`project_native`
  替换为现有词汇，并移除对应的 KerSor lint debt。

## [0.11.0] - 2026-07-09

### 新增（Added）

- **Manifest 新增 `routing.emits[]` / `routing.consumes[]`(跨 DSL 算法先验元数据)。**
  可选、信息性字段，声明工作流产出或消费"算法先验"证据类(分区策略 / 瓶颈分类 /
  数值下限)——该证据在可移植 DSL → backend-native 升级中可跨 DSL 边界传递。
  `AKO4X/manifest.yaml` 声明 `routing.emits: [algorithmic_priors]`;
  `CUDAAgent/manifest.yaml` 声明 `routing.consumes: [algorithmic_priors]`;
  `docs/manifest-schema.yaml` 文档化两个字段。仅供 KerSor 审计工具消费,
  不影响 AKW 调度本身。涉及文件:`AKO4X/manifest.yaml`、`CUDAAgent/manifest.yaml`、
  `docs/manifest-schema.yaml`。
- **`CUDAAgent` Implement 阶段的跨 DSL 先验提示词补丁。** 当 Triton / TileLang
  工作流的传递对象携带 `validated_win`(分区策略:`split_k` / `stream_k` /
  `persistent_kernel`)、`bottleneck`(界类型:`compute_bound` / `memory_bound` /
  `latency_bound`)或 `metric_contract`(数值下限)条目时,CUDAAgent 的 Implement
  doer 会以它们作为首个候选的算法起点——并被明确要求忽略 handoff 中的 tile 形状、
  warp 数、`num_stages`、`cluster_shape` 或任何其他精细调度,因为这些是 Triton
  编译器的运行点,并不会迁移到手写 CUDA。涉及文件:
  `CUDAAgent/cuda-agent-kernel-optimization.js`。上游设计(在 KerSor 中):
  `docs/superpowers/specs/2026-07-09-triton-first-cuda-escalation-priors-design.md`。

### 变更（Changed）

- **`AGENTS.md`:新增"不可协商"的 workflow 代码硬规则**,从源头阻止 authoring drift。
  新 agent 构建/修改 workflow 时会被明确告知(并附上每条规则对应的 CI guard):共享 helper
  只在 `_meta/scaffolding/` 单一来源——不要手工编辑 `BEGIN/END inlined` 区块,改 SSOT 再跑
  codemod;资格判断走 manifest `routing.accepts`,而非已废弃的 `WORKFLOW_SUITABILITY` /
  `assertWorkflowSuitability`;以及运行时 sandbox 约束(禁 `import`、禁 `Date.now`/
  `Math.random`、`agent()` 用 `agentRetry` 包裹、substrate 用 `--artifact/--problem/--out`、
  写文件到 `args.exp_dir`)。涉及文件:`AGENTS.md`。

### 修复（Fixed）

- **`Agent.md`:修正过时的资格判断指引。** 参数命名一节此前要求作者发出
  `WORKFLOW_SUITABILITY` + `assertWorkflowSuitability()`,该做法已被 manifest
  `routing.accepts` + KerSor selector(issue #24)取代,且现已被 generator 禁止。已重写对齐。
  涉及文件:`Agent.md`。

- **收敛 driver-backed workflow 的残留 profiling 耦合。** `Generalist` 现在通过共享的
  driver Layer-A envelope profiling baseline、当前最佳和候选 attempt，因此 Triton/其他
  driver path 不再渲染旧的 `ncu_command`/benchmark prompt。`AKO4X` 在 perf-heuristic
  路径也会发出统一的 `driver-profile-*` envelope，并继续把吞吐归一化交给 substrate
  profiling normalizer；driver-backed prompt 使用中性的 profile 词汇与 artifact
  目录。AKO4X Triton dry-run guard 现在会大小写无关地检查 `ncu` 泄漏，并显式断言
  `perf_to_evidence.py` 路径。
  (`Generalist/generalist-kernel-optimization.js`,
  `AKO4X/ako4x-kernel-optimizer.js`,
  `_meta/tools/test/ako4x-triton-dryrun.test.js`)

- **移除迁移后的旧文档引用。** Substrate 文档现在使用 `/kersor:optimize`
  命令名和 `KerSor/docs/transfer-object.md`；agent 指南把 manifest 作者指向
  `docs/manifest-schema.yaml`，不再指向已删除的 `_manifests/schema.yaml`。
  (`_substrate/ARCHITECTURE.md`, `_substrate/SOLVER-SDK.md`, `Agent.md`)

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

- **GemmPTX workflow `GemmPTX/`**。新增一个 GEMM 专用的 CUDA/CuTe/CUTLASS
  optimizer，从硬件资源探测和 PTX/SASS 指令证据出发：候选必须先编译、通过正确性，
  并用反汇编证明预期的 `mma.sync` / `wgmma.mma_async` / TMA / `tcgen05`
  指令路径确实出现，之后才允许用 benchmark/profile 证据晋升。它为用户提供
  GEMM 指令路径调优 workflow，同时明确不声称覆盖通用 compute-bound 优化。该
  workflow 现在随附本地 `gemmptx-instruction-evidence` skill，记录架构/指令映射
  和 PTX/SASS 证据门；count/badge 32 → 33。
  (`GemmPTX/`, `README.md`, `README.zh-CN.md`,
  `_meta/tools/test/gemmptx-contract.test.js`, `badges/workflows.json`)
- **AutoMegaKernel adapter workflow `AutoMegaKernel/`**。新增 AKW 第一条严格
  external-harness adapter：`automegakernel-megakernel-optimization.js`、中英文
  README、manifest 与契约测试。该 workflow 依赖已有 AutoMegaKernel checkout，
  并把权威的 ScheduleConfig / `kernel_knobs` 搜索、validate-before-launch、正确性、
  latency/roofline 证据和 keep/revert 委托给 AMK（`amk propose/eval/loop/autoresearch`），
  因此它不是 standalone CUDA optimizer，也不是 AMK 独立重写。Count/badge 31 → 32。
  (`AutoMegaKernel/`, `README.md`, `README.zh-CN.md`,
  `_meta/tools/test/automegakernel-adapter-contract.test.js`,
  `badges/workflows.json`)
- **权威 Ascend/AscendC workflow `AscendC/`**（#16，P0）。新增一条 Ascend 原生的
  catalog 条目（`ascendc-kernel-optimization.js` + 中英文 README + `manifest.yaml`），
  源自 910b-exp 多个 session 中演化并验证的 session-local 变体。面向 Ascend 910B 上的
  AscendC，经 msprof 与 substrate `ascend` backend（`ascendc_direct_launch`），让 Ascend 任务
  不再在选型阶段 STALL、无需每次重新演化 session-local 变体。Count/badge 30 → 31。
- **`agentRetry` + null 守卫默认脚手架覆盖所有 `agent()` 类 workflow**（#17）。新增权威助手
  `_meta/scaffolding/agent-retry.js` 与 codemod `scripts/add-agent-retry-scaffolding.js`
  （感知字符串/模板/正则），把每个 `agent()` 调用（573 处 / 31 文件）包裹成有限次重试，并对
  解引用点加 null 守卫，使瞬态 API 429 / agent 跳过不再让整个 run 崩溃。KDA 的 Implement
  prompt 还加了 turn 边界指令；新的 AscendC workflow 内建 turn 边界 + 逐文件 Bash 写入 +
  NO HARNESS MANIPULATION。
- **后端无关的通用 workflow 开放 Ascend 路由**（#16）。`Generalist` 与 `KDA`（均为
  `method_supported_backends: any`、`portability: clean`）现声明 `ascendc`/`ascend`，经 substrate
  ascend backend 路由 Ascend（faithful but simplified）。`InPlacePatch` 刻意不开放——它是
  `vendor_locked`/`intrinsic_to: nvcc/hipcc`，没有 Ascend（bisheng）路径。
- **WarpSpeed:对齐 AKW v0.2 genome 与 KerSor 派发。** 支持 `exp_dir` 写入
  `genome.jsonl` 并镜像报告;KerSor 参数别名(`compile_command`、`kernel_path`、
  `ggml_root`);阶段与 Screen/Confirm/Profile 的内联 genome 自报告;manifest 补全
  topology/inputs/fidelity 字段。
  (`WarpSpeed/warpspeed-kernel-search.js`, `WarpSpeed/manifest.yaml`)

## [0.10.0] - 2026-07-08

### 新增（Added）

- **sol-execbench 成为第一类集成方法。** 在共享的 `_substrate/integration` registry +
  strategist 注册 `sol_execbench_solution`,用新的 `sol_execbench_cli` 主机能力门控
  (无 sol-execbench 的主机行为不变——S9b/S9c 覆盖)。在 sol 可用的主机上,
  can-standalone=`no` 的内核现在路由到 `sol_execbench_solution` 而非抛 `derive_adapter`。
  新增确定性底座脚本 `_substrate/integration/pack_sol_candidate.py`
  (候选内核 → sol-execbench `solution.json`;裸内核无 torch binding 时显式报错)
  与 `parse_sol_bench.py`(bench.jsonl → per-workload `speedup_factor` geomean,
  输出 `SPEEDUP=/STATUS=/WORKLOADS=` 行)。新增 `_substrate/embedded/sol_execbench_eval.js`
  底座(pack → run → parse 计划 + proposal 契约),经 `scripts/patch-sol-execbench-eval.js`
  内联进 CUDAAgent、ARGUS、Generalist;各自新增与 `IS_EMBEDDED` 互斥的 `IS_SOL`
  proposal + eval 分支。manifest 在 `routing.integration_patterns` 声明
  `sol_execbench_solution`(CUDAAgent 的 `all_args` 补齐 7 个 `sol_*` 参数)。打通
  KerSor 对 FlashInfer-Bench / sol-execbench 任务的路由。纯增量、向后兼容:全部经
  `IS_SOL` / `sol_execbench_cli` 门控;standalone/embedded 路径字节不变。

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
