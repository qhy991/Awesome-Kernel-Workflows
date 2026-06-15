# 多后端声明层全量同步 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库的声明层(README 适配矩阵、manifest backend 块、过时副本)与 spec §7.2 的 driver 能力集对齐,消除 "代码已多后端/声明仍是单后端" 的核心矛盾。

**Architecture:** 5 个独立的声明层变更批次,每个批次可独立提交/审查。不改任何 `.js` 的 prompt 行为或 portability tier 判定。权威数据来源:spec §7.2 (L731–744) 分级表 + 各 `.js` 的 `WORKFLOW_SUITABILITY`/`supported_languages` legacy 值。

**Tech Stack:** Git (分批提交) + YAML (manifest) + Markdown (README 中英双语) + basic Bash 校验

**上游 spec:** `docs/superpowers/specs/2026-06-15-multi-backend-declaration-sync-design.md`

---

### Task 1: README 适配矩阵重写 (EN)

**Files:**
- Modify: `README.md` (L139–170 → 完整重写)

**关键数据(来自 spec §4 表):**

A 组 (clean, 15 个):
| Workflow | 新后端列值 |
|---|---|
| AdaExplore | `CUDA · Triton via driver (experimental)` |
| KernelAgent | `Triton (default) · CUDA via driver (experimental)` |
| AKO4X | `Triton (default) · CUDA/CuTe/TileLang/C++/PyTorch via driver` |
| Astra | `CUDA (default) · Triton via driver (experimental)` |
| CUDALLM | `CUDA (default) · Triton via driver (experimental)` |
| KDA | `CUDA (default) · Triton via driver (experimental)` |
| KSearch | `Triton (default) · CUDA/Python via driver` |
| ReGraphT | `CUDA (default) · Triton via driver (experimental)` |
| STARK | `CUDA (default) · Triton via driver (experimental)` |
| StitchCUDA | `CUDA (default) · Triton via driver (experimental)` |
| KernelFoundry | `CUDA (default) · Triton via driver (experimental)` |
| KernelFoundryDx | `Triton (method-intrinsic)` |
| KernelSkill | `CUDA (default) · Triton via driver (experimental)` |
| KernelBand | `Triton (default) · CUDA via driver` |
| Generalist | `CUDA (default) · Triton via driver (experimental)` |

B 组 (vendor_locked, 8 个):
| Workflow | 新后端列值 |
|---|---|
| AccelOpt | `CUDA (default) · Triton via driver (vendor-locked: ncu)` |
| CUDAAgent | `CUDA (vendor-locked: ncu)` |
| cuPilot | `CUDA (vendor-locked: ncu)` |
| KEET | `CUDA (vendor-locked: ncu)` |
| KernelBlaster | `CUDA (vendor-locked: ncu)` |
| GPUForecasters | `CUDA (vendor-locked: ncu)` |
| TritorX | `Triton (vendor-locked: linter)` |
| Xe-Forge | `Intel XPU (vendor-locked: xpu)` |

C 组 (method_intrinsic, 4 个):
| Workflow | 新后端列值 |
|---|---|
| ArchAgent | `C++ / ChampSim (method-intrinsic)` |
| CutlassGEMM | `CUTLASS / C++ (method-intrinsic)` |
| FACT | `CUTLASS / C++ (method-intrinsic)` |
| ARGUS | `ROCm/CUDA/Triton/ARGUS-DSL (legacy; driver pending)` |

D 组 (spec §7.2 未覆盖, 2 个):
| Workflow | 新后端列值 |
|---|---|
| InPlacePatch | `CUDA/ROCm (vendor-locked: nvcc/hipcc)` |
| LlamacppEmbeddedSearch | `llama.cpp ggml-cuda (method-intrinsic)` |

- [ ] **Step 1: 定位 README.md 旧矩阵范围并备份**

旧矩阵在 L139–170(`Every top-level workflow declares...` 到矩阵末行)。精确行号可能因之前的编辑而偏移,先确认:
```bash
cd Awesome-Kernel-Workflows
grep -n "Every top-level workflow declares\|Supported language/backend" README.md | head -5
```

- [ ] **Step 2: 重写 L139 prose 使其与新值口径一致**

旧 L139 prose:
> Every top-level workflow declares `WORKFLOW_SUITABILITY` after `meta`. If you explicitly pass an unsupported `args.language` or `args.problem_type`, the workflow fails before doing any work and reports the supported values plus the reason. The check is deliberately conservative: workflows do not infer language or problem type from natural-language `problem_definition`.

新 prose:
> Every top-level workflow declares its backend capability in `<Workflow>/manifest.yaml` (the source of truth). The table below shows the **driver-path capability** — the backend(s) the workflow supports when invoked with `args.backend` + `args.backend_dir`. When invoked without a driver directory (legacy path), the `.js` `WORKFLOW_SUITABILITY` guard may accept a narrower subset; see each workflow's `manifest.yaml` `notes` field for legacy/driver differences. The check is deliberately conservative: workflows do not infer language or problem type from natural-language `problem_definition`.

- [ ] **Step 3: 替换整个矩阵体(L141–170)**

用上面 29 行 (A15 + B8 + C4 + D2) 的值替换旧矩阵。格式保持现有 5 列:
```
| [AccelOpt](AccelOpt/) | CUDA (default) · Triton via driver (vendor-locked: ncu) | `cuda-kernel-optimization`, `cuda-kernel-generation` | Existing CUDA kernels, or CUDA seed generation followed by NCU/benchmark-driven optimization | Triton/SYCL/XPU tasks or missing benchmark/profile contract |
```

注意:保持 `problem_type`、`Good fit`、`Avoid when` 列不变(这些是方法语义,不属 backend 声明)。

- [ ] **Step 4: 检查 Meta-Workflow 行是否保留**

确认矩阵末尾的 `| [Meta-Workflow](_meta/) | Tooling | N/A | ...` 行仍在。

- [ ] **Step 5: 提交**

```bash
git add README.md
git commit -m "docs: rewrite README adapter matrix with driver-capability backend columns

Align all 29 workflow rows with spec §7.2 portability tiers and manifest
driver capability sets. Add InPlacePatch and LlamacppEmbeddedSearch rows
(spec §7.2 currently omits these two). Update L139 prose to explain the
driver-path vs legacy-path distinction.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: README 适配矩阵重写 (ZH)

**Files:**
- Modify: `README.zh-CN.md` (L139–170 → 完整重写,与 EN 一致)

> 中文版 "Supported language/backend" 列值、prose 与 EN 一致的语义,只翻译语言。保持中文 README 的所有其他列不变。

- [ ] **Step 1: 定位 README.zh-CN.md 旧矩阵范围**

```bash
grep -n "WORKFLOW_SUITABILITY\|Supported language/backend\|支持的语言/后端" README.zh-CN.md | head -5
```

- [ ] **Step 2: 更新 L139 prose (中文)**

> 每个顶层 workflow 在 `<Workflow>/manifest.yaml` 中声明其 backend 能力(唯一权威来源)。下表展示 **driver 路径能力**——workflow 在传入 `args.backend` + `args.backend_dir` 时支持的后端。在无 driver 目录时(legacy 路径),`.js` 的 `WORKFLOW_SUITABILITY` 守卫可能接受更窄的子集;差异详见各 workflow 的 `manifest.yaml` `notes` 字段。该检查是保守的:workflow 不会从自然语言 `problem_definition` 推断语言或问题类型。

- [ ] **Step 3: 替换矩阵体**

用 Task 1 的 29 行值(中文措辞翻译),保持 5 列。示例:
```
| [AccelOpt](AccelOpt/) | CUDA (默认) · Triton via driver (vendor-locked: ncu) | `cuda-kernel-optimization`, `cuda-kernel-generation` | 已有 CUDA kernel,或 CUDA 种子生成后接 NCU/benchmark 驱动优化 | Triton/SYCL/XPU 任务或缺少 benchmark/profile 契约 |
```

- [ ] **Step 4: 提交**

```bash
git add README.zh-CN.md
git commit -m "docs: rewrite README.zh-CN adapter matrix with driver-capability backend columns

Mirror the EN README rewrite — 29 rows aligned with spec §7.2 tiers and
manifest driver capability sets. L139 prose updated to driver-path/legacy
distinction.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: A 组 manifest 校验 — Generalist 修正 + 6 个 legacy 子集补 notes

**Files:**
- Modify: `Generalist/manifest.yaml` (backend 块)
- Modify: `AdaExplore/manifest.yaml` (backend 块加 notes)
- Modify: `KernelAgent/manifest.yaml` (backend 块加 notes)
- Modify: `Astra/manifest.yaml` (backend 块加 notes)
- Modify: `CUDALLM/manifest.yaml` (backend 块加 notes)
- Modify: `KDA/manifest.yaml` (backend 块加 notes)
- Modify: `STARK/manifest.yaml` (backend 块加 notes)
- Modify: `StitchCUDA/manifest.yaml` (backend 块加 notes)

**A 组其余 5 个(manifest 已正确,无需改):**
- `AKO4X/manifest.yaml` — `supported: [triton, cuda, cute-dsl, tilelang, cpp, pytorch]` ✓
- `KernelBand/manifest.yaml` — `supported: [triton, cuda]` ✓
- `KernelFoundryDx/manifest.yaml` — `supported: [triton]` ✓
- `KSearch/manifest.yaml` — `supported: [triton, cuda, python]` ✓
- `ReGraphT/manifest.yaml` — `supported: [cuda, triton]` ✓

- [ ] **Step 1: 修正 Generalist/manifest.yaml — 扩 `supported` 为 `[cuda, triton]`**

当前 `Generalist/manifest.yaml` 的 backend 块:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda]
  default: cuda
  # The Generalist beam loop and all seven composable axis components ...
```

改为:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
  # The Generalist beam loop and all seven composable axis components
  # (evidence_schema, anti_cheat, diagnose, memory_store, method_gate,
  # verify_insight, ceiling/stagnation gates) are backend-agnostic.
  # The only language surface is the LANGUAGE arg threaded into
  # plan/implement/evaluate prompts and the code-fence token wrapping
  # kernel snippets. Driver-path triton is contract-tested (generalist-
  # triton-dryrun.test.js); legacy path (no backend_dir) is still guarded
  # by .js WORKFLOW_SUITABILITY=['cuda'] until P5 sync.
```

使用 Edit 工具精确替换 `supported: [cuda]` 为 `supported: [cuda, triton]`,并更新注释。

- [ ] **Step 2: 给 AdaExplore/manifest.yaml 的 backend 块加 notes(legacy 仅 triton / default 分歧)**

当前 `AdaExplore/manifest.yaml` 的 backend 块:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
```

在 `default: cuda` 后追加:
```yaml
  # Legacy path (no backend_dir): .js WORKFLOW_SUITABILITY accepts only
  # ['triton'] with default triton. Manifest default cuda reflects driver
  # capability; the default mismatch (cuda vs triton) is a known divergence
  # between driver and legacy paths — see B2 in design spec.
  notes: >
    Legacy WORKFLOW_SUITABILITY=['triton'] (default triton). Driver-path
    supported=[cuda, triton] (default cuda). The default mismatch is a
    known divergence; triton-dryrun.test.js contract-tests the driver path.
```

- [ ] **Step 3: 给 KernelAgent/manifest.yaml 的 backend 块加 notes(legacy 仅 triton)**

当前 `KernelAgent/manifest.yaml` 的 backend 块:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: triton
```

在 `default: triton` 后追加:
```yaml
  # Legacy path (no backend_dir): .js WORKFLOW_SUITABILITY accepts only
  # ['triton']. cuda-dryrun.test.js contract-tests the cuda driver path.
  notes: >
    Legacy WORKFLOW_SUITABILITY=['triton']. Driver-path supported=[cuda, triton].
    cuda-dryrun.test.js contract-tests the cuda driver path.
```

- [ ] **Step 4: 给 Astra/manifest.yaml 的 backend 块加 notes(legacy 仅 cuda)**

当前 `Astra/manifest.yaml` 的 backend 块:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
  # Note: integration_mode='sglang' is vendor-locked (NVIDIA + sglang);
```

在注释块末尾追加:
```yaml
  # Notes: legacy WORKFLOW_SUITABILITY=['cuda']; driver-path supported=
  # [cuda, triton]. triton-dryrun.test.js contract-tests the driver path.
  notes: >
    Legacy WORKFLOW_SUITABILITY=['cuda']. Driver-path supported=[cuda, triton].
    integration_mode='sglang' is NVIDIA-only (guard throws on non-cuda).
```

- [ ] **Step 5: 给 CUDALLM/manifest.yaml 的 backend 块加 notes(legacy 仅 cuda)**

当前 `CUDALLM/manifest.yaml` 的 backend 块:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
```

在 `default: cuda` 后追加:
```yaml
  notes: >
    Legacy WORKFLOW_SUITABILITY=['cuda']. Driver-path supported=[cuda, triton].
    triton-dryrun.test.js contract-tests the driver path.
```

- [ ] **Step 6: 给 KDA/manifest.yaml 的 backend 块加 notes(legacy 仅 cuda)**

当前 `KDA/manifest.yaml` 的 backend 块:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
```

在 `default: cuda` 后追加:
```yaml
  notes: >
    Legacy WORKFLOW_SUITABILITY=['cuda']. Driver-path supported=[cuda, triton].
    triton-dryrun.test.js contract-tests the driver path.
```

- [ ] **Step 7: 给 STARK/manifest.yaml 的 backend 块加 notes(legacy 仅 cuda)**

当前 `STARK/manifest.yaml` 的 backend 块:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
  # STARK's tree-search + ε-greedy + grounded-instruction methodology is ...
```

在注释块末尾追加:
```yaml
  notes: >
    Legacy WORKFLOW_SUITABILITY=['cuda']. Driver-path supported=[cuda, triton].
    triton-dryrun.test.js contract-tests the driver path.
```

- [ ] **Step 8: 给 StitchCUDA/manifest.yaml 的 backend 块加 notes(legacy 仅 cuda + KernelBench guard)**

当前 `StitchCUDA/manifest.yaml` 的 backend 块:
```yaml
backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
```

在 `default: cuda` 后追加:
```yaml
  notes: >
    Legacy WORKFLOW_SUITABILITY=['cuda']. Driver-path supported=[cuda, triton].
    triton-dry-run.test.js contract-tests the driver path. kernelbench_config.
    benchmark_suite requires CUDA driver (intersectional guard in workflow body).
```

- [ ] **Step 9: 提交**

```bash
git add Generalist/manifest.yaml AdaExplore/manifest.yaml KernelAgent/manifest.yaml \
        Astra/manifest.yaml CUDALLM/manifest.yaml KDA/manifest.yaml \
        STARK/manifest.yaml StitchCUDA/manifest.yaml
git commit -m "docs: sync A-group manifest backend blocks with spec §7.2 driver sets

- Generalist: expand supported from [cuda] to [cuda, triton] (triton
  contract-tested via generalist-triton-dryrun.test.js)
- AdaExplore/KernelAgent/Astra/CUDALLM/KDA/STARK/StitchCUDA: add notes
  recording legacy WORKFLOW_SUITABILITY subsets (6 of 8 have .js legacy
  narrower than driver capability)
- AdaExplore: explicit note on default mismatch (manifest cuda vs legacy
  triton) — known divergence per B2

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 新建 B/C/D 组 manifest (14 个 vendor_locked/method_intrinsic)

**Files:**
- Create: `AccelOpt/manifest.yaml`
- Create: `CUDAAgent/manifest.yaml`
- Create: `cuPilot/manifest.yaml`
- Create: `KEET/manifest.yaml`
- Create: `KernelBlaster/manifest.yaml`
- Create: `GPUForecasters/manifest.yaml`
- Create: `TritorX/manifest.yaml`
- Create: `Xe-Forge/manifest.yaml`
- Create: `ArchAgent/manifest.yaml`
- Create: `CutlassGEMM/manifest.yaml`
- Create: `FACT/manifest.yaml`
- Create: `ARGUS/manifest.yaml`
- Create: `InPlacePatch/manifest.yaml`
- Create: `LlamacppEmbeddedSearch/manifest.yaml`

**模板(最小头 + backend 块):**

```yaml
schema_version: "1.1"
name: <workflow-name>
entrypoint: <workflow>.js

backend:
  portability: <vendor_locked | method_intrinsic>
  matrix_eligible: <false | partial>
  intrinsic_to: <anchor>
  method_supported_backends: <any | [list]>
  supported: [<...>]
  default: <...>
  notes: "<fidelity caveat>"

source:
  paper_title: "<from _manifests or README, verbatim only>"
  paper_url: "<if known>"
  notes: "<portability>; full phase spec deferred."

workflow:
  name: "<same as name>"
  description: "<from meta.description>"
  when_to_use: "<from meta.whenToUse>"
  output_filename: "<entrypoint>"
  directory: "<WorkflowDir>/"
```

- [ ] **Step 1: 创建 AccelOpt/manifest.yaml**

```yaml
schema_version: "1.1"
name: accelopt-kernel-optimization
entrypoint: accelopt-kernel-optimization.js

backend:
  portability: vendor_locked
  matrix_eligible: partial
  intrinsic_to: NVIDIA NCU
  method_supported_backends: [cuda, triton]
  supported: [cuda, triton]
  default: cuda
  requires_capability:
    metrics: [dram_pct, sm_pct]
  notes: >
    Tier per spec §7.2 (vendor_locked, intrinsic to NCU). User review 6/15
    argues method is clean (CUDA-idiom residue is an implementation issue,
    not a method constraint) — see tier-revision-proposal.md. Legacy
    WORKFLOW_SUITABILITY=['cuda','triton']; accelopt-triton-dryrun.test.js
    contract-tests the driver path.

source:
  paper_title: "AccelOpt: Accelerating Kernel Optimization via Self-Improving LLM Agents"
  paper_url: "https://arxiv.org/abs/2511.15915"
  paper_venue: "MLSys 2026"
  notes: "vendor_locked; full phase spec deferred."

workflow:
  name: accelopt-kernel-optimization
  description: "Self-improving kernel optimization loop with profiler-driven evidence (AccelOpt methodology; NCU on the cuda backend)"
  when_to_use: "When you need to iteratively optimize a GPU kernel through plan-execute-profile-learn cycles. Uses the backend driver"
  output_filename: "accelopt-kernel-optimization.js"
  directory: "AccelOpt/"
```

- [ ] **Step 2: 创建 CUDAAgent/manifest.yaml**

```yaml
schema_version: "1.1"
name: cudaagent-kernel-optimization
entrypoint: cudaagent-kernel-optimization.js

backend:
  portability: vendor_locked
  matrix_eligible: partial
  intrinsic_to: NVIDIA NCU
  method_supported_backends: [cuda]
  supported: [cuda]
  default: cuda
  requires_capability:
    metrics: [dram_pct, sm_pct]
  notes: >
    Tier per spec §7.2 (vendor_locked, intrinsic to NCU). Legacy
    WORKFLOW_SUITABILITY=['cuda']; cudaagent-guard.test.js validates.

source:
  paper_title: "CUDA-Agent: RL-Trained LLM Agents for GPU Kernel Optimization"
  paper_url: "https://arxiv.org/abs/2602.24286"
  paper_venue: "arXiv 2026 (ByteDance/Tsinghua)"
  notes: "vendor_locked; full phase spec deferred."

workflow:
  name: cudaagent-kernel-optimization
  description: "Skill-integrated multi-turn CUDA kernel optimization with profiling-driven iterative refinement (CUDA Agent methodology)"
  when_to_use: "When optimizing CUDA kernels from PyTorch operator specifications through iterative code generation, compilation, correctness testing, and profiling feedback. Follows the CUDA Agent skill-based workflow: profile baseline → identify bottlenecks → implement kernel + bindings → compile → verify correctness → measure speedup → refine until target met."
  output_filename: "cudaagent-kernel-optimization.js"
  directory: "CUDAAgent/"
```

- [ ] **Step 3: 创建 cuPilot/manifest.yaml**

```yaml
schema_version: "1.1"
name: cupilot-kernel-optimization
entrypoint: cupilot-kernel-optimization.js

backend:
  portability: vendor_locked
  matrix_eligible: partial
  intrinsic_to: NVIDIA NCU
  method_supported_backends: [cuda]
  supported: [cuda]
  default: cuda
  requires_capability:
    metrics: [dram_pct, sm_pct]
  notes: >
    Tier per spec §7.2 (vendor_locked, intrinsic to NCU). Legacy
    WORKFLOW_SUITABILITY=['cuda'].

source:
  paper_title: "cuPilot: Evolutionary Multi-Agent GPU Kernel Optimization via Roofline-Guided Strategies"
  paper_url: "https://arxiv.org/abs/2512.16465"
  paper_venue: "arXiv 2025 (SEU/Tsinghua)"
  notes: "vendor_locked; full phase spec deferred."

workflow:
  name: cupilot-kernel-optimization
  description: "Strategy-coordinated evolutionary multi-agent CUDA kernel optimization with roofline-guided prompting (cuPilot methodology)"
  when_to_use: "When evolving CUDA kernels through multi-generation optimization that requires sophisticated strategies (tensor cores, tiling, pipelining, memory swizzling). Uses strategy as an intermediate semantic representation to decouple evolutionary crossover from low-level code, with roofline model guidance and RAG-based strategy initialization."
  output_filename: "cupilot-kernel-optimization.js"
  directory: "cuPilot/"
```

- [ ] **Step 4: 创建 KEET/manifest.yaml**

```yaml
schema_version: "1.1"
name: keet-performance-explanation
entrypoint: keet-performance-explanation.js

backend:
  portability: vendor_locked
  matrix_eligible: partial
  intrinsic_to: NVIDIA NCU
  method_supported_backends: [cuda]
  supported: [cuda]
  default: cuda
  requires_capability:
    metrics: [dram_pct, sm_pct]
  notes: >
    Tier per spec §7.2 (vendor_locked, intrinsic to NCU). Method is a
    single-pass explanation pipeline, not an optimization loop. Legacy
    WORKFLOW_SUITABILITY=['cuda'].

source:
  paper_title: "KEET: Kernel Performance Explanation via LLM-Agent Pipeline"
  paper_url: "https://arxiv.org/abs/2605.04467"
  paper_venue: "arXiv 2026 (UMD/NVIDIA/LLNL)"
  notes: "vendor_locked; full phase spec deferred."

workflow:
  name: keet-performance-explanation
  description: "LLM-agent pipeline for interpreting NCU profiles into actionable kernel performance explanations (KEET methodology)"
  when_to_use: "When you need to understand WHY a CUDA kernel performs the way it does. Produces a natural language explanation report from NCU profiles + source code, identifying bottlenecks and suggesting specific code changes. Use as a standalone analysis tool, or as a pre-step to feed optimization context into downstream tasks."
  output_filename: "keet-performance-explanation.js"
  directory: "KEET/"
```

- [ ] **Step 5: 创建 KernelBlaster/manifest.yaml**

```yaml
schema_version: "1.1"
name: kernelblaster-kernel-optimization
entrypoint: kernelblaster-kernel-optimization.js

backend:
  portability: vendor_locked
  matrix_eligible: partial
  intrinsic_to: NVIDIA NCU
  method_supported_backends: [cuda]
  supported: [cuda]
  default: cuda
  requires_capability:
    metrics: [dram_pct, sm_pct]
  notes: >
    Tier per spec §7.2 (vendor_locked, intrinsic to NCU). Method anchored
    on NCU elapsed-cycles feedback and CUDA optimization memory. Legacy
    WORKFLOW_SUITABILITY=['cuda'].

source:
  paper_title: "KernelBlaster: Memory-Augmented In-Context RL for CUDA Kernel Optimization"
  paper_url: ""
  notes: "vendor_locked; full phase spec deferred."

workflow:
  name: kernelblaster-kernel-optimization
  description: "Memory-augmented in-context RL loop for CUDA kernel optimization with a persistent, state-keyed optimization knowledge base (KernelBlaster / MAIC-RL methodology)"
  when_to_use: "When you want to optimize a CUDA kernel via profile-guided RL rollouts that ACCUMULATE experience across kernels and runs. KernelBlaster classifies each kernel into a hardware performance state (memory / compute / latency bound), retrieves the best-known optimization for that state from a persistent knowledge base keyed by bottleneck pattern, applies it, measures the real reward (Elapsed Cycles delta via NCU), and updates the database so future rollouts and future kernels reuse what worked. Pass optimization_db_path to carry the knowledge base across invocations."
  output_filename: "kernelblaster-kernel-optimization.js"
  directory: "KernelBlaster/"
```

- [ ] **Step 6: 创建 GPUForecasters/manifest.yaml**

```yaml
schema_version: "1.1"
name: gpuforecasters-kernel-optimization
entrypoint: gpuforecasters-kernel-optimization.js

backend:
  portability: vendor_locked
  matrix_eligible: partial
  intrinsic_to: NVIDIA NCU
  method_supported_backends: [cuda]
  supported: [cuda]
  default: cuda
  requires_capability:
    metrics: [dram_pct, sm_pct]
  notes: >
    Tier per spec §7.2 (vendor_locked, intrinsic to NCU). Legacy
    WORKFLOW_SUITABILITY=['cuda'].

source:
  paper_title: "GPU Forecasters: Learned Speedup Forecasting for Kernel Optimization"
  paper_url: ""
  notes: "vendor_locked; full phase spec deferred."

workflow:
  name: gpuforecasters-kernel-optimization
  description: "Kernel optimization with learned speedup forecasting and PUCT search"
  when_to_use: "Use for kernel optimization guided by learned performance forecasting models"
  output_filename: "gpuforecasters-kernel-optimization.js"
  directory: "GPUForecasters/"
```

- [ ] **Step 7: 创建 TritorX/manifest.yaml**

```yaml
schema_version: "1.1"
name: tritorx-operator-generation
entrypoint: tritorx-operator-generation.js

backend:
  portability: vendor_locked
  matrix_eligible: false
  intrinsic_to: Triton dialect + custom linter
  method_supported_backends: [triton]
  supported: [triton]
  default: triton
  notes: >
    Tier per spec §7.2 (vendor_locked-single, Triton dialect + custom
    linter). Method targets ASIC/NPU platforms via Triton as a compilation
    frontend; correctness-first, not performance. Legacy
    WORKFLOW_SUITABILITY=['triton'].

source:
  paper_title: "TritorX: FSM-based Agentic Generation of Triton ATen Kernels for Emerging Accelerators"
  paper_url: "https://arxiv.org/abs/2512.10977"
  paper_venue: "arXiv 2025 (Meta)"
  notes: "vendor_locked; full phase spec deferred."

workflow:
  name: tritorx-operator-generation
  description: "FSM-based agentic generation of Triton ATen kernels for emerging accelerator platforms with linter-driven feedback (TritorX methodology)"
  when_to_use: "When generating functionally correct Triton kernel + wrapper pairs for an entire PyTorch ATen operator set on a new accelerator platform (ASIC/NPU). Prioritizes correctness and coverage over performance. Uses a finite state machine with custom linter, JIT compilation, and OpInfo test harness for iterative refinement."
  output_filename: "tritorx-operator-generation.js"
  directory: "TritorX/"
```

- [ ] **Step 8: 创建 Xe-Forge/manifest.yaml**

```yaml
schema_version: "1.1"
name: xe-forge-kernel-optimization
entrypoint: xe-forge-kernel-optimization.js

backend:
  portability: vendor_locked
  matrix_eligible: false
  intrinsic_to: Intel XPU
  method_supported_backends: [xpu]
  supported: [xpu]
  default: xpu
  notes: >
    Tier per spec §7.2 (vendor_locked-single, Intel XPU). User review 6/15
    argues method_intrinsic is more accurate (VTune/XMX/SPIR-V anchor) —
    see tier-revision-proposal.md. .js WORKFLOW_SUITABILITY=['triton',
    'sycl', 'xpu'] — triton/sycl are compilation frontends, the hardware
    anchor is xpu.

source:
  paper_title: "Xe-Forge: Multi-Stage CoVeR Optimization for Intel XPU Kernels"
  paper_url: ""
  notes: "vendor_locked; full phase spec deferred."

workflow:
  name: xe-forge-kernel-optimization
  description: "Multi-stage CoVeR optimization for Intel XPU kernels"
  when_to_use: "Use for optimizing kernels on Intel XPUs (Data Center GPU Max) with Triton and SYCL"
  output_filename: "xe-forge-kernel-optimization.js"
  directory: "Xe-Forge/"
```

- [ ] **Step 9: 创建 ArchAgent/manifest.yaml**

```yaml
schema_version: "1.1"
name: archagent-cache-policy-search
entrypoint: archagent-cache-policy-search.js

backend:
  portability: method_intrinsic
  matrix_eligible: false
  intrinsic_to: cpu-champsim
  method_supported_backends: [cpp]
  supported: [cpp]
  default: cpp
  notes: >
    Tier per spec §7.2 (method_intrinsic, ChampSim IPC). Not a GPU kernel
    workflow — cache replacement policy search using ChampSim. Bypasses
    diagnose.py entirely. Legacy WORKFLOW_SUITABILITY=['cpp'].

source:
  paper_title: "AlphaEvolve: Evolutionary Search for Cache Replacement Policies"
  paper_url: ""
  notes: "method_intrinsic; full phase spec deferred."

workflow:
  name: archagent-cache-policy-search
  description: "Evolutionary search for cache replacement policies using AlphaEvolve"
  when_to_use: "Use for discovering novel cache replacement policies through evolutionary search"
  output_filename: "archagent-cache-policy-search.js"
  directory: "ArchAgent/"
```

- [ ] **Step 10: 创建 CutlassGEMM/manifest.yaml**

```yaml
schema_version: "1.1"
name: cutlassgemm-gemm-optimization
entrypoint: cutlassgemm-gemm-optimization.js

backend:
  portability: method_intrinsic
  matrix_eligible: false
  intrinsic_to: CUTLASS device-level API
  method_supported_backends: [cutlass]
  supported: [cutlass]
  default: cutlass
  notes: >
    Tier per spec §7.2 (method_intrinsic, CUTLASS). Targets SOL-ExecBench
    GEMM problems on Ampere/Hopper. Legacy WORKFLOW_SUITABILITY=['cutlass',
    'cuda', 'cpp'].

source:
  paper_title: "CUTLASS GEMM Optimization with NCU Profiling and cuBLAS Fallback"
  paper_url: ""
  notes: "method_intrinsic; full phase spec deferred."

workflow:
  name: cutlassgemm-gemm-optimization
  description: "CUTLASS-based GEMM optimization with NCU profiling, ceiling detection, split-K, and cuBLAS hybrid fallback for SOL-ExecBench"
  when_to_use: "When optimizing dense GEMM kernels (C = A @ B or C = A @ B.T) using NVIDIA CUTLASS on Ampere/Hopper GPUs. Targets SOL-ExecBench problems with variable M dimension. Uses NCU profiling for root-cause analysis, split-K for small M parallelism, and cuBLAS fallback when CUTLASS overhead dominates."
  output_filename: "cutlassgemm-gemm-optimization.js"
  directory: "CutlassGEMM/"
```

- [ ] **Step 11: 创建 FACT/manifest.yaml**

```yaml
schema_version: "1.1"
name: fact-kernel-synthesis
entrypoint: fact-kernel-synthesis.js

backend:
  portability: method_intrinsic
  matrix_eligible: false
  intrinsic_to: CUTLASS pattern registry
  method_supported_backends: [cutlass]
  supported: [cutlass]
  default: cutlass
  notes: >
    Tier per spec §7.2 (method_intrinsic, CUTLASS). Compositional pattern
    synthesis via discovery + realization + composition + ablation. Legacy
    WORKFLOW_SUITABILITY=['cutlass', 'cuda', 'cpp'].

source:
  paper_title: "FACT: Compositional Kernel Synthesis with Pattern Discovery and Realization"
  paper_url: ""
  notes: "method_intrinsic; full phase spec deferred."

workflow:
  name: fact-kernel-synthesis
  description: "Compositional kernel synthesis with pattern discovery and realization"
  when_to_use: "Use for generating optimized CUTLASS kernels through compositional pattern synthesis"
  output_filename: "fact-kernel-synthesis.js"
  directory: "FACT/"
```

- [ ] **Step 12: 创建 ARGUS/manifest.yaml**

```yaml
schema_version: "1.1"
name: argus-kernel-optimization
entrypoint: argus-kernel-optimization.js

backend:
  portability: method_intrinsic
  matrix_eligible: false
  intrinsic_to: AMD MI300X MFMA invariants
  method_supported_backends: [rocm]
  supported: [rocm]
  default: rocm
  notes: >
    Tier per spec §7.2 (method_intrinsic, AMD MI300X MFMA invariants).
    Implementation currently uses legacy args.language (no backend_dir/
    driver). .js WORKFLOW_SUITABILITY=['cuda', 'rocm', 'triton',
    'argus-dsl'] — the four-language set is the legacy guard; manifest
    supported=[rocm] is the spec §7.2 target. Driver migration pending —
    do not matrix-smoke until wired.

source:
  paper_title: "ARGUS: Agentic GPU Kernel Optimization via Data-Flow Invariants and ICRL Planning"
  paper_url: "https://arxiv.org/abs/2604.18616"
  paper_venue: "arXiv 2026 (CausalFlow/HKUST/Stanford)"
  notes: "method_intrinsic; full phase spec deferred. Legacy args.language, driver migration pending."

workflow:
  name: argus-kernel-optimization
  description: "Agentic GPU kernel optimization guided by data-flow invariants with ICRL planning (ARGUS methodology)"
  when_to_use: "When optimizing GPU kernels that require coordinated reasoning over tiling, shared-memory staging, software pipelining, and instruction scheduling. Uses compile-time data-flow invariants for dense structured feedback instead of sparse pass/fail signals. Best for performance-critical kernels (GEMM, attention, MoE) targeting near-library-level throughput."
  output_filename: "argus-kernel-optimization.js"
  directory: "ARGUS/"
```

- [ ] **Step 13: 创建 InPlacePatch/manifest.yaml**

```yaml
schema_version: "1.1"
name: inplacepatch-kernel-optimization
entrypoint: inplacepatch-kernel-optimization.js

backend:
  portability: vendor_locked
  matrix_eligible: false
  intrinsic_to: nvcc/hipcc
  method_supported_backends: [cuda, rocm]
  supported: [cuda, rocm]
  default: cuda
  notes: >
    Not in spec §7.2 (needs backfill). Byte-exact in-place patch workflow
    — nvcc/hipcc compile single .so, dlopen via Python harness. No driver
    directory; uses project-native build/test/benchmark commands verbatim.
    Every iteration snapshots the original file, proposes one focused Edit,
    runs build/test/benchmark, keeps if (correct AND faster) else reverts.

source:
  paper_title: "In-Place Patch: Iterative Kernel Optimization for Embedded Kernels"
  paper_url: ""
  notes: "vendor_locked; not in spec §7.2. Full phase spec deferred."

workflow:
  name: inplacepatch-kernel-optimization
  description: "Iterative in-place patch -> project-native build -> test -> benchmark loop for kernels embedded in a larger codebase (no exp_dir variant materialization)"
  when_to_use: "When the kernel under optimization cannot be compiled standalone (depends on project headers, template instantiation, or dispatch tables - e.g. llama.cpp ggml-cuda kernels). Uses the project build/test/benchmark commands verbatim. Each iteration: snapshot original, propose ONE focused patch via Edit, run project build, run project test, run project benchmark, keep if (correct AND faster) else revert."
  output_filename: "inplacepatch-kernel-optimization.js"
  directory: "InPlacePatch/"
```

- [ ] **Step 14: 创建 LlamacppEmbeddedSearch/manifest.yaml**

```yaml
schema_version: "1.1"
name: llamacpp-embedded-search
entrypoint: llamacpp-embedded-search.js

backend:
  portability: method_intrinsic
  matrix_eligible: false
  intrinsic_to: llama.cpp ggml-cuda
  method_supported_backends: [llama.cpp]
  supported: [llama.cpp]
  default: llama.cpp
  notes: >
    Not in spec §7.2 (needs backfill). Multi-variant fan-out search for
    kernels embedded in llama.cpp ggml-cuda dispatch — cannot be compiled
    standalone. Each subagent authors a new .cuh; workflow registers it
    into fattn.cu, runs project build/test/bench, then unregisters.

source:
  paper_title: "Llama.cpp Embedded Kernel Search: Multi-Variant Fan-Out for ggml-cuda Kernels"
  paper_url: ""
  notes: "method_intrinsic; not in spec §7.2. Full phase spec deferred."

workflow:
  name: llamacpp-embedded-search
  description: "Multi-variant fan-out search for kernels embedded in llama.cpp ggml-cuda dispatch (each variant = new .cuh + register + project build + test + bench + unregister)"
  when_to_use: "When the kernel under optimization is embedded in llama.cpp ggml-cuda (cannot be compiled standalone) AND the bottleneck is structural enough that you want to fan out N independent kernel designs rather than iteratively patch one. Each subagent authors a complete new .cuh; the workflow registers it into fattn.cu via scripts/llamacpp_register_variant.py, runs the project build/test/bench, then unregisters. Variants are evaluated SERIALLY (project build is the bottleneck) but proposals are drafted in parallel."
  output_filename: "llamacpp-embedded-search.js"
  directory: "LlamacppEmbeddedSearch/"
```

- [ ] **Step 15: 提交**

```bash
git add AccelOpt/manifest.yaml CUDAAgent/manifest.yaml cuPilot/manifest.yaml \
        KEET/manifest.yaml KernelBlaster/manifest.yaml GPUForecasters/manifest.yaml \
        TritorX/manifest.yaml Xe-Forge/manifest.yaml ArchAgent/manifest.yaml \
        CutlassGEMM/manifest.yaml FACT/manifest.yaml ARGUS/manifest.yaml \
        InPlacePatch/manifest.yaml LlamacppEmbeddedSearch/manifest.yaml
git commit -m "docs: add minimal manifest.yaml for 14 vendor_locked/method_intrinsic workflows

Backend blocks follow spec §7.2 tiers: 8 vendor_locked (AccelOpt, CUDAAgent,
cuPilot, KEET, KernelBlaster, GPUForecasters, TritorX, Xe-Forge) + 4
method_intrinsic (ArchAgent, CutlassGEMM, FACT, ARGUS) + 2 not-in-§7.2
(InPlacePatch, LlamacppEmbeddedSearch). All use minimal template —
backend block + source/workflow header only; no phases/method long-form.
AccelOpt/Xe-Forge/KernelSkill tier disagreements noted in notes field.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 新建 A 组缺失 manifest (KernelFoundry + KernelSkill) + 过时副本标注

**Files:**
- Create: `KernelFoundry/manifest.yaml`
- Create: `KernelSkill/manifest.yaml`
- Create: `_manifests/README.md`
- Create: `_meta/manifests/README.md`

- [ ] **Step 1: 创建 KernelFoundry/manifest.yaml**

```yaml
schema_version: "1.1"
name: kernelfoundry-kernel-optimization
entrypoint: kernelfoundry-kernel-optimization.js

backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
  notes: >
    Tier per spec §7.2 (clean, matrix-eligible). Legacy WORKFLOW_SUITABILITY
    =['sycl','cuda','triton']; SYCL is an output language with no registered
    driver — supported lists only backends with registered drivers under
    _substrate/backends/. SYCL path requires args.backend_dir pointing to a
    user-provided driver. kernelfoundry-triton-dryrun.test.js contract-tests
    the triton driver path.

source:
  paper_title: "Kernel Foundry: Evolutionary MAP-Elites Quality-Diversity Search for GPU Kernels"
  paper_url: "https://arxiv.org/abs/2603.12440"
  paper_venue: "arXiv 2026 (Intel)"
  notes: "clean; full phase spec deferred."

workflow:
  name: kernelfoundry-kernel-optimization
  description: "Evolutionary MAP-Elites quality-diversity search with meta-prompt evolution and templated parameter tuning (KernelFoundry methodology)"
  when_to_use: "When generating GPU kernels (SYCL/CUDA/Triton) from PyTorch operator specs and you need diverse, high-quality solutions across multiple optimization strategies. Prevents mode collapse via behavioral-descriptor archive. Best for KernelBench-style tasks, custom operators, and cross-platform (Intel/NVIDIA) kernel generation."
  output_filename: "kernelfoundry-kernel-optimization.js"
  directory: "KernelFoundry/"
```

- [ ] **Step 2: 创建 KernelSkill/manifest.yaml**

```yaml
schema_version: "1.1"
name: kernelskill-kernel-optimization
entrypoint: kernelskill-kernel-optimization.js

backend:
  portability: clean
  matrix_eligible: true
  method_supported_backends: any
  supported: [cuda, triton]
  default: cuda
  notes: >
    Tier per spec §7.2 (clean, matrix-eligible). User review 6/15 argues
    vendor_locked/cuda/NCU is more accurate (no triton dry-run test, CUDA
    skill library + __global__ vocabulary throughout) — see tier-revision-
    proposal.md. Legacy WORKFLOW_SUITABILITY=['cuda']; kernelskill-cuda-
    dryrun.test.js contract-tests the cuda driver path. No triton dry-run
    exists; triton driver capability is declared per spec §7.2 but not
    yet contract-tested.

source:
  paper_title: "KernelSkill: Dual-Memory Multi-Agent CUDA Kernel Optimization"
  paper_url: ""
  notes: "clean per spec §7.2; full phase spec deferred. Tier disputed — see tier-revision-proposal.md."

workflow:
  name: kernelskill-kernel-optimization
  description: "Multi-agent CUDA kernel optimization with a dual-level memory: a curated long-term expert skill library (deterministic decision policy + method knowledge) and short-term per-task optimize/repair trajectory memory. Each refinement round branches into a repair path (invalid kernel) or an optimize path (valid kernel) driven by ncu+nsys profiling (KernelSkill / KernelMem methodology)."
  when_to_use: "When generating and optimizing custom CUDA kernels from a PyTorch reference, and you want optimization-method selection to be explicit, auditable, and grounded in profiling evidence rather than relying on the model"
  output_filename: "kernelskill-kernel-optimization.js"
  directory: "KernelSkill/"
```

- [ ] **Step 3: 创建 `_manifests/README.md` (过时副本标注)**

```markdown
# DEPRECATED — 历史归档

本目录为历史归档,**不再作为 source of truth**。

各 workflow 的 backend 声明权威来源为 `<Workflow>/manifest.yaml`。
本目录中的 `.yaml` 文件可能已过时(多为旧格式,无 `schema_version`/`backend` 块)。

如需查阅或修改 workflow 的 backend 能力,请直接编辑对应的 `<Workflow>/manifest.yaml`。
```

- [ ] **Step 4: 创建 `_meta/manifests/README.md` (过时副本标注 + 迁移状态)**

```markdown
# DEPRECATED — 迁移中草稿

本目录为 backend-driver 迁移的**中间草稿区**,**不再作为 source of truth**。

各 workflow 的 backend 声明权威来源为 `<Workflow>/manifest.yaml`。

## 迁移状态

| 文件 | 格式 | 状态 |
|------|------|------|
| `generalist.yaml` | 新(schema v1.1) | 已转新格式;需与 `Generalist/manifest.yaml` 同步 |
| `kernelfoundry.yaml` | 新(schema v1.1) | 已转新格式 |
| `kernelskill.yaml` | 新(schema v1.1) | 已转新格式;`intrinsic_to: ""` 待补 |
| 其余 | 旧(无 schema_version) | 待迁 |

如需查阅或修改 workflow 的 backend 能力,请直接编辑对应的 `<Workflow>/manifest.yaml`。
```

- [ ] **Step 5: 提交**

```bash
git add KernelFoundry/manifest.yaml KernelSkill/manifest.yaml \
        _manifests/README.md _meta/manifests/README.md
git commit -m "docs: add KernelFoundry/KernelSkill manifests + deprecate old manifest dirs

- KernelFoundry/manifest.yaml: clean, [cuda, triton], SYCL as legacy
  output language (no registered driver)
- KernelSkill/manifest.yaml: clean per spec §7.2, [cuda, triton], tier
  disputed (see tier-revision-proposal.md)
- _manifests/README.md: DEPRECATED — points to <Workflow>/manifest.yaml
  as SoT
- _meta/manifests/README.md: DEPRECATED — migration draft status, points
  to <Workflow>/manifest.yaml as SoT

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 校验与验收

- [ ] **Step 1: 核对 manifest 数量**

```bash
echo "=== 应有 29 个 manifest (13 原有 + 16 新建) ==="
ls -d */manifest.yaml 2>/dev/null | wc -l
echo "=== 列出所有 manifest ==="
ls */manifest.yaml 2>/dev/null
```

预期: 29

- [ ] **Step 2: 核对 README 矩阵行数**

```bash
echo "=== 英文 README 矩阵行数 (从表头到 Meta-Workflow) ==="
awk '/^Every top-level workflow declares/,/Meta-Workflow/' README.md | grep -cE "^\| \["
echo "=== 中文 README 矩阵行数 ==="
awk '/^Every top-level workflow declares/,/Meta-Workflow/' README.zh-CN.md | grep -cE "^\| \["
```

预期: 29 (英文) + 29 (中文)

- [ ] **Step 3: 快速 YAML 语法检查(新 manifest)**

```bash
for f in AccelOpt CUDAAgent cuPilot KEET KernelBlaster GPUForecasters \
         TritorX Xe-Forge ArchAgent CutlassGEMM FACT ARGUS \
         InPlacePatch LlamacppEmbeddedSearch KernelFoundry KernelSkill; do
  python3 -c "import yaml; yaml.safe_load(open('$f/manifest.yaml'))" 2>&1 || echo "FAIL: $f"
done
```

- [ ] **Step 4: 核对 spec §7.2 tier 一致性**

逐行检查(人工):
- clean 的 15 个: manifest `portability: clean` ✓
- vendor_locked 的 8 个: manifest `portability: vendor_locked` ✓
- method_intrinsic 的 4 个: manifest `portability: method_intrinsic` ✓
- spec §7.2 未覆盖的 2 个: 已标注 `notes: > Not in spec §7.2` ✓

- [ ] **Step 5: 运行 fidelity 契约检查(不回归)**

```bash
node scripts/check-fidelity-contracts.js 2>&1
```

预期: 无新增 failure(与本次改动前一致)。

- [ ] **Step 6: 确认 badges 计数不变**

```bash
bash scripts/count-workflows.sh && cat badges/workflows.json
```

预期: 26(不增不减)。

- [ ] **Step 7: 提交(若有修正)**

如果验收发现任何问题,修正后提交。否则本 Task 无独立提交。

---

### Task 7: 创建 tier 修订提案

**Files:**
- Create: `docs/superpowers/specs/2026-06-15-tier-revision-proposal.md`

- [ ] **Step 1: 创建提案文档**

```markdown
# Tier 修订提案 — AccelOpt / KernelSkill / Xe-Forge

> **状态:** 提案,待 P5 决策
> **日期:** 2026-06-15
> **上游权威:** `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md` §7.2
> **背景 spec:** `docs/superpowers/specs/2026-06-15-multi-backend-declaration-sync-design.md`

---

## 背景

在 2026-06-15 的多后端声明层同步 review 中,用户(qhy991)基于实现证据对 spec §7.2 的 3 处
portability tier 判定提出了修订建议。本次声明层同步**按 spec §7.2 落盘**,分歧记录于此提案,
留待后续 P5 决策。

## 分歧 1: AccelOpt — vendor_locked vs clean

| 维度 | spec §7.2 | 用户 6/15 拍板 |
|------|-----------|---------------|
| Tier | vendor_locked | clean |
| Intrinsic to | NVIDIA NCU | — |
| Matrix | partial | true |
| 理由 | "P4 pilot, intrinsic to NCU" (§8 L763) | "prompt CUDA 残留是实现问题,不是方法约束;方法本质 clean" |

**证据:**
- AccelOpt 已接 driver 机制(`args.backend` + `_substrate/backends/${BACKEND}`)
- `accelopt-triton-dryrun.test.js` 已契约测试 triton driver 路径
- `.js` prompt 仍有 `.cu`/`__global__`/`PYBIND11_MODULE` 硬编码(L196, L501)—这是实现问题,不是方法约束
- 设计 spec §8 的 seam inventory 识别了 21 处 CUDA/NCU 耦合,但未区分"方法 intrinsic"vs"prompt 残留"

**建议:** 待 P5 prompt cleanup 完成后 re-evaluate;若 CUDA idiom 可被 driver 注入完全替代,则升为 clean。

## 分歧 2: KernelSkill — clean vs vendor_locked

| 维度 | spec §7.2 | 用户 6/15 拍板 |
|------|-----------|---------------|
| Tier | clean | vendor_locked |
| Intrinsic to | — | NVIDIA NCU |
| Matrix | yes | partial |
| 理由 | "clean/any" (§7.2 L735) | "无 triton dry-run + skill library + __global__ 词汇贯穿;方法本体绑 NCU" |

**证据:**
- `kernelskill-cuda-dryrun.test.js` 存在并断言**无 Triton token 泄漏**
- 无 `kernelskill-triton-dryrun.test.js`(对比 Generalist/StitchCUDA 均有)
- `.js` 使用 `ncu`+`nsys` profiler、CUDA skill library、`__global__` 词汇
- `_meta/manifests/kernelskill.yaml` 已定 `vendor_locked`、`supported: ["cuda"]`、`intrinsic_to: ""`(空)

**建议:** 若无 triton driver 契约测试计划,降为 vendor_locked/cuda/NCU;若有,需先补 triton-dryrun 测试。

## 分歧 3: Xe-Forge — vendor_locked vs method_intrinsic

| 维度 | spec §7.2 | 用户 6/15 拍板 |
|------|-----------|---------------|
| Tier | vendor_locked | method_intrinsic |
| Intrinsic to | Intel XPU | Intel XPU |
| Matrix | false | false |
| 理由 | "vendor_locked-single, ['xpu']" | "方法本质(VTune/XMX/SPIR-V)更像 intrinsic;硬件锚点强于工具锁" |

**证据:**
- `.js` `supported_languages: ['triton', 'sycl', 'xpu']` — triton/sycl 是编译前端,不是硬件锚点
- 无 xpu driver 注册,无 driver 路径
- 方法名 "CoVeR staged refinement" 是 Intel XPU 专用流程
- 两个 tier 在实际矩阵行为上等价(均为 `matrix_eligible: false`)

**建议:** 两个 tier 在实际行为上等价;建议统一到 method_intrinsic(硬件锚点语义更准确),或保留 vendor_locked 但将 `intrinsic_to` 从 "Intel XPU" 改为 "Intel XPU (VTune/XMX/SPIR-V)"。

---

## 影响面

3 处修订对矩阵 smoke CI 的影响:
- AccelOpt clean→vendor_locked: 无影响(当前 manifest 按 vendor_locked 落盘,matrix partial)
- KernelSkill clean→vendor_locked: 从 matrix yes 降为 partial(减少 CI 负载)
- Xe-Forge vendor↔method_intrinsic: 无影响(均为 matrix false)

## 决策流程

待 P5 系列迁移推进到对应 workflow 时,由 P5 lead 基于本文档 + 最新实现证据决定。
```

- [ ] **Step 2: 提交**

```bash
git add docs/superpowers/specs/2026-06-15-tier-revision-proposal.md
git commit -m "docs: add tier revision proposal for AccelOpt/KernelSkill/Xe-Forge

Records 3 portability tier disagreements between spec §7.2 and user 6/15
review — AccelOpt (vendor_locked vs clean), KernelSkill (clean vs
vendor_locked), Xe-Forge (vendor_locked vs method_intrinsic). Each entry
includes evidence, impact analysis, and recommendation. Decision deferred
to P5 lead.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 自审

- **Spec 覆盖:** §2 目标 5 项全部有对应 Task: README 矩阵(Task 1/2)✓, manifest 校验(Task 3)✓, manifest 新建(Task 4/5)✓, 过时副本标注(Task 5)✓, tier 提案(Task 7)✓
- **无占位符:** 所有 manifest 内容、README 矩阵值、prose 文本均为完整内容,无 TBD/TODO
- **类型一致性:** 所有 manifest 使用一致的 `schema_version: "1.1"` 格式,`backend:` 块字段名与 13 个现有 manifest 一致(`portability`/`matrix_eligible`/`method_supported_backends`/`supported`/`default`/`intrinsic_to`/`notes`)
- **计数:** 16 新建 manifest = Task 4(14 B/C/D) + Task 5(2 A 组) = 16 ✓