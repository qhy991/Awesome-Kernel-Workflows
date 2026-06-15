# 多后端声明层全量同步 — 设计文档

> **状态:** Draft,待用户复审
> **日期:** 2026-06-15
> **范围:** 方案 B(声明层全量同步)+ tier 修订提案(另出)
> **上游权威:** `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md` §7.2

---

## 1. 背景与问题

仓库正在进行一套 **Backend Driver 横切轴**迁移(spec `2026-06-05-backend-driver-axis-design.md`,
P4 pilot → P5a/b/c/d 系列 plans,最近更新 6/9)。该迁移把 `(源语言) × (硬件/profiler 厂商)`
从 workflow body 抽到 `_substrate/backends/<id>/` 可插拔 driver,workflow 通过 `args.backend` +
`args.backend_dir` 切换。

迁移已推进到**代码层**(多数 Tier① workflow 已接 driver 机制),但**声明层严重滞后**:

| 载体 | 状态 |
|---|---|
| `README.md` / `README.zh-CN.md` 适配矩阵(L141–170) | **过时** — 16 个已接 driver 的 workflow 仍标单后端;缺 InPlacePatch/Llamacpp 两行 |
| `<Workflow>/manifest.yaml` 的 `backend:` 块(SoT) | 13 个已有,内容多数正确但未系统校验 |
| 16 个无目录 manifest 的 workflow | 缺口,声明散落在 `.js` 的 `WORKFLOW_SUITABILITY`/`supported_languages` |
| 顶层 `_manifests/` + `_meta/manifests/` 旧格式副本 | 过时,与 SoT 并存,易误导 |

**核心矛盾:** 代码已多后端、声明仍是单后端;manifest SoT 与 README 矩阵不一致;存在新旧
manifest 格式与多份过时副本。

## 2. 目标

**只做声明层一致性,不改任何 workflow 的 prompt / 行为 / portability tier 判定。**

具体:
1. 重写 README 适配矩阵(中英双语),逐行对齐各 workflow 的实际 backend 能力,加诚实标注
   (driver / experimental / method-intrinsic / legacy)。
2. 校验并(必要时)修正 13 个已有 `<Workflow>/manifest.yaml` 的 `backend:` 块,使其与 `.js`
   实际 + 上游 spec §7.2 一致。
3. 给 16 个无目录 manifest 的 workflow 新建最小 manifest(仅 `backend` 块 + `source`/`workflow`
   头,不补 `phases`/`method` 长文)。
4. 过时副本目录 `_manifests/` + `_meta/manifests/` 加 DEPRECATED 指向,标明 SoT 在
   `<Workflow>/manifest.yaml`。
5. **另出** `docs/superpowers/specs/2026-06-15-tier-revision-proposal.md`,记录 3 处 tier 分歧
   (§6),不在本次擅自改 spec §7.2。

## 3. 核心执行原则(方案 B 的推论)

| 字段 | 取值来源 |
|---|---|
| `portability` tier | **上游 spec §7.2**(权威,本次不改) |
| `intrinsic_to` | **上游 spec §7.2** |
| `supported` 列表 | **fidelity-first 对齐 `.js` 实际**(`WORKFLOW_SUITABILITY`/`supported_languages`) |
| `default` | `.js` 实际(`default_backend` 或 `args.language`/`args.backend` 默认值) |
| `matrix_eligible` | spec §7.2(clean/vendor_locked-partial=true 或 partial;method_intrinsic=false) |

> **legacy vs driver 双轨:** 目录 `manifest.yaml` 描述 **driver 路径下的声明能力**;`.js` 的
> `WORKFLOW_SUITABILITY` 在 `USE_DRIVER`(有 `backend_dir`)时被跳过,仅代表 legacy 路径守卫。
> 两者短期允许不一致;manifest 反映 driver 能力,README 标注 legacy 差异。

## 4. 27 个 workflow 的 backend 声明(对齐 spec §7.2 + `.js` 实际)

> spec §7.2 原文权威分级见 `2026-06-05-backend-driver-axis-design.md:733-750`。下表 `supported`/
> `default` 列来自本次对 `.js` 的核实。

### A 组 — clean(matrix-eligible)— 15 个

| Workflow | supported(default) | manifest 现状 | 本次动作 |
|---|---|---|---|
| AdaExplore | `[cuda, triton]`(cuda) | ✅ 一致 | 校验 |
| KernelAgent | `[cuda, triton]`(triton) | ✅ | 校验 |
| AKO4X | `[triton, cuda, cute-dsl, tilelang, cpp, pytorch]`(triton) | ✅ | 校验 |
| Astra | `[cuda, triton]`(cuda) | ✅ | 校验 |
| CUDALLM | `[cuda, triton]`(cuda) | ✅ | 校验 |
| KDA | `[cuda, triton]`(cuda) | ✅ | 校验 |
| KSearch | `[triton, cuda, python]`(triton) | ✅ | 校验 |
| ReGraphT | `[cuda, triton]`(cuda) | ✅ | 校验 |
| STARK | `[cuda, triton]`(cuda) | ✅ | 校验 |
| StitchCUDA | `[cuda, triton]`(cuda) | ✅ | 校验;note: KernelBench 交集守卫要求 cuda driver |
| KernelFoundry | `[cuda, triton]`(cuda);SYCL 经 `args.backend_dir` 自带 | ❌ 缺 | **新建** |
| KernelFoundryDx | `[triton]`(triton) · matrix `partial` | ✅ | 校验 |
| KernelSkill | `[cuda]`(cuda) | ❌ 缺;spec=clean/any,但 `.js` 仅 cuda + 无 triton dry-run | **新建,按 spec clean/any 落盘**;分歧记 §6 |
| KernelBand | `[triton, cuda]`(triton);φ-gate NVIDIA 利用率 | ✅ | 校验;note: 阈值 driver-resolved |
| Generalist | `[cuda, triton]`(cuda);legacy path 仅 cuda | ⚠️ 现 manifest `[cuda]` | **改 → `[cuda, triton]`/any**;note: legacy path 守卫至 P5 同步 |

### B 组 — vendor_locked(matrix partial)— 8 个

| Workflow | intrinsic_to | supported(default) | 本次动作 |
|---|---|---|---|
| AccelOpt | NVIDIA NCU | `[cuda, triton]`(cuda) | **新建**;⚠️ 用户拍板 clean,spec=vendor_locked — 按 **spec vendor_locked** 落盘,分歧记 §6 |
| CUDAAgent | NVIDIA NCU | `[cuda]`(cuda) | 新建 |
| cuPilot | NVIDIA NCU | `[cuda]`(cuda) | 新建 |
| KEET | NVIDIA NCU | `[cuda]`(cuda) | 新建 |
| KernelBlaster | NVIDIA NCU | `[cuda]`(cuda) | 新建 |
| GPUForecasters | NVIDIA NCU | `[cuda]`(cuda) | 新建 |
| TritorX | Triton dialect + linter | `[triton]`(triton) · matrix `false` | 新建 |
| Xe-Forge | Intel XPU | `[xpu]`(xpu) · matrix `false` | 新建;⚠️ 用户拍板 method_intrinsic,spec=vendor_locked — 按 **spec vendor_locked** 落盘,分歧记 §6 |

### C 组 — method_intrinsic(matrix false)— 4 个(spec §7.2)

| Workflow | intrinsic_to | supported | 本次动作 |
|---|---|---|---|
| ArchAgent | ChampSim IPC(LLC cache policy) | `[cpp]` | 新建 |
| CutlassGEMM | CUTLASS device-level API | `[cutlass]` | 新建 |
| FACT | CUTLASS pattern registry | `[cutlass]` | 新建 |
| ARGUS | AMD MI300X MFMA invariants | `[rocm]`(spec 目标);`.js` 实际 `[cuda, rocm, triton, argus-dsl]` | 新建;note: legacy `args.language`,driver 迁移 pending |

### D 组 — spec §7.2 未覆盖(本次按 `.js` 实际归类,标注需回填 §7.2)— 2 个

| Workflow | 拟定 tier | 依据 | 本次动作 |
|---|---|---|---|
| InPlacePatch | `vendor_locked`(nvcc/hipcc) | `.js` 提及 nvcc/hipcc 编译单 `.so`,byte-exact CUDA patch 语义 | 新建;**标注:需回填 spec §7.2** |
| LlamacppEmbeddedSearch | `method_intrinsic`(llama.cpp ggml-cuda) | `.js` 明确 kernel 嵌入 llama.cpp,非独立编译 | 新建;**标注:需回填 spec §7.2** |

**计数核对:** 15 clean + 8 vendor_locked + 4 method_intrinsic + 2 未覆盖 = **29 workflow** ✓

## 5. 具体落盘动作

### 5.1 README 适配矩阵重写(中英双语)

- **列结构不变**(5 列:Workflow | Supported language/backend | problem_type | Good fit | Avoid when)。
- **只改 "Supported language/backend" 列**的值,按类型套用统一措辞:

| 类型 | 措辞模板 |
|---|---|
| clean 多后端(driver) | `CUDA (default) · Triton via driver (experimental)` |
| clean 单后端(legacy only) | `CUDA (legacy; driver migration pending)` |
| vendor_locked | `CUDA (vendor-locked: ncu)` / `Triton (vendor-locked: linter)` |
| method_intrinsic | `Intel XPU (method-intrinsic)` / `C++ / ChampSim (method-intrinsic)` |
| ARGUS | `ROCm/CUDA/Triton/ARGUS-DSL (legacy; driver pending)` |

- **补 2 行**(InPlacePatch、LlamacppEmbeddedSearch),矩阵从 27 → 29 行 + Meta-Workflow。
- 中英两版逐行对齐。

### 5.2 manifest 校验与新建

- **A 组 13 个**:逐个核对 `backend:` 块与 `.js` 实际 + spec §7.2;修正 Generalist(`[cuda]`→`[cuda,triton]`/any)。
- **B/C/D 组 16 个**:用最小模板新建 `<Workflow>/manifest.yaml`(见 5.3)。
- 校验手段:`scripts/check-fidelity-contracts.js` 覆盖有限,**不替代**人工 backend 核对;辅以既有
  `_meta/tools/test/*-guard.test.js` / `*-dryrun.test.js` 作为 driver 能力证据。

### 5.3 新 manifest 最小模板

对齐 `_substrate/tests/fixtures/manifest_v11_minimal.yaml` + 13 个现有新格式头部:

```yaml
schema_version: "1.1"
name: <workflow-name>
entrypoint: <workflow>.js

backend:
  portability: method_intrinsic | vendor_locked   # 按 spec §7.2
  matrix_eligible: false | partial                 # 按 spec §7.2
  intrinsic_to: <anchor>                           # portability != clean 时必填
  method_supported_backends: <any | [list]>        # spec §7.2
  supported: [<fidelity-first 对齐 .js>]
  default: <.js 默认>
  notes: "<迁移/caveat 说明,若有>"

source:
  paper_title: "<from _manifests 或 README, 仅 verbatim>"
  paper_url: "<if known>"
  notes: "Method-intrinsic; full phase spec deferred."

workflow:
  name: "<同 name>"
  description: "<one line from meta.description>"
  when_to_use: "<one line from meta.whenToUse>"
  output_filename: "<entrypoint>"
  directory: "<WorkflowDir>/"
```

**不补** `phases` / `plan_angles` / `topology` / `args` 全量(YAGNI + fidelity)。例外:若该 workflow
已有经审计的富 manifest 可搬运,只搬 `source`+`workflow` 头,**不搬 phases**(避免双 SoT)。

### 5.4 过时副本标注

- `_manifests/README.md`(新建)+ `_meta/manifests/README.md`(新建)加 DEPRECATED 段:
  "本目录为历史归档/草稿,**SoT 为各 `<Workflow>/manifest.yaml`**。`_meta/manifests/` 含迁移中
  草稿(generalist/kernelfoundry/kernelskill 已转新格式,其余待迁)。"
- **不删**文件(避免破坏既有引用)。

## 6. 3 处 tier 分歧(另出修订提案,本次不改)

| Workflow | spec §7.2 | 用户 6/15 拍板 | 本次落盘 | 修订提案记录 |
|---|---|---|---|---|
| AccelOpt | vendor_locked / NCU | clean | **vendor_locked**(spec) | 用户认为 prompt CUDA 残留是实现问题,方法 clean;待 P5 决策 |
| KernelSkill | clean / any | vendor_locked / cuda / ncu | **clean**(spec) | 无 triton dry-run + skill gate 证据建议收紧;待 P5 决策 |
| Xe-Forge | vendor_locked / xpu | method_intrinsic / xpu | **vendor_locked**(spec) | 方法本质(VTune/XMX/SPIR-V)更像 intrinsic;待 P5 决策 |

提案文档:`docs/superpowers/specs/2026-06-15-tier-revision-proposal.md`(含每项的 spec 原值、
新证据、影响面、建议,但不定论)。

## 7. 不在本次范围(YAGNI / scope 外)

- 不改任何 `.js` 的 `WORKFLOW_SUITABILITY` / prompt / CUDA idiom 残留(那是 P5 代码层迁移)。
- 不把 ARGUS 从 legacy `args.language` 迁到 driver(代码层)。
- 不重构 "language vs backend id" 口径混乱(README/manifest 混用 CUDA/Triton/SYCL/CUTLASS)。
- 不做 driver 真实 GPU/NPU 端到端验证(无硬件;deferred to CI tier,见 REGISTRY.md)。
- 不改 spec §7.2(经 §6 修订提案流程)。
- 不动 `badges/workflows.json`(纯计数徽章,无关 backend)。

## 8. 校验与验收

1. 每个 Tier① workflow:`<Workflow>/manifest.yaml` 的 `supported` ↔ `.js` 实际 ↔ spec §7.2 三方一致(或分歧入 §6)。
2. 中英 README 矩阵逐行与 manifest `backend:` 块对齐;补齐 29 行。
3. `node scripts/check-fidelity-contracts.js` 不回归(若有覆盖)。
4. 16 个新 manifest 通过 `_substrate/backends/validate_backend.py` 的 L0 结构校验(若 manifest 校验接入)。
5. 过时副本 README 存在且指向 SoT。
6. `scripts/count-workflows.sh` + `badges/workflows.json` 计数不变(本次不增减 workflow)。

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 新 manifest 的 `source`/`description` 凭空写,违反 fidelity | 仅 verbatim 搬运既有 `_manifests`/README/`meta.description`,不杜撰;method-intrinsic 标 "full phase spec deferred" |
| 16 个新 manifest 与未来 P5e/f 真正迁移时冲突 | 用最小模板,明确标 deferred;tier 一律跟 spec §7.2,P5 迁移时以代码为准覆写 |
| README 中英不同步 | 作为验收项 2 强制逐行对齐 |
| 改动面大(2 README + 16 manifest + 13 校验 + 2 副本 README) | 分批提交:README 矩阵 → A 组校验 → B/C/D 新建 → 副本标注 → 提案 |
