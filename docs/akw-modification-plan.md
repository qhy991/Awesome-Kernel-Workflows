# Awesome-Kernel-Workflows 修改方案

## 0. 目标

让 AKW 能够支持本次会话暴露出的三类任务形态:

1. **In-codebase kernel** (本场景: llama.cpp 内嵌的 `fattn-rdna-apu.cuh`) — 不允许把候选写到 exp_dir 独立编译, 必须用项目原生 build/test 命令
2. **AMD ROCm / RDNA APU 后端** — 需要 `rocprofv3` / `llvm-objdump` 而不是 `ncu`
3. **不允许工作流被 args 序列化 bug 静默吞掉而 fabricate 结果** — 这是 Round 1 GPUForecasters 幻觉的根因

按"最小侵入 / 复用现有 substrate / 不破坏 27 个工作流的现有契约"的原则, 改动分 4 层。

---

## 1. P0 修复: `args` 序列化防御 (阻塞性 bug, 影响全部 27 个工作流)

### 现状
KerSor 通过 `Workflow({ scriptPath, args })` 调用时, `args` 在某些路径下到达 JS 脚本时是 JSON 字符串, 不是对象。所有 AKW 脚本里的 `args.kernel_path` / `args.exp_dir` 等访问会静默返回 `undefined`, 触发兜底分支 (例如 CUDAAgent 抛 `Provide one of kernel_path...`, GPUForecasters 直接 fabricate 数据)。

### 修改
在 `_substrate/` 新增 `arg_guard.js`, 然后在每个工作流 `.js` 文件顶部加 6 行 import + 防御:

```js
// _substrate/arg_guard.js
export function unwrapArgs(args) {
  if (args == null) return {}
  if (typeof args === 'string') {
    try { return JSON.parse(args) } catch (e) {
      throw new Error(`Workflow args arrived as a non-JSON string: ${args.slice(0, 120)}`)
    }
  }
  if (typeof args === 'object') return args
  throw new Error(`Workflow args has unexpected type: ${typeof args}`)
}
```

每个工作流脚本顶部 (在 `export const meta` 之后, 第一次访问 `args.*` 之前) 插入:

```js
import { unwrapArgs } from '../_substrate/arg_guard.js'
args = unwrapArgs(args)
```

可以用一次性脚本批量改:
```bash
node scripts/patch-arg-guard.js  # 新增, 遍历 27 个 .js
```

### 验收
对 GPUForecasters / CUDAAgent / AKO4X 三个工作流分别用字符串 args 和对象 args 调一次, 行为应一致。

---

## 2. P0 新增: ROCm backend driver (本次任务的硬阻塞)

### 现状
`_substrate/backends/` 只有 `cuda/` 和 `triton/`。catalog 里 12 个工作流标 `backends: [rocm]` 但 substrate 不存在, AKO4X/AccelOpt 等依赖 `backend_dir` 的工作流会静默回退到 CUDA idioms 或直接失败。

### 修改
新增 `_substrate/backends/rocm/` (镜像 `cuda/` 的结构):

```
_substrate/backends/rocm/
├── manifest.json          # backend_id="rocm", compiler="hipcc" 或 "cmake", profiler="rocprofv3"
├── build.sh               # 支持 --build-cmd 模板 (本场景必需), 否则调 hipcc
├── run.sh                 # 运行 test-backend-ops 风格的 harness
├── profile.sh             # 包装 rocprofv3 (有则启用, 否则降级为纯 latency)
├── to_evidence.py         # rocprof CSV -> 通用 evidence schema
├── idioms.json            # RDNA 特有 idioms: wave64 vs wave32, AGPR, v_dot2_f32_f16
└── _evidence_amd.py       # 共享 AMD 计数器映射 (镜像 _evidence_nvidia.py)
```

`manifest.json` 关键字段:
- `requires_tools: ["hipcc"]`, `optional_tools: ["rocprofv3", "llvm-objdump"]`
- `bottleneck_classes: ["memory_bound","compute_bound","latency_occupancy","barrier_overhead"]`
- `precisions: ["fp32","fp16","bf16"]`
- `threshold_profile: "amd_rdna"`

`build.sh` 必须像 `cuda/build.sh` 一样支持 `--build-cmd "cmake --build ... --target ..."` 模板, 这是 in-codebase 场景的命门 — 不能默认 `hipcc -shared`。

`to_evidence.py` 最少要把 `rocprofv3` 的 `SQ_WAVES / VGPRs / AGPRs / TCP_TOTAL_CACHE_HITS_sum / TCP_TOTAL_CACHE_ACCESSES_sum / GRBM_GUI_ACTIVE` 映射到通用 `latency_ms / occupancy / l1_hit_pct / sm_pct`。无 rocprof 时只填 `latency_ms`, 其它字段为 `null` 而不是 0 (防止下游误判)。

注册到 `_substrate/backends/REGISTRY.md`:
```
| rocm | `rocm/` | amd | experimental | (unassigned) |
```

### 验收
`python3 _substrate/backends/validate_backend.py rocm` 通过 L0 conformance。

---

## 3. P1 新增: `in_place` 工作流 (本场景核心缺口)

### 现状
所有 27 个工作流的"评估单元"都是 `exp_dir/variants/<candidate>.cu`, 由工作流自己 `nvcc -shared` 出 `.so` 后 dlopen 跑。这种契约与 in-codebase kernel 完全不兼容: 你不能把 `fattn-rdna-apu.cuh` 复制到 `exp_dir/`, 因为它依赖 llama.cpp 的整套 include / template instantiation / dispatch。

### 修改

#### 3.1 新增工作流 `InPlacePatch/in-place-patch-optimization.js`

最小骨架 (~150 行):
```js
export const meta = {
  name: 'in-place-patch-optimization',
  description: 'Iterative patch -> project-native build -> test -> benchmark loop for kernels embedded in a larger codebase',
  whenToUse: 'When the kernel cannot be compiled standalone (depends on project headers, template machinery, or dispatch). Uses project build command verbatim, no exp_dir variant materialization.',
  phases: [
    { title: 'Snapshot', detail: 'Read kernel + backup original' },
    { title: 'Propose',  detail: 'Subagent proposes a single focused patch' },
    { title: 'Apply',    detail: 'Apply patch to original path in-place' },
    { title: 'Build',    detail: 'Run project build_command verbatim' },
    { title: 'Test',     detail: 'Run correctness command, parse pass/fail + NMSE' },
    { title: 'Bench',    detail: 'Run benchmark_command on N workload sizes' },
    { title: 'Decide',   detail: 'Keep if speedup>1 + correct; else revert' },
  ],
}
import { unwrapArgs } from '../_substrate/arg_guard.js'
args = unwrapArgs(args)

const KERNEL_PATH    = args.kernel_path        // required, absolute path inside project
const BUILD_CMD      = args.build_command      // required, no templating
const TEST_CMD       = args.test_command       // required
const BENCH_CMD      = args.benchmark_command  // required
const TOLERANCE      = args.tolerance || 5e-4
const MAX_ITER       = args.max_iterations || 5
const TARGET_SPEEDUP = args.target_speedup || 1.05
const HANDOFF        = args.op_description || ''

// 1. snapshot
const original = await readFile(KERNEL_PATH)

for (let i = 0; i < MAX_ITER; ++i) {
  // 2. propose
  const proposal = await agent(
    `Read ${KERNEL_PATH}. Previous attempts: ${JSON.stringify(history)}. ` +
    `Propose ONE focused patch. Apply it with Edit. Then STOP.`,
    { phase: 'Propose', schema: PATCH_SCHEMA }
  )
  // 3. build  4. test  5. bench  (all real Bash, fail-fast)
  // 6. if better and correct -> keep; else revert via Write(original)
}
return { best_kernel_code: bestCode, speedup: bestSpeedup, ... }
```

required_args 严格断言, 缺一即抛错 (不进 fabricate 路径)。

#### 3.2 manifest

新增 `_manifests/in_place_patch.yaml`:
```yaml
name: in-place-patch-optimization
js_path: InPlacePatch/in-place-patch-optimization.js
required_args: [kernel_path, build_command, test_command, benchmark_command]
all_args: [..., tolerance, max_iterations, target_speedup, op_description, handoff_context]
speedup_field: speedup
best_kernel_field: best_kernel_code
method_category: iterative_patch
fidelity_boundary: strict_high_fidelity
languages: [cuda, hip, cpp]
backends: [cuda, rocm]
problem_types: [in-codebase-kernel-optimization]
```

这样 KerSor 的 select-workflow.sh 在看到 `state.md` 里 `input_mode: kernel_file` + `kernel_path` 指向真实仓库内文件时, 才能选中它。

### 验收
在 llama.cpp 上, 直接调:
```js
Workflow({ name: 'in-place-patch-optimization', args: {
  kernel_path: '.../fattn-rdna-apu.cuh',
  build_command: 'cmake --build build-amd --config Release --target test-backend-ops',
  test_command:  'build-amd/bin/test-backend-ops.exe test -o FLASH_ATTN_EXT -b ROCm0',
  benchmark_command: 'build-amd/bin/test-backend-ops.exe perf -o FLASH_ATTN_EXT -p hsk=256 -b ROCm0',
  max_iterations: 3,
}})
```
应能完成 3 轮真实 build + test + bench, 任何一步失败立即 revert 而不是 fabricate。

---

## 4. P1 修改: GPUForecasters 等"易幻觉"工作流加 grounding gate

### 现状
GPUForecasters 工作流要求子代理跑 `surrogate_train.py` / `calibrate_abstention.py` / `refinement_search.py`, 但这些脚本不存在 (是论文术语, 不是真实工具)。子代理跑 `ls` 发现没有, 但工作流用 schema 强制结构化 JSON 输出, 于是 LLM 编一份。这是 Round 1 的根因。

### 修改
两步:

#### 4.1 给每个有 phase 子代理的工作流加 "grounding contract"

在 `_substrate/` 新增 `grounding.js`:
```js
export const GROUNDING_INSTRUCTION = `
GROUNDING CONTRACT (mandatory):
- Before reporting any numeric result, you MUST have executed a real Bash command that produced it.
- If the prescribed tool/script does not exist, return {"grounded": false, "missing": "<what>"} and STOP. Do NOT invent values.
- If you call Bash and the command fails (non-zero exit), return {"grounded": false, "error": "<stderr tail>"}.
- Schemas accepting numeric fields will be rejected if "grounded": false. Do not work around this.
`
```

每个工作流的 agent() 调用 prompt 末尾追加 `GROUNDING_INSTRUCTION`, schema 增加可选 `grounded: boolean` 和 `missing: string`。result-analyzer 看到 `grounded: false` 就把整轮标记为 `verdict: not_grounded` 而不是 `failed`, 让 KerSor 的 selector 知道是"工作流-环境不匹配", 不要再选它。

#### 4.2 GPUForecasters 特别处理

它的 `surrogate_train.py` 等是论文术语而非真实工具, 应改成 "如果用户没提供 `surrogate_train_command` 等 args, 工作流直接抛错退出" (不试图让子代理跑 ls 后自己编)。

### 验收
重跑 Round 1 GPUForecasters, 应得到 `verdict: not_grounded` 而不是 fabricated numbers。

---

## 5. P2 新增: `rocprofv3` evidence agent + `llvm-objdump` disasm agent

### 现状
KerSor 的 `kernel-profiler` 知道 `rocprofv3` 存在但只是写进 profile 字段, 没有任何工作流真的去跑它并把输出回灌给优化决策。本次会话 Round 3 已经卡在"3% 残差 = 编译器级"的诊断点, 缺这两个工具就无法继续。

### 修改
两个独立小工作流, 不进 selector 默认池, 由其他工作流主动调用:

#### 5.1 `_tools/rocprof-counters/rocprof-counters.js`
- input: `kernel_path`, `benchmark_command`, `counters` (默认 `SQ_WAVES,SQ_INSTS_VALU,VALUInsts,VFetchInsts,VWriteInsts,TCP_TOTAL_CACHE_HITS_sum,TCP_TOTAL_CACHE_ACCESSES_sum`)
- output: `{ counters: {...}, vgpr_count: N, agpr_count: N, occupancy: 0..1 }`

#### 5.2 `_tools/amdgcn-disasm/amdgcn-disasm.js`
- input: `build_artifact` (so/dylib 路径), `kernel_name_regex`
- 跑 `llvm-objdump -d --disassemble-symbols=...`
- output: `{ vmem_stall_cycles: N, valu_stall_cycles: N, barrier_count: N, longest_dep_chain: N }`

这两个是 InPlacePatch / AKO4X 在 ROCm 后端的"证据闭环"信息源。

---

## 6. P2 修改: catalog + selector 协议

### 现状
KerSor 的 `select-workflow.sh` 用 `backends/method_category/fidelity_boundary` 打分, 但不验证工作流是否能"真正被 grounding" (例如缺 `build_command` 必填 arg)。

### 修改
在 `_manifests/schema.yaml` 加字段:
```yaml
grounding_requirements:
  type: object
  properties:
    needs_project_build_command:  {type: boolean}
    needs_real_benchmark_command: {type: boolean}
    needs_profiler:               {enum: [none, ncu, rocprofv3]}
    needs_disasm:                 {enum: [none, cuobjdump, llvm-objdump]}
```

每个工作流的 manifest 声明自己的 grounding_requirements。 KerSor 侧的 select-workflow.sh 把它和 `test-method.md` / 环境探测结果做合规检查 — 不满足直接过滤, 不进打分阶段。这样 GPUForecasters 在缺 `surrogate_train_command` 时会被自动排除, 不会再被选中。

(这一步在 KerSor repo 里也要对应改 `select-workflow.sh` 和 selector 算分逻辑, 不属于 AKW 改动范围, 此处仅声明契约。)

---

## 7. 优先级与拆分建议

| 优先级 | 改动 | 估算工作量 | 解锁的能力 |
|--------|------|------------|------------|
| **P0** | §1 arg_guard | 2 小时 | 27 个工作流不再静默 fabricate |
| **P0** | §2 ROCm backend driver | 1 天 | 所有 ROCm 标注的工作流可真正运行 |
| **P1** | §3 InPlacePatch 工作流 | 1 天 | llama.cpp 这类 in-codebase kernel 可被自动优化 |
| **P1** | §4 grounding contract | 半天 | 杜绝 GPUForecasters 式幻觉 |
| **P2** | §5 rocprof / disasm tools | 1 天 | 突破"3% 编译器级残差"诊断点 |
| **P2** | §6 catalog 协议升级 | 半天 + KerSor 联动改动 | selector 不再选错工作流 |

P0 两项做完, AKW 至少不会再"用幻觉浪费一整轮"。P1 做完, llama.cpp 的 in-codebase kernel 优化进入闭环。P2 做完, 才能继续追小于 5% 的残差。

---

## 8. 边界 / 不做的事

- 不重写现有 27 个工作流的核心算法 (KSearch 的 MCTS、KernelFoundry 的 MAP-Elites 等都保留)
- 不引入新的依赖 (rocprof / llvm-objdump 都是 ROCm SDK 自带)
- 不破坏现有 cuda/triton 后端 (新增 rocm/ 是平行目录)
- 不替换 KerSor 的 selector (只在 AKW 侧声明 grounding 契约, KerSor 是否消费由对方决定)

---

## 9. 决策点

在动手前需要你确认:
- (a) ROCm backend 是只支持 RDNA 3.5 + cmake 构建, 还是要兼容 CDNA + hipcc 单文件构建?
- (b) InPlacePatch 工作流是否要支持"多文件 patch" (例如同时改 `fattn-rdna-apu.cuh` 和 `fattn.cu`), 还是只单文件?
- (c) P0 / P1 / P2 是分批做, 还是一次性提一个 PR?
