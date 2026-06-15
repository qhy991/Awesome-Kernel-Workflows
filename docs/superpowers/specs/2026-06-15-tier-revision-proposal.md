# Tier 修订提案 — AccelOpt / KernelSkill / Xe-Forge

> **状态:** 提案，待 P5 决策
> **日期:** 2026-06-15
> **上游权威:** `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md` §7.2
> **背景 spec:** `docs/superpowers/specs/2026-06-15-multi-backend-declaration-sync-design.md`

---

## 背景

在 2026-06-15 的多后端声明层同步 review 中，用户（qhy991）基于实现证据对 spec §7.2 的 3 处
portability tier 判定提出了修订建议。本次声明层同步**按 spec §7.2 落盘**，分歧记录于此提案，
留待后续 P5 决策。

## 分歧 1: AccelOpt — vendor_locked vs clean

| 维度 | spec §7.2 | 用户 6/15 拍板 |
|------|-----------|---------------|
| Tier | vendor_locked | clean |
| Intrinsic to | NVIDIA NCU | — |
| Matrix | partial | true |
| 理由 | "P4 pilot, intrinsic to NCU" (§8 L763) | "prompt CUDA 残留是实现问题，不是方法约束；方法本质 clean" |

**证据:**
- AccelOpt 已接 driver 机制（`args.backend` + `_substrate/backends/${BACKEND}`）
- `accelopt-triton-dryrun.test.js` 已契约测试 triton driver 路径
- `.js` prompt 仍有 `.cu`/`__global__`/`PYBIND11_MODULE` 硬编码（L196, L501）— 这是实现问题，不是方法约束
- 设计 spec §8 的 seam inventory 识别了 21 处 CUDA/NCU 耦合，但未区分"方法 intrinsic" vs "prompt 残留"

**建议:** 待 P5 prompt cleanup 完成后 re-evaluate；若 CUDA idiom 可被 driver 注入完全替代，则升为 clean。

## 分歧 2: KernelSkill — clean vs vendor_locked

| 维度 | spec §7.2 | 用户 6/15 拍板 |
|------|-----------|---------------|
| Tier | clean | vendor_locked |
| Intrinsic to | — | NVIDIA NCU |
| Matrix | yes | partial |
| 理由 | "clean/any" (§7.2 L735) | "无 triton dry-run + skill library + `__global__` 词汇贯穿；方法本体绑 NCU" |

**证据:**
- `kernelskill-cuda-dryrun.test.js` 存在并断言**无 Triton token 泄漏**
- 无 `kernelskill-triton-dryrun.test.js`（对比 Generalist/StitchCUDA 均有）
- `.js` 使用 `ncu`+`nsys` profiler、CUDA skill library、`__global__` 词汇
- `_meta/manifests/kernelskill.yaml` 已定 `vendor_locked`、`supported: ["cuda"]`、`intrinsic_to: ""`（空）

**建议:** 若无 triton driver 契约测试计划，降为 vendor_locked/cuda/NCU；若有，需先补 triton-dryrun 测试。

## 分歧 3: Xe-Forge — vendor_locked vs method_intrinsic

| 维度 | spec §7.2 | 用户 6/15 拍板 |
|------|-----------|---------------|
| Tier | vendor_locked | method_intrinsic |
| Intrinsic to | Intel XPU | Intel XPU |
| Matrix | false | false |
| 理由 | "vendor_locked-single, ['xpu']" | "方法本质（VTune/XMX/SPIR-V）更像 intrinsic；硬件锚点强于工具锁" |

**证据:**
- `.js` `supported_languages: ['triton', 'sycl', 'xpu']` — triton/sycl 是编译前端，不是硬件锚点
- 无 xpu driver 注册，无 driver 路径
- 方法名 "CoVeR staged refinement" 是 Intel XPU 专用流程
- 两个 tier 在实际矩阵行为上等价（均为 `matrix_eligible: false`）

**建议:** 两个 tier 在实际行为上等价；建议统一到 method_intrinsic（硬件锚点语义更准确），或保留 vendor_locked 但将 `intrinsic_to` 从 "Intel XPU" 改为 "Intel XPU (VTune/XMX/SPIR-V)"。

---

## 影响面

3 处修订对矩阵 smoke CI 的影响：
- AccelOpt clean→vendor_locked: 无影响（当前 manifest 按 vendor_locked 落盘，matrix partial）
- KernelSkill clean→vendor_locked: 从 matrix yes 降为 partial（减少 CI 负载）
- Xe-Forge vendor↔method_intrinsic: 无影响（均为 matrix false）

## 决策流程

待 P5 系列迁移推进到对应 workflow 时，由 P5 lead 基于本文档 + 最新实现证据决定。