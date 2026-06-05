# 变更日志 — `_substrate/`

KerSor Solver SDK 共享底座的显著变更记录。格式 loosely 遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [2026-06-05] Backend Driver 基础设施（P1 + P2）

- **分支**：`dev/solver-substrate`
- **计划**：[2026-06-05-backend-driver-foundation.md](../docs/superpowers/plans/2026-06-05-backend-driver-foundation.md)
- **规格**：[2026-06-05-backend-driver-axis-design.md](../docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md)

全程 TDD（红 → 绿 → 提交），subagent 驱动执行。**36 个测试全部通过**
（`python3 -m unittest discover -s _substrate/tests -p 'test_*.py'`）。

### 新增

#### 测试基础设施（Task 0）

- `_substrate/tests/test_smoke.py` — discovery harness 冒烟测试；一条命令跑完全部用例。
- `_substrate/tests/test_diagnose.py` — golden 表征测试（NVIDIA 字节一致）、null-rule、
  vendor profile、occ 优先级、CLI 返回码覆盖。
- `_substrate/tests/test_anti_cheat.py` — golden 表征测试（CUDA 无文件路径字节一致）、
  `--vendor-patterns-file` 集成（Metal MPS + C++ stub）、`load_vendor_patterns` 单元测试。
- `_substrate/tests/test_validate_backend.py` — 1 个合规 + 4 个缺陷 fixture 路径、
  非 dict 顶层 JSON 拒绝、exit-3 参数错误守卫。
- `_substrate/tests/fixtures/{good,bad_*}/` — 10 个 JSON fixture（1 合规 driver +
  4 种单点缺陷变体），供 L0 校验使用。

#### Backend driver 轴（Task B1–B4）

- `_substrate/backends/validate_backend.py` — 纯 stdlib 的 L0 结构校验器，校验 driver
  目录（`manifest.json` + `idioms.json`）。`import method_gate.TABLE` 读取实时 method
  名以校验 idiom 引用。退出码：`0` 通过 · `1` L0 错误 · `3` 参数错误。
- `_substrate/backends/REGISTRY.md` — 面向人类的 driver 索引表；已 seed `cuda`（`planned`）。
- `_substrate/BACKEND-DRIVER-SDK.md` — 六文件 driver 契约、manifest/idioms 字段表、
  L0–L3 一致性阶梯、Part A 底座改动交叉引用，以及 **相对 spec 的有据偏离**
  （JSON 非 YAML；Python 校验器非 Node/JSON-Schema；`backend_id` 规范形校验延后）。

#### 设计文档

- `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md` — backend driver 轴
  完整设计规格（语言 × vendor 翻译层）。
- `docs/superpowers/plans/2026-06-05-backend-driver-foundation.md` — 可执行的 P1+P2
  实现计划（1554 行、9 个任务、Definition of Done）。

### 变更

#### `diagnose.py`（Task A1 + A2）

- **Vendor 阈值 profile** — 新增 `PROFILES` 映射，由 metrics 中的 `_vendor` 键选择
  （默认 `nvidia`；新增 `apple`）。不带 `_vendor` 的既有 NVIDIA 输入保持 **字节级一致**
  （golden 测试锁定）。
- **部分指标的 null-rule** — 当双侧判别器（`memory_bound` / `overhead_bound`）需要
  `dram_pct` 与 `sm_pct` 同时存在、但只测到一侧时，返回 `unknown` 及
  `"insufficient measured metrics"`，不再从单侧操作数推断。
- **优先级不变量保留** — 当 occ 与 compute 信号同时存在时，`latency_occupancy`（低
  `occupancy`）仍优先于 `compute_bound`（测试已 pin）。

#### `anti_cheat.py`（Task A3 + A4）

- **`--vendor-patterns-file`** — 可选的 per-backend 作弊模式文件，含 `[fallback]` 与
  `[skip]` 两段（spec §5.3.3）。模式与内置 CUDA 列表合并；默认无文件路径保持 **字节级一致**。
- **`load_vendor_patterns()`** — 解析分段 regex 文件；加载时用 `re.compile` 校验每条
  正则（无效模式在加载期抛 `ValueError`，避免在 `static_flags` 中晚到 traceback）。

#### `validate_backend.py` 加固（代码审查跟进）

- `_load_json` 捕获 `OSError` / `PermissionError` — Bash 调用方始终在 stdout 拿到 JSON，
  不会出现裸 traceback。
- 拒绝非 dict 顶层 JSON（`null`、`[]`、`"str"`），给出明确 L0 错误。
- exit-3 路径（driver 目录不存在）已有测试覆盖。

### 未改动（明确非目标）

按计划 Definition of Done，以下底座脚本保持 **字节级一致**：

- `method_gate.py`
- `evidence_schema.py`
- `memory_store.py`
- `verify_insight.py`

未触及任何 workflow `.js` 文件。

### 相对 spec 的有据偏离（有意为之；P3 须继承）

| 主题 | spec 写法 | 本次交付 |
|---|---|---|
| 机读 driver 文件 | `manifest.yaml` / `idioms.yaml` | `manifest.json` / `idioms.json`（stdlib `json`，无 PyYAML/Node 依赖） |
| L0 校验器 | `_meta/tools/validate-backend.js` + JSON-Schema | `_substrate/backends/validate_backend.py` 手写 stdlib 检查 |
| `backend_id` L0 校验 | 三方 `== dir == normalizeSuitabilityValue(id)` | 仅字面量 `backend_id == basename(dir)`；规范形规则延后至 P3 |
| `bottleneck_classes` | 4 个有意义类 ∪ `{unknown}` | 校验器允许声明列表中出现 `unknown`（审查修复 #4） |

### 提交映射（14 个提交，`00b920b..HEAD`）

| 提交 | 摘要 |
|---|---|
| `0bb3a8e` | docs: backend-driver 轴设计规格 |
| `cec9010` | docs: backend-driver 基础设施 P1+P2 计划 |
| `13b6163` | 新增 `_substrate/tests` unittest discovery harness |
| `c0986b6` | 新增 diagnose golden + null-rule 表征测试（RED） |
| `b9914e1` | 实现 diagnose vendor profile + measured-operand null-rule（GREEN） |
| `0681ac3` | 补测 occ 优先级、apple vendor profile、CLI returncode |
| `a81493e` | 新增 anti_cheat golden + vendor-pattern 表征测试（RED） |
| `28c9642` | 实现 anti_cheat `--vendor-patterns-file`（GREEN） |
| `9c7ac6b` | vendor-pattern 加载时校验 regex；补 `load_vendor_patterns` 单元测试 |
| `31df6f5` | 新增 backend L0 校验器 fixture 与失败测试（RED） |
| `c3fde1c` | 实现确定性 L0 backend driver 校验器（GREEN） |
| `23bba53` | 加固 `validate_backend`：捕获 OSError、拒绝非 dict JSON、测试 exit-3 |
| `33ddb05` | 新增 backend driver 注册表，seed `cuda` |
| `a324efc` | 新增 backend driver SDK 契约文档 |

### 后续（P3+，不在本次发布范围）

- 第一个真实 driver 目录：`_substrate/backends/cuda/`（六文件布局）。
- 将 `normalizeSuitabilityValue` 规范形检查移植进 L0 校验器。
- 按 `BACKEND-DRIVER-SDK.md` 补齐 L1–L3 一致性冒烟 fixture。
