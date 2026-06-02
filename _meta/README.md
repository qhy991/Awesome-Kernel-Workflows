# Meta-Workflow: Paper → Workflow Generation Pipeline

**English** · [简体中文](README.zh-CN.md)

This directory contains the **meta-workflow** infrastructure — tools that automate the process of turning a research paper or repository into a validated Claude Code Workflow.

## What is the Meta-Workflow?

The meta-workflow is itself a Claude Code Workflow that:

1. **Researches** a paper/repo to understand its optimization methodology
2. **Models** the loop topology, phases, agents, args, and plan angles
3. **Assembles** a structured YAML manifest conforming to the schema
4. **Generates** a complete `.js` workflow from a template + manifest
5. **Validates** the generated workflow for correctness

```
Paper URL / Repo URL
        │
        ▼
┌───────────────────┐
│   Research Phase   │  Parallel: analyze paper + analyze repo
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    Model Phase     │  Parallel: topology + phases + args + angles
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Assemble Phase   │  Compose manifest YAML from structured data
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Generate Phase   │  Template + manifest → .js workflow file
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Validate Phase   │  Static + semantic checks → fix if needed
└─────────┬─────────┘
          │
          ▼
    Workflow (.js) + Manifest (.yaml)
```

## Directory Structure

```
_meta/
├── README.md                    # This file
├── tools/
│   ├── generate-workflow.js     # Meta-workflow: paper → manifest → workflow
│   └── validate-workflow.js     # Validation workflow: checks a .js workflow
├── skills/
│   └── claude-workflow-gotchas/
│       └── SKILL.md             # Workflow authoring pitfalls and hard rules
├── templates/
│   ├── iterative-loop.js        # Template for Plan→Execute→Profile→Learn loops
│   ├── search-based.js          # Template for population/evolutionary search
│   ├── single-pass.js           # Template for Analyze→Transform→Verify pipelines
│   └── tree-exploration.js      # Template for tree-of-thought / beam search
└── manifests/
    ├── schema.yaml              # Manifest schema definition (all fields documented)
    ├── accelopt.yaml            # AccelOpt manifest (reference example)
    ├── keet.yaml                # KEET manifest
    ├── argus.yaml               # ARGUS manifest
    └── ksearch.yaml             # KSearch manifest
```

## Usage

### Generate a new workflow from a paper

```javascript
Workflow({name: 'generate-workflow', args: {
  paper_url: 'https://arxiv.org/abs/XXXX.XXXXX',
  repo_url: 'https://github.com/org/repo',          // optional
  output_dir: '/path/to/Awesome-Kernel-Workflows',
  method_name: 'MethodName',                          // optional override
}})
```

The workflow will:
1. Fetch and analyze the paper to extract methodology
2. Classify the loop topology (iterative / search / tree / pipeline)
3. Generate a manifest YAML and a `.js` workflow file
4. Run validation and auto-fix issues

### Validate an existing workflow

```javascript
Workflow({name: 'validate-workflow', args: {
  workflow_path: '/path/to/my-workflow.js',
  strict: false,    // treat warnings as errors?
}})
```

Validation checks:
- Meta completeness (name, description, whenToUse, phases)
- Phase consistency (every `phase()` call matches a `meta.phases` title)
- Agent schemas (label, phase, schema structure, required fields)
- Args documentation (header comment covers all `args.X` references)
- Return envelope (non-empty `return {}` outside loop)
- Parallelism integrity (`parallel()` receives arrow function thunks)
- Semantic coherence (data flow between phases makes sense)

### Workflow authoring gotchas

For practical pitfalls and runtime constraints (including the `new Date()` / `Date.now()` restriction), see:

- [`skills/claude-workflow-gotchas/SKILL.md`](skills/claude-workflow-gotchas/SKILL.md)

## Templates

Templates are parameterized skeletons for common workflow topologies.

| Template | Topology | Suitable for |
|----------|----------|--------------|
| `iterative-loop.js` | Plan→Execute→Evaluate→Learn→Repeat | AccelOpt, ARGUS, most optimization loops |
| `search-based.js` | Generate→Sample→Evaluate→Prune | Autotuning, evolutionary search, KSearch |
| `single-pass.js` | Analyze→Transform→Verify | KEET, compiler passes, rule-based optimization |
| `tree-exploration.js` | Branch→Score→Expand→Backtrack | Tree-of-thought, beam search |

Each template uses `{{TOKEN}}` placeholders that the generate-workflow fills from the manifest.

## Manifest Schema

The manifest (`_meta/manifests/schema.yaml`) is the structured description of a kernel optimization method. It has these sections:

| Section | Purpose |
|---------|---------|
| `source` | Paper title, URL, venue, repo, license |
| `workflow` | name, description, whenToUse, output filename |
| `method` | category, core_insight, feedback_signal |
| `topology` | type (iterative/search/tree/pipeline), state variables, convergence |
| `phases` | Ordered list of phases with agents, schemas, parallelism |
| `plan_angles` | Orthogonal optimization dimensions for planners |
| `args` | Required and optional workflow parameters |
| `correctness` | How the evaluator checks functional correctness |
| `return_fields` | What the workflow returns |
| `template` | Which template base + optional feature blocks |

See `schema.yaml` for the full annotated reference.

## How We Actually Used It

For the workflows in this repository, the meta-workflow served as a **design framework** rather than a fully automated pipeline:

| Workflow | Generation method |
|---------|------------------|
| AccelOpt | Hand-crafted first, manifest back-filled as reference |
| KSearch | Hand-crafted following the same patterns |
| KEET | Manually written following the meta-workflow's Research→Model→Generate→Validate mental model |
| ARGUS | Manually written following the same process |

The `generate-workflow.js` meta-workflow can be run end-to-end for future additions, but complex methods (like ARGUS with its ICRL loop) often benefit from human judgment in the Modeling phase.

## Adding a New Method

Recommended process:

1. **Read the paper** — identify the core loop, feedback signal, and key innovation
2. **Classify topology** — is it iterative, search-based, tree, or pipeline?
3. **Draft a manifest** — fill the YAML following existing examples (start from `accelopt.yaml` for iterative, `keet.yaml` for pipeline)
4. **Write the workflow** — use the appropriate template as a starting point
5. **Validate** — run `validate-workflow` to catch structural issues
6. **Add README** — bilingual (EN/ZH), with prerequisites, usage, architecture diagram

Or run `generate-workflow` with the paper URL and iterate on the output.

---

# Meta-Workflow：论文 → 工作流生成流水线

本目录包含 **meta-workflow** 基础设施 — 自动将研究论文或代码仓库转化为经过验证的 Claude Code Workflow 的工具。

## 什么是 Meta-Workflow？

Meta-workflow 本身是一个 Claude Code Workflow，它：

1. **研究**论文/仓库以理解其优化方法论
2. **建模**循环拓扑、阶段、agent、参数和规划角度
3. **组装**符合 schema 的结构化 YAML manifest
4. **生成**从模板 + manifest 产出的完整 `.js` 工作流
5. **验证**生成的工作流的正确性

## 使用方法

### 从论文生成新工作流

```javascript
Workflow({name: 'generate-workflow', args: {
  paper_url: 'https://arxiv.org/abs/XXXX.XXXXX',
  repo_url: 'https://github.com/org/repo',    // 可选
  output_dir: '/path/to/Awesome-Kernel-Workflows',
  method_name: 'MethodName',                    // 可选
}})
```

### 验证已有工作流

```javascript
Workflow({name: 'validate-workflow', args: {
  workflow_path: '/path/to/my-workflow.js',
  strict: false,
}})
```

## 模板

| 模板 | 拓扑 | 适用于 |
|------|------|--------|
| `iterative-loop.js` | 规划→执行→评估→学习→重复 | AccelOpt, ARGUS 等优化循环 |
| `search-based.js` | 生成→采样→评估→剪枝 | 自动调优、进化搜索、KSearch |
| `single-pass.js` | 分析→转换→验证 | KEET、编译器 pass、规则优化 |
| `tree-exploration.js` | 分支→评分→扩展→回溯 | 思维树、束搜索 |

## Manifest Schema

Manifest 是对内核优化方法的结构化描述。主要部分：

| 部分 | 用途 |
|------|------|
| `source` | 论文标题、URL、会议、仓库、许可证 |
| `workflow` | 名称、描述、使用场景、输出文件名 |
| `method` | 类别、核心创新、反馈信号 |
| `topology` | 类型、状态变量、收敛条件 |
| `phases` | 有序阶段列表及其 agent |
| `plan_angles` | 正交优化维度 |
| `args` | 必需和可选参数 |
| `correctness` | 正确性检查方式 |
| `return_fields` | 工作流返回值 |
| `template` | 使用的模板基础 + 可选特性块 |

## 添加新方法的流程

1. **阅读论文** — 识别核心循环、反馈信号、关键创新
2. **分类拓扑** — 迭代、搜索、树、还是流水线？
3. **起草 manifest** — 参照已有示例填写 YAML
4. **编写工作流** — 以对应模板为起点
5. **验证** — 运行 `validate-workflow`
6. **添加 README** — 中英文双语，含前置依赖、使用方法、架构图

或者直接运行 `generate-workflow` 输入论文 URL，然后迭代改进输出。
