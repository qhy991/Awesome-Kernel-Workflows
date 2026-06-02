# Meta-Workflow 文档

[English](README.md) · **简体中文**

`_meta/` 目录用于集中管理“从论文/仓库生成 Claude Code workflow”的元基础设施：模板、manifest、生成器、校验器与规范 skill。

## 目录说明

- `tools/`：生成与校验工作流
  - `generate-workflow.js`
  - `validate-workflow.js`
- `templates/`：不同拓扑模板（iterative / search / single-pass / tree）
- `manifests/`：方法清单与 schema
- `skills/`：工作流编写规范与坑点
  - `claude-workflow-gotchas/SKILL.md`

## 目标

- 把“论文 → 可执行 workflow”的流程标准化
- 降低新增方法时的手工成本
- 让 workflow 生成可校验、可复现、可维护
