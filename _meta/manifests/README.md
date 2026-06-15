# DEPRECATED — 迁移中草稿

本目录为 backend-driver 迁移的**中间草稿区**，**不再作为 source of truth**。

各 workflow 的 backend 声明权威来源为 `<Workflow>/manifest.yaml`。

## 迁移状态

| 文件 | 格式 | 状态 |
|------|------|------|
| `generalist.yaml` | 新（schema v1.1） | 已转新格式；需与 `Generalist/manifest.yaml` 同步 |
| `kernelfoundry.yaml` | 新（schema v1.1） | 已转新格式 |
| `kernelskill.yaml` | 新（schema v1.1） | 已转新格式；`intrinsic_to: ""` 待补 |
| 其余 | 旧（无 schema_version） | 待迁 |

如需查阅或修改 workflow 的 backend 能力，请直接编辑对应的 `<Workflow>/manifest.yaml`。