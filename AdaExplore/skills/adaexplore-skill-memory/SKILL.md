---
name: adaexplore-skill-memory
description: Local workflow entry for AdaExplore failure-driven skill memory. Use when `adaexplore-kernel-optimization` reads or updates `skill_memory_path` using evaluated failure logs.
---

# AdaExplore Skill Memory Entry

This is a method-memory contract, not an external agent skill dependency.

The workflow reads `skill_memory_path` during setup and may update it only when `memory_update` is enabled. Memory lines use:

```text
You cannot ... || score
```

Rules must be grounded in evaluated compile/runtime/correctness failures. Speculative LLM self-assessment is not valid memory evidence.
