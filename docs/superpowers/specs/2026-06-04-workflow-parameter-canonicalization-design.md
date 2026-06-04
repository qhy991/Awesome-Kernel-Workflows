# Workflow Parameter Canonicalization Design

## Context

The repository contains many kernel optimization workflows derived from different papers and systems. Their argument names have drifted: GPU target, benchmark command, iteration budget, source input, and task-definition fields often use different names for the same concept.

The migration should be intentionally breaking: old aliases are removed from workflow docs and code so agents learn one canonical API. This must not erase method-specific fidelity contracts. Parameters that own a paper mechanism or evidence artifact remain method-specific.

A new requirement is that workflows should handle both existing-kernel optimization and problem-definition-only runs. If a caller provides only a problem definition, the workflow must first synthesize an initial kernel, verify it, then enter the method's optimization loop.

## Goals

- Establish one canonical argument vocabulary for common workflow inputs.
- Support `problem_definition` / `problem_path` as first-class inputs, not only `kernel_path`.
- Add a consistent generate-then-optimize preamble for applicable workflows.
- Preserve fidelity-critical parameters such as profiler commands, invariant artifact paths, archive update paths, and method-specific search controls.
- Update workflow JavaScript, READMEs, manifests, templates, and generator tooling consistently.

## Non-Goals

- Do not turn every workflow into a generic optimizer. KEET-style explanation-only workflows can document unsupported generation mode.
- Do not rename evidence artifact parameters just to make them shorter.
- Do not replace method-specific loops such as MCTS rollouts, RL rollout count, archive insertion, or invariant checking with generic `iterations`.
- Do not keep backward-compatible aliases in code or docs; this migration intentionally standardizes the public API.

## Canonical Arguments

### Common Inputs

- `kernel_path`: Existing kernel/source file to optimize. If present, the workflow starts from this file.
- `problem_definition`: Inline natural-language or code-like problem definition. Used when no initial kernel is available.
- `problem_path`: File containing the problem definition, such as a KernelBench task, PyTorch reference, operator spec, or structured task markdown.
- `op_description`: Short human description of the operator or workload. This remains optional context, not the authoritative problem definition.
- `language`: Target implementation language such as `cuda`, `triton`, `cute-dsl`, `tilelang`, `cpp`, or `python`.
- `target_gpu`: Target accelerator or architecture string.

At least one of `kernel_path`, `problem_definition`, or `problem_path` must be provided for optimization/generation workflows. If both `kernel_path` and a problem input are provided, the workflow optimizes the existing kernel and uses the problem input as correctness/context metadata. If no `kernel_path` is provided, the workflow enters generation mode.

### Commands And Evidence

- `compile_command`: Compilation/import command. It may use `{kernel_path}` and `{result_path}` placeholders.
- `test_command`: Correctness command. It should write JSON at `{result_path}` with at least `compiled` and `correct`.
- `benchmark_command`: Performance command. It should write JSON at `{result_path}` with `compiled`, `correct`, and one of `latency_ms`, `throughput`, `score`, or `speedup`.
- `ncu_command`: Full Nsight Compute profiling command when the workflow needs a custom NCU invocation.
- `ncu_binary`: Path/name for the NCU binary when the workflow builds NCU commands itself.
- `nsys_binary`: Path/name for the Nsight Systems binary when required.

The old names `bench_command`, `eval_command`, `evaluation_command`, `verify_command`, and `validation_command` should be migrated to `benchmark_command` or `test_command` according to evidence semantics. Combined evaluator commands that compile, test, and benchmark should use `benchmark_command` and document the JSON contract.

### Budgets And Search Controls

- `iterations`: Canonical outer-loop budget for generic iterative/search cycles.
- `seed_candidates`: Canonical count for initial generated candidates in generation mode.
- `breadth`: Canonical count for parallel planning branches in iterative optimizers.
- `samples_per_plan`: Canonical count for implementation samples per plan.
- `population_size`: Remains canonical for population/evolution methods.
- `exp_dir`: Experiment artifact directory.
- `rtol` / `atol`: Correctness tolerances.

Method-specific controls stay method-specific when the name carries the source method's mechanism, for example `rl_iterations`, `rollouts_per_select`, `max_llm_calls_per_attempt`, `descriptor_result_path`, `archive_update_result_path`, `feature_vector_result_path`, and `invariant_result_path`.

## Rename Map

| Current names | Canonical name | Notes |
| --- | --- | --- |
| `gpu_target`, `gpu_type`, `gpu_arch`, `hardware_target`, `target_hardware` | `target_gpu` | Use for GPU/accelerator target in prompts, profiling, and docs. |
| `kernel_language`, `target_language` | `language` | Document as implementation language. |
| `bench_command`, performance-only `eval_command`, performance-only `evaluation_command` | `benchmark_command` | Must state expected JSON result contract. |
| correctness-only `verify_command`, `validation_command` | `test_command` | Keep `compile_command` and `lint_command` separate. |
| `max_iterations`, generic `rounds`, generic `budget`, generic `steps` | `iterations` | Preserve method-specific loop names when required for fidelity. |
| `initial_kernel_path`, `baseline_code_path`, `source_code_path`, `reference_kernel_path`, source-kernel `model_path` | `kernel_path` | Only when the field is an existing kernel/source to optimize. |
| `problem_description`, `task_spec`, inline `operator_spec` | `problem_definition` | Inline authoritative problem input. |
| `task_spec_path`, `task_path`, PyTorch-task `reference_path`, problem-file `problem_path` variants | `problem_path` | File-backed authoritative problem input. |
| `seed_count`, `max_seeds` | `seed_candidates` | Only for initial generation candidate count. |

## Input Resolution Flow

Every applicable workflow should begin with a shared setup policy:

1. Resolve `exp_dir` and create subdirectories for generated kernels, evaluator JSON, profiles, and reports.
2. Resolve input mode:
   - `optimize_existing` when `kernel_path` is provided.
   - `generate_then_optimize` when no `kernel_path` is provided and `problem_definition` or `problem_path` is provided.
3. Load problem context from `problem_definition` or `problem_path` when available.
4. If `generate_then_optimize`, synthesize `seed_candidates` initial kernels in `language`.
5. Materialize each generated candidate under `${exp_dir}/generated/`.
6. Run `compile_command` / `test_command` / `benchmark_command` as available.
7. Select the best compiled and correct generated candidate as the initial `kernel_path` for the method loop.
8. If no generated candidate is verified correct and a real evaluator was provided, fail hard or return `success: false` before optimization.
9. If no real evaluator was provided, mark the run as `correctness_status: "unverified"` and preserve the workflow's fidelity limitation in the report.

## Generation Phase Contract

Generation mode is not just an extra prompt. It must produce artifacts the optimizer can consume:

- `generated_kernel_path`: Path to the selected initial kernel.
- `initial_candidates`: Array of generated candidate records.
- `initial_generation_result`: Object with compile/correctness/benchmark evidence.
- `input_mode`: `optimize_existing` or `generate_then_optimize`.
- `problem_path` / `problem_definition`: Reflected in return fields when supplied.

The initial generated kernel must not be treated as a valid starting point unless it compiles and passes `test_command` or a correctness-covering `benchmark_command`. If the workflow cannot verify correctness, it may continue only as a documented low-fidelity or exploratory run.

## Workflow Applicability

Upgrade first:

- Kernel optimizers that already accept `kernel_path` and run iterative optimization: AKO4X, AccelOpt, Generalist, KernelBand, KernelBlaster, ARGUS, KDA, CUDAAgent, Astra, ReGraphT.
- Workflows already close to problem-definition generation: KernelAgent, CUDALLM, KernelFoundry, KernelFoundryDx, KernelSkill, KSearch, AdaExplore.

Document or defer:

- KEET is primarily a profile explanation workflow. It can keep requiring `kernel_path` / NCU report and document that generation mode is unsupported.
- STARK and TritorX need careful migration because their current input contracts are tied to reference kernels or operator lists.

## Manifest And Template Changes

Add a manifest-level `inputs` section:

```yaml
inputs:
  supports_existing_kernel: true
  supports_problem_definition: true
  required_one_of:
    - kernel_path
    - problem_definition
    - problem_path
  generation:
    enabled: true
    seed_candidates_arg: seed_candidates
    output_arg: generated_kernel_path
    requires_correctness_evidence: true
```

Update template setup blocks to emit the input-resolution policy before method-specific setup. Existing `args.required` should no longer require `kernel_path` for workflows that support generation mode; instead, the generated workflow should validate the `required_one_of` set.

## Testing And Verification

- Add or update a static checker that flags old common names in workflow args and README argument tables.
- Run `node scripts/check-fidelity-contracts.js` after workflow edits.
- Run `_tools/validate-workflow.js` or `_meta/tools/validate-workflow.js` against generated workflows when manifests change.
- For generated-mode workflows, add smoke examples showing:
  - Existing kernel: `kernel_path`.
  - Inline problem: `problem_definition`.
  - File-backed problem: `problem_path`.

## Rollout Plan

1. Update schema, templates, and generator tooling with canonical arguments and the `inputs` section.
2. Migrate workflows in small groups by input model:
   - Existing-kernel optimizers.
   - Problem-definition generators.
   - Hybrid/search workflows.
   - Explanation or specialized workflows.
3. Update English and Chinese READMEs after each group.
4. Run static checks and fidelity checks after each group.
5. Update README catalog guidance and `Agent.md` so future workflows use the same API.

## Risks

- Strong migration breaks old invocations. The benefit is a cleaner agent-facing API.
- Some old names are ambiguous. For example, `reference_path` may mean a PyTorch problem file in one workflow and correctness reference in another. Each migration must inspect usage before renaming.
- Generation mode can reduce fidelity if correctness evidence is missing. The workflow must explicitly label unverified generated kernels instead of treating them as measured successes.
- Over-standardizing method budgets can erase paper mechanics. Keep method-specific controls when they represent a real selection or search rule.
