# Workflow-owned candidate paths

## Observed failure

M4 run `run-20261010T014420Z-c74e574688dc`, native Workflow
`wf_ecbc4873-8ab`, failed at the last first-iteration AccelOpt evaluation.
The producer was told to write `iter_0_plan_2_sample_1.py` beneath its assigned
workspace. Its Bash result and retained file confirm that it wrote that file.
Its structured `variant_path` then omitted `native-delivery-8254c276-20261010`
and changed the cohort date from `20261007` to `20261010`.

The consumer preferred that echoed path to the path the workflow had assigned.
The evaluation agent correctly refused the missing file. Consequently no scored
result slot was created. One read-only recovery ran and also failed because no
frozen candidate/result existed for that slot. The earlier readback fix cannot
recover an evaluation that never ran.

## Correction and scope

CUDAAgent, AccelOpt, CUDALLM-FSR, KernelFoundry and KSearch now evaluate the exact
candidate path assigned before their implementation activation. The producer's
returned path remains in its raw reply; it cannot redirect an assigned slot.
KSearch also uses its assigned path for subsequent diagnostic context.

No filesystem discovery, basename matching, source reconstruction or substitute
candidate is used. The unchanged task runner must find, freeze and evaluate the
real assigned file. Missing source, missing evidence, model uncertainty, policy
refusal and failed correctness still stop or reject according to their existing
contracts. Initial seed-set validation is unchanged.

Search topology, counts, scoring, budgets, final acceptance and existing result
file reconciliation are unchanged. Frozen runs retain their code and failures.
This patch must be qualified in a successor before a new device/model run.

## Verification

- Five full-workflow replay regressions inject a wrong returned path after the
  workflow has assigned the producer's output slot. All five fail before this
  correction and pass afterward.
- 126 focused Node tests pass, including measurement/source rejection and bounded
  readback behavior. One prior execution-boundary assertion still expected the
  old `test_result_json={}` wording; it now checks the existing explicit
  StructuredOutput tool instruction and empty-object JSON payload.
- All 35 workflows parse with the Host's actual async wrapper.
- The fidelity contract checker passes.

These CPU checks do not certify a completed native/GPU successor. The original
canary remains failed; its earlier passing measurements are not a final accepted
workflow result.
