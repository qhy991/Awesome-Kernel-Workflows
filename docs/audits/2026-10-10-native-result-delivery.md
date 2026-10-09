# Native result delivery after a completed task command

M4 run-20261009T162553Z-42e789a7f14e, stage3, failed in __nativeTaskEvaluate because the evaluation subagent ended without StructuredOutput after the native nudge. Its exact task slot already contained bound candidate_failure evidence and a terminal broker receipt. The old catch path threw HOLD before the existing read-only reconciliation path could inspect that slot.

The successor handles only that exact missing-StructuredOutput message, with qualified native file handoff, a declared read-only command, and no additional error code or cause. It runs the existing readback at most once. It does not rerun the evaluator or GPU command. Missing/unknown/corrupt evidence still fails normal validation. Transport, refusal and model-unknown errors retain HOLD. A failed repair never creates a third agent.

Prompts now explicitly request the StructuredOutput tool call. The full diagnostics remain in error records and canonical files; the readback prompt carries the reason and file path rather than reinlining the entire failed reply.

The canonical task-result scaffold is refreshed in all five methods by the repository codemod. The codemod refreshes an existing block in place, including native-only workflows without the old sol-execbench insertion anchor.

Validation:92 focused Node tests passed, including the missing-output regression, refusal/unknown counterexamples, bounded repair, diagnostic preservation, and native-only codemod idempotence. Fidelity contracts passed. This is CPU qualification; native deployment and live recovery are recorded separately. The old M4 sources and failed calls were not modified.
