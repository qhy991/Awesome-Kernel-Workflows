// runtime-safe-point.js — CANONICAL cooperative-stop/checkpoint helper.
//
// Workflow scripts execute in a restricted sandbox without fs/process imports.
// File IO and wall-clock inspection therefore stay in one mechanical agent turn.
// Each search workflow calls this helper only at a natural iteration boundary:
// after evaluation and in-memory best-state updates, never mid-build or mid-eval.
//
// The helper has two responsibilities:
//   1. atomically materialize a changed best kernel and checkpoint;
//   2. observe the supervisor termination file/deadline and return normally.
//
// Call sites own the workflow-specific checkpoint payload and must include the
// common schema_version/workflow/progress/compiled/correct/metric/path fields.

async function __workflowRuntimeSafePoint(ctx) {
  const checkpointPath = ctx.checkpointPath || `${ctx.expDir}/checkpoint.json`
  const materialize = ctx.materializeBest && ctx.bestKernelPath && ctx.bestKernelSourcePath
    ? `Atomically copy the exact bytes from immutable candidate ${ctx.bestKernelSourcePath} to ${ctx.bestKernelPath}. ` +
      `Use a small Python program: read the source as bytes, require its SHA-256 to equal ` +
      `${ctx.bestKernelExpectedSha256 || '<missing-required-sha256>'}, write a temporary file in the destination directory, ` +
      `fsync it, then os.replace it. Recompute the destination SHA-256 and fail if it differs. ` +
      `Never regenerate, reformat, or reconstruct the source from a prompt.`
    : ctx.materializeBest && ctx.bestKernelPath && ctx.bestKernelCode
    ? `Atomically write this exact best source to ${ctx.bestKernelPath} using a temporary file in the same directory followed by rename:\n` +
      `\`\`\`${ctx.bestLanguage || ''}\n${ctx.bestKernelCode}\n\`\`\``
    : (ctx.bestKernelPath
      ? `Preserve the existing best source at ${ctx.bestKernelPath}; do not rewrite it.`
      : 'There is no verified best source yet; do not create a best-kernel file.')

  return agentRetry(() => agent(`Workflow runtime safe point.

1. ${materialize}
2. Check cooperative termination:
   - termination file: ${ctx.terminationFile || '<none>'}
   - deadline epoch: ${ctx.deadlineEpoch || 0}
   A non-empty termination file requests stop. If it contains JSON, use its
   "reason"; otherwise use "supervisor_request". A positive deadline requests
   stop when the current epoch from \`date +%s\` is at or beyond it.
3. Start from this exact checkpoint object:
${JSON.stringify(ctx.checkpoint)}
   If step 2 requests stop, set termination_requested=true and set
   termination_reason to the observed reason. Otherwise preserve the planned
   termination fields. Atomically write the resulting JSON to ${checkpointPath}
   using a temporary file in the same directory followed by os.replace/rename.
   Do not change metric.name or metric.value.
4. Return only the termination decision and checkpoint path.
`, {
    model: MODEL.mechanical,
    label: ctx.label,
    phase: ctx.phase,
    schema: {
      type: 'object',
      properties: {
        termination_requested: { type: 'boolean' },
        termination_reason: { type: 'string' },
        checkpoint_path: { type: 'string' },
      },
      required: ['termination_requested', 'checkpoint_path'],
    },
  }), { retries: 5 })
}
