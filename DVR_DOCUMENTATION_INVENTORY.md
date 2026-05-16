# DVR Documentation Inventory

This file is an index for root-level DVR notes in this worktree. It does not
delete, rename, or supersede any existing record.

## Upstream-Facing Candidates

- `DVR_KL_TESTING_GUIDE.md`: keep as the seed for a smaller upstream testing
  note. It contains the most reusable strict-KL oracle workflow.
- `DVR_UPDATE_REPORT.md`: current branch-level implementation report against
  `upstream/sglang-miles`.
- `DVR_V3_MERGE_CONTEXT.md`: keep as historical context while preparing the
  final PR description, then condense heavily.
- `DVR_V3_CONTROLLED_MIGRATION_PLAN.md`: useful as merge rationale, but should
  be reduced to design notes before upstreaming.

## Development-Only Records

- `DVR_CLEANUP_PHASE1_PLAN.md`
- `DVR_CLEANUP_PHASE2_PLAN.md`
- `DVR_CLEANUP_PHASE3_PLAN.md`
- `DVR_CLEANUP_PHASE4_PLAN.md`
- `DVR_CLEANUP_PHASE5_PLAN.md`
- `DVR_CLEANUP_PHASE6_PLAN.md`
- `DVR_EAGLE_POSTPROCESS_EXPERIMENT.md`
- `DVR_GDN_EXPERIMENT_NOTES.md`
- `DVR_GDN_MERGE_PLAN.md`
- `DVR_QWEN35_DIVERGENCE_REPORT.md`
- `DVR_QWEN35_DIVERGENCE_REPORT.py`
- `DVR_QWEN35_DIVERGENCE_REPORT_256.md`
- `DVR_QWEN35_DIVERGENCE_SUMMARY.md`
- `DVR_QWEN3_BATCH_KL_TEST.py`
- `DVR_V3_BASELINE_RESULTS.md`
- `DVR_V3_IMPLEMENTATION_PLAN.md`

These files are useful for local debugging and audit trails, but they should
not be copied directly into an upstream PR.

## Consolidation Notes

- Keep strict-KL testing commands and oracle behavior.
- Keep the rationale for fixed `FLA_CHUNK_SIZE + draft` physical verify
  windows.
- Keep the GDN state lifecycle explanation only after the implementation is
  stable enough to describe without experiment-specific branches.
- Drop raw divergence tables and temporary scripts from upstream-facing docs.
