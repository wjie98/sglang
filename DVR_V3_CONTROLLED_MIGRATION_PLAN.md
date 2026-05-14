# DVR V3 Controlled Migration Plan

This file tracks the cleanup migration from the experimental
`dvr-eagle-postprocess-exp` branch back onto the cleaner v3 DVR branch.

The goal is not to copy the experimental branch wholesale. The experimental
branch is treated as evidence: keep only the parts that are known to help, and
rebuild the main path on v3 with small, reviewable changes.

## Baseline

- Clean target branch: `dvr-clean-v3`
- Experimental reference branch: `dvr-eagle-postprocess-exp`
- Upstream merge base target: `sglang-miles`
- Attention-only test model: `/home/hwj/Qwen3-0.6B`
- GDN test model: `/home/hwj/Qwen3.5-0.8B`
- Runtime environment: `conda activate dvr_dev`
- DeepGEMM must be disabled on RTX 5090:
  - `SGLANG_ENABLE_JIT_DEEPGEMM=0`
  - `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`

## Design Rules

1. Keep the DVR verify window fixed at `CHUNK_SIZE + NUM_DRAFT`.
   The default shape is `64 + 16 = 80`.
2. Describe the fixed window as:
   - `verified_token`
   - `draft_token`
   - `padding_token`
3. Reuse EAGLE-style postprocess semantics for chain acceptance, anchor token
   handling, bonus token handling, and request result updates.
4. Do not change DVR into normal EAGLE verify. DVR still needs the fixed
   chunk-aligned verify window for GDN and future cuda graph capture.
5. Keep `custom_mask=None` for DVR chain target verify so attention-only models
   use normal causal attention in both cuda graph and non-cuda graph modes.
6. Treat conv state as a first-class state, not as incidental metadata. Its
   lifecycle should mirror the GDN/FLA temporal state lifecycle.
7. Prefer DVR-local hooks and comments. Avoid modifying shared SGLang interfaces
   unless there is no clean DVR-local place to put the logic.

## Phase 0: Inventory And Branch Setup

Tasks:

- Confirm `dvr-clean-v3` is clean.
- Confirm `dvr-eagle-postprocess-exp` is only used as a read-only reference.
- Create a new controlled migration branch from `dvr-clean-v3`.
- Record this plan before code edits.

Expected result:

- A clean migration branch where each step can be reviewed or reverted
  independently.

## Phase 1: Attention-Only DVR Cleanup

Tasks:

- Keep self-decode draft provider only; remove stale self-extend draft choices.
- Align DVR anchor/bonus/result lifecycle with EAGLE's chain verify
  postprocess model.
- Keep the DVR fixed verify window as `verified_token + draft_token +
  padding_token`.
- Keep target verify `custom_mask=None` for chain DVR.
- Remove experimental debug prints and offset hacks.

Expected result:

- Qwen3 attention-only strict full-prefill oracle KL remains exactly 0.
- Long generation across multiple 64-token boundaries does not crash.

Status:

- Started on branch `dvr-v3-controlled-migration`.
- Migrated the EAGLE-style verify postprocess split from
  `dvr-eagle-postprocess-exp`:
  - select accepted logits/hidden states by `accepted_indices`;
  - commit DVR/GDN state before adding returned logprobs;
  - prepare the next draft from `verify_output.draft_input`;
  - preserve EAGLE's `accept_length + 1` contract instead of rewriting bonus
    token semantics.
- Added a live Mamba/GDN backup slot used by the GDN path. The attention-only
  path is unaffected, but this preserves the useful exp-branch direction for
  later conv-state cleanup.
- Static checks passed:
  - `conda run -n dvr_dev python -m py_compile python/sglang/srt/speculative/dvr_worker.py`
  - `git diff --check --cached && git diff --check`
- Qwen3-0.6B strict full-prefill oracle passed with cuda graph disabled:
  - `max_new_tokens`: `1, 8, 16, 31, 64, 80, 127, 128, 191`
  - every case reported `maxdiff=0.0`
- Qwen3-0.6B strict full-prefill oracle passed with cuda graph enabled:
  - `max_new_tokens`: `1, 8, 16, 31, 64, 80, 127, 128, 191`
  - every case reported `maxdiff=0.0`

## Phase 2: Fixed Verify Window Clarification

Tasks:

- Make the 80-token verify window construction explicit and easy to inspect.
- Keep the beginning of the verify window aligned to `FLA_CHUNK_SIZE`.
- Restore EAGLE-style metadata after target verify so scheduler integration is
  minimally different from normal speculative decoding.
- Add only DVR-local comments that explain why the logical window is larger than
  the draft-token count.

Expected result:

- The code clearly separates verified tokens, draft tokens, and padding tokens.
- The scheduler still sees normal accepted output tokens and logprobs.

## Phase 3: GDN Input Capture

Tasks:

- Preserve the existing v3 qkvg_beta capture path if it is already correct.
- Recheck cuda graph compatibility: capture-time tensors must have stable
  addresses.
- Ensure saved qkvg_beta belongs to the fixed 80-token DVR verify window.

Expected result:

- Qwen3 remains KL=0.
- Qwen3.5 can run target verify and save qkvg_beta without crashes in cuda graph
  and non-cuda graph modes.

Current inspection:

- Do not wholesale-port the dirty exp branch GDN diff.
- Keep the useful idea from exp: DVR GDN verify needs a fixed logical window and
  split state semantics.
- Do not migrate exp debug prints.
- Do not migrate padding-zeroing as a correctness fix; it did not remove the
  observed GDN drift in previous experiments.
- Avoid changing `cuda_graph_runner.py` first. Prefer making `EagleVerifyInput`
  and the DVR worker carry enough metadata for GDN while keeping shared graph
  code close to upstream.
- The next minimal code boundary should be:
  - add DVR-local fixed-window metadata only where needed;
  - keep attention-only path unchanged;
  - make GDN qkvg_beta capture explicitly understand
    `verified_token + draft_token + padding_token`;
  - make conv state restoration semantics explicit before touching state commit
    kernels.

## Phase 4: Conv State Lifecycle

Tasks:

- Add/clean a DVR-owned chunk-boundary conv state beside the chunk-boundary GDN
  temporal state.
- Before self-decode draft, use the current live state.
- Before target verify, restore the chunk-boundary temporal state and conv
  state.
- After target verify, commit live tail temporal state and live tail conv state
  according to accepted tokens.
- When accepted tokens cross a chunk boundary, commit the new boundary temporal
  state and boundary conv state, then slide the rolling window.

Expected result:

- Conv state no longer depends on guessed offsets from the experimental branch.
- The implementation follows the same state ownership model as EAGLE where
  possible, while preserving the DVR 80-token verify shape.

## Phase 5: GDN State Commit

Tasks:

- Use chunkwise scan semantics for deterministic chunk-boundary state.
- Use saved qkvg_beta for any recurrent tail-state reconstruction that is still
  needed between boundaries.
- Keep checkpoint/radix-cache reusable state at chunk boundaries deterministic.
- Do not require local FLA source modifications unless experiments prove they
  are unavoidable.

Expected result:

- Qwen3.5 strict oracle KL is 0 for most tested settings.
- Any remaining nonzero KL cases are documented with prompt length, generation
  length, batch size, cuda graph mode, and first mismatch token.

## Phase 6: Test Matrix

Attention-only Qwen3:

- bs=1, cuda graph off/on
- `max_new_tokens`: `1, 8, 16, 31, 64, 80, 127, 128, 191`
- strict KL target: exactly 0

GDN Qwen3.5:

- bs=1 and bs>1
- cuda graph off/on
- random and irregular `max_new_tokens`, including values beyond 64
- different prompt lengths, including non-64-aligned prompts
- temporary accepted-length hacks if needed to stress different batch boundary
  transitions

## Commit Policy

Create small commits after stable milestones:

1. `dvr: start controlled v3 migration`
2. `dvr: align chain postprocess with eagle`
3. `dvr: clarify fixed chunk verify window`
4. `dvr(gdn): clean qkvg beta capture`
5. `dvr(gdn): manage boundary conv state`
6. `dvr(gdn): commit deterministic chunk state`
7. `test(dvr): record strict kl coverage`

Each commit message should include:

- What changed.
- Why it differs from normal EAGLE.
- Which tests were run.
- Known limitations.
