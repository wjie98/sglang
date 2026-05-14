# DVR EAGLE Postprocess Experiment

Branch: `dvr-eagle-postprocess-exp`

Base: `dvr-clean-v3` at `f4bc4b1ac`

Goal:

- Keep DVR verify/postprocess semantics close to EAGLE.
- Fix self-decode-specific GDN state side effects without changing `EagleVerifyOutput` semantics.

Changes in this experiment:

- Split DVR verify into EAGLE-like helper steps:
  - select accepted logits by `accepted_indices`
  - commit DVR/GDN state
  - add spec-v1 output logprobs
  - prepare next draft from `verify_output.draft_input`
- Use EAGLE-style accepted token metadata for GDN state commit.
- Preserve EAGLE's `accept_length + 1` contract; do not hide or rewrite bonus tokens.
- Add live Mamba/GDN state backup before self decode.
- During target verify restore:
  - temporal state from the chunk boundary snapshot
  - conv state from the live pre-draft snapshot

Validated:

- Qwen3-0.6B pure attention, deterministic DVR, `temperature=0.7`:
  - `max_new_tokens`: 1, 8, 16, 31, 48, 65, 96, 128
  - full-prefill scoring oracle: strict `max_abs_diff = 0.0`
  - acceptance mostly 1.0, long cases still near 1.0

- Qwen3.5-0.8B GDN, deterministic DVR, `temperature=0.7`:
  - before live conv restore: divergence starts at output index 16, max diff around 9 on 32-token generation.
  - after live conv restore:
    - 16 tokens: `max_abs_diff ~= 4.5e-6`
    - 32 tokens: `max_abs_diff ~= 0.0126`, acceptance 1.0
    - 48 tokens: `max_abs_diff ~= 0.0126`, acceptance 1.0
    - 80 tokens: `max_abs_diff ~= 0.0744`, acceptance 1.0

Rejected experiment:

- Shifting/skipping the anchor row in the GDN q/k/v/g/beta rolling cache reduced one symptom but broke longer generation.
- Reason: fixing only GDN qkvg rows is not enough, because the anchor row hidden state also passes through the rest of the transformer layer stack. Reusing the previous anchor correctly likely needs a broader hidden-state path or a second scoring/verify path, not a local qkvg-only remap.

Current interpretation:

- The EAGLE-compatible verify/postprocess contract is correct for DVR and should be preserved.
- The major GDN bug was restoring conv state to the chunk boundary before target verify. SSM temporal state should be boundary-aligned, but conv state for current-token preprocessing must come from the live pre-draft tail.
- Remaining nonzero GDN KL is probably an anchor/hidden-state reuse issue at verify window boundaries, not a simple accepted-token postprocess bug.
