# DVR Page Size Fix Plan

## Goal

Support `page_size > 1` for DVR chunk-boundary verify without changing generic
allocator or attention-backend ownership rules. The first target is the common
configuration:

- `FLA_CHUNK_SIZE = 64`
- `speculative_num_draft_tokens = 16`
- `page_size = 16`
- `speculative_eagle_topk = 1`

The correctness target remains strict equality against the full-prefill scoring
oracle.

## What EAGLE Teaches

EAGLE does not treat speculative KV as a flat token-wise allocation when
`page_size > 1`.

- Draft allocation is page-aware. For `topk == 1`, it uses
  `alloc_paged_token_slots_extend()` from the current sequence length to
  `seq_len + speculative_num_steps`, so draft tokens continue inside the
  current partial page before allocating new pages.
- For `topk > 1`, EAGLE duplicates the last partial prefix page per branch and
  moves KV with `source_cache_loc` / `target_cache_loc`.
- After verify, EAGLE frees only page-safe KV. For `topk == 1`, it uses
  `align_evict_mask_to_page_size()` so a partial page that still contains an
  accepted token is not freed.

DVR is chain-only (`topk == 1`), so the useful part is the page-aware allocation
and page-safe free behavior. DVR cannot directly reuse EAGLE's variable verify
window because DVR target verify physically runs a fixed
`FLA_CHUNK_SIZE + draft` window.

## Current Failure

With `page_size=16`, the DVR fixed verify window expects 80 rows, but a request
with a short verified tail produced only 73 `out_cache_loc` rows:

```text
verified_tail 9 + draft 16 + padding allocation 48 = 73
expected fixed verify window = 80
```

The missing 7 rows come from using token-wise `alloc_token_slots()` with a paged
allocator for a non-page-aligned padding length. The allocator returns full
pages, while DVR assumes the requested logical length.

## Implementation Phases

### Phase 1: Page-Aware Self Draft Allocation

For `page_size > 1`, use the same topk=1 layout idea as EAGLE:

- prefix length: `batch.seq_lens`
- end length: `batch.seq_lens + num_draft_tokens`
- last loc: `get_last_loc(...)`
- allocation: `alloc_paged_token_slots_extend(...)`

Keep `assign_draft_cache_locs()` as the writer into `req_to_token`; its topk=1
path is compatible once allocation covers the correct logical range.

### Phase 2: Fixed Verify Padding Ownership

Construct every DVR fixed verify row explicitly:

```text
verified_tail_locs + draft_locs + padding_locs = verify_window_size
```

For padding:

- Reuse already-allocated page-tail slots that follow the last real token when
  they are part of the same physical page. These slots belong to the draft
  allocation and must not be freed by DVR fixed-window cleanup.
- Allocate additional padding-only full pages only when the fixed window extends
  beyond those existing page-tail slots.
- Track only padding-only allocations in `padding_locs` so cleanup releases
  memory it owns.

### Phase 3: Launch Validation

Allow `page_size > 1` only when:

- `FLA_CHUNK_SIZE % page_size == 0`
- `speculative_num_draft_tokens % page_size == 0`
- `speculative_eagle_topk == 1`

Otherwise fall back to `page_size=1` with a warning.

GDN/Mamba state settings remain separate from attention KV page size:

- `mamba_scheduler_strategy = extra_buffer`
- `mamba_track_interval = FLA_CHUNK_SIZE`
- `mamba_ssm_dtype = float32`

Mamba prefix/radix state is still chunk-aligned to `FLA_CHUNK_SIZE`; this is not
the same concept as attention KV page size.

### Phase 4: Validation

Run static checks first:

- `py_compile` for touched files
- `git diff --check`

Then run strict KL grids with DeepGEMM disabled:

- Qwen3 attention-only, graph, `page_size=16`, bs=2, lengths `17,65,129`
- Qwen3 attention-only, no graph, `page_size=16`, bs=2, lengths `17,65,129`
- Qwen3.5 GDN, graph, `page_size=16`, bs=2, lengths `65,129,257`
- Qwen3.5 GDN, no graph, `page_size=16`, bs=2, lengths `65,129,257`

If graph fails but no-graph passes, isolate CUDA graph metadata/buffer sizing.
If both fail, inspect physical `out_cache_loc` lengths and page ownership first.

## 2026-05-16 Results

Implemented the aligned page-size path for `page_size=16`.

- Self-draft KV allocation now uses the EAGLE topk=1 paged extend allocator.
- DVR fixed-window padding now derives same-page tail slots from the last real
  physical KV location instead of reading unwritten `req_to_token` tail entries.
- Padding cleanup owns only newly allocated padding-only pages; page-tail slots
  remain owned by the draft allocation.
- Launch validation allows `page_size > 1` only when it divides both
  `FLA_CHUNK_SIZE` and `speculative_num_draft_tokens`; unsupported cases fall
  back to `1` in server-args postprocessing.

Validation, all with DeepGEMM disabled:

- Qwen3 attention-only no-graph, `page_size=16`, bs=2, lengths `17,65,129`:
  strict KL=0.
- Qwen3 attention-only graph, `page_size=16`, bs=2, lengths `17,65,129`:
  strict KL=0.
- Qwen3.5 GDN graph, `page_size=16`, bs=2, lengths `65,129,257`: strict KL=0.
- Qwen3.5 GDN no-graph, `page_size=16`, bs=2, lengths `65,129,257`: strict
  KL=0.

## 2026-05-16 GDN Track-State Guard

For GDN DVR, the Mamba/GDN prefill tracking interval must be exactly
`FLA_CHUNK_SIZE`, not merely a multiple of it. Larger multiples are chunk
aligned, but the current extra_buffer prefill path stores only one tracked
checkpoint per prefill/extend pass. If the interval is larger than
`FLA_CHUNK_SIZE`, a first prompt prefill can miss the latest chunk boundary that
DVR needs as the next target-verify starting state.

The launch path now rewrites GDN DVR `mamba_track_interval` to `FLA_CHUNK_SIZE`
with an explicit warning, and the DVR worker asserts the same invariant at
runtime. It also asserts fp32 temporal/intermediate SSM state storage, because
bf16/fp16 chunk-boundary checkpoints round the chunkwise scan state and can
diverge from full prefill across chunks.

Validation:

- Qwen3.5 GDN no-graph launched with `--mamba-track-interval 256`; server-args
  postprocessing reset it to `64`, and bs=2 lengths `65,129,257` passed strict
  KL=0.
- Qwen3 attention-only no-graph, bs=2 lengths `65,129`, still passed strict
  KL=0.
