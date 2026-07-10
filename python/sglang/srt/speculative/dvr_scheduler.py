from __future__ import annotations

from typing import Any


def maybe_filter_running_batch_with_dvr_state(
    *,
    batch: Any,
    future_map: Any,
    enable_overlap: bool,
) -> bool:
    """Filter a DVR running batch with spec-v2 logical-finish state if needed."""

    spec_algorithm = batch.spec_algorithm
    if not enable_overlap:
        return False
    if spec_algorithm.is_dvr_self_draft() and not batch.enable_overlap:
        return False

    future_map.resolve_seq_lens_cpu(batch)
    keep_indices = []
    for i, req in enumerate(batch.reqs):
        if req.finished():
            continue

        max_new_tokens = req.sampling_params.max_new_tokens
        dvr_finished = False
        if max_new_tokens is not None:
            max_new_tokens = int(max_new_tokens)
            if batch.seq_lens_cpu is not None:
                seq_len = int(batch.seq_lens_cpu[i].item())
            elif batch.seq_lens is not None:
                seq_len = int(batch.seq_lens[i].item())
            else:
                seq_len = None
            if max_new_tokens > 0 and seq_len is not None:
                # Decode seq_lens includes KV-visible generated tokens; the
                # newest sampled bonus token is materialized into Req.output_ids
                # one result-processing step later, hence the final visible
                # token corresponds to max_new_tokens - 1.
                dvr_finished = (
                    seq_len - len(req.origin_input_ids) >= max_new_tokens - 1
                )
        if not dvr_finished:
            keep_indices.append(i)
    batch.filter_batch(keep_indices=keep_indices)
    return True
