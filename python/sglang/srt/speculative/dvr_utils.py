from contextlib import contextmanager


@contextmanager
def dvr_causal_verify_cuda_graph_metadata(
    model_runner, attn_backend, forward_mode, spec_info, fallback_custom_mask=None
):
    """Keep DVR cuda graph verify on causal attention without backend edits.

    Some attention backends still read `spec_info.custom_mask.shape` while
    building cuda-graph metadata. For DVR topk=1 the real mask is causal, so
    temporarily provide the graph buffer only to satisfy metadata shape code,
    then clear the captured metadata before graph capture/replay uses it.
    """

    old_custom_mask = getattr(spec_info, "custom_mask", None)
    should_clear = (
        model_runner.spec_algorithm.is_decode_verify_rollback()
        and forward_mode.is_target_verify()
        and spec_info is not None
    )
    if should_clear and old_custom_mask is None and fallback_custom_mask is not None:
        # DVR target verify is a topk=1 chain, so the real attention mask is
        # causal and `custom_mask` should stay None. Some cuda-graph metadata
        # builders still read `spec_info.custom_mask.shape` before producing
        # their metadata, so provide the fixed graph buffer only for that shape
        # bookkeeping. The metadata is cleared below before graph capture/replay.
        spec_info.custom_mask = fallback_custom_mask
    try:
        yield
    finally:
        if should_clear:
            spec_info.custom_mask = old_custom_mask
            backends = [attn_backend]
            full_attn_backend = getattr(attn_backend, "full_attn_backend", None)
            if full_attn_backend is not None:
                backends.append(full_attn_backend)
            for backend in backends:
                metadata = getattr(backend, "forward_metadata", None)
                if metadata is None:
                    continue
                # Restore DVR's causal semantics after metadata construction:
                # do not let the temporary tree-mask buffer select a custom-mask
                # attention path in the captured graph or replay metadata. For
                # hybrid GDN models the real attention metadata lives inside the
                # nested full-attention backend, so clear both levels.
                if hasattr(metadata, "custom_mask"):
                    metadata.custom_mask = None
                if hasattr(metadata, "mask_indptr"):
                    metadata.mask_indptr = None


@contextmanager
def dvr_runtime_verify_window(attn_backend, num_verify_tokens: int):
    """Temporarily make full-attention target verify use DVR's fixed window.

    GDN DVR verifies a `CHUNK_SIZE + num_draft_tokens` linear window, while the
    generic speculative attention backends cache their CLI draft length on the
    backend object. Keep that generic code untouched and locally override only
    the full-attention backend for this DVR verify forward.
    """

    backend = getattr(attn_backend, "full_attn_backend", attn_backend)
    old_num_draft_tokens = getattr(backend, "num_draft_tokens", None)
    old_speculative_num_draft_tokens = getattr(
        backend, "speculative_num_draft_tokens", None
    )
    try:
        if old_num_draft_tokens is not None:
            backend.num_draft_tokens = num_verify_tokens
        if old_speculative_num_draft_tokens is not None:
            backend.speculative_num_draft_tokens = num_verify_tokens
        yield
    finally:
        if old_num_draft_tokens is not None:
            backend.num_draft_tokens = old_num_draft_tokens
        if old_speculative_num_draft_tokens is not None:
            backend.speculative_num_draft_tokens = old_speculative_num_draft_tokens
