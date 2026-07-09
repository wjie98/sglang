from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, List, Optional

import torch

from sglang.srt.speculative.eagle_info import EagleDraftExtendInput, EagleVerifyInput


@dataclass
class DVRMambaCheckpoint:
    track_idx: Optional[int] = None
    seqlen: Optional[int] = None

    @property
    def valid(self) -> bool:
        return self.track_idx is not None and self.seqlen is not None


@dataclass
class DVRFinalLogprobRepair:
    """Exact final output logprobs produced by a DVR worker.

    Spec-v2 overlap materializes accepted tokens into ``Req.output_ids`` after
    the worker returns.  DVR workers may still compute a full-prefix logprob
    oracle on the forward stream; carry the final rows here and apply them only
    after scheduler result processing owns the final output token list.
    """

    output_ids: list[int]
    output_logprobs: list[float]


@dataclass
class DVRSpecResultAux:
    """Worker-to-scheduler DVR metadata carried by GenerationBatchResult."""

    pending_mamba_checkpoints: Optional[list[Optional[DVRMambaCheckpoint]]] = None
    final_logprob_repairs: Optional[
        list[Optional[DVRFinalLogprobRepair]]
    ] = None


def build_dvr_spec_result_aux(
    *,
    track_indices: Optional[list[Optional[int]]],
    seqlens: Optional[list[Optional[int]]],
    final_logprob_repairs: Optional[list[Optional[DVRFinalLogprobRepair]]] = None,
) -> Optional[DVRSpecResultAux]:
    if track_indices is None and seqlens is None and final_logprob_repairs is None:
        return None

    if track_indices is None:
        track_indices = [None] * (len(seqlens) if seqlens is not None else 0)
    if seqlens is None:
        seqlens = [None] * len(track_indices)

    checkpoints = [
        (
            DVRMambaCheckpoint(track_idx=track_idx, seqlen=seqlen)
            if track_idx is not None and seqlen is not None
            else None
        )
        for track_idx, seqlen in zip(track_indices, seqlens, strict=True)
    ]
    return DVRSpecResultAux(
        pending_mamba_checkpoints=checkpoints or None,
        final_logprob_repairs=final_logprob_repairs,
    )


class DVRPendingOutputPrefix:
    """Worker-owned output prefix journal for DVR overlap paths.

    Spec-v2 can start the next replay before result processing appends accepted
    tokens into ``Req.output_ids``.  DVR therefore records the real
    client-visible accepted tokens produced by each verify step.  Replay
    prefixes are built only from ``origin_input_ids`` + materialized
    ``Req.output_ids`` + this pending output journal; draft-token streams and
    broad fallback reconstruction are deliberately excluded.
    """

    def __init__(self) -> None:
        self._tokens_by_req: weakref.WeakKeyDictionary[Any, list[int]] = (
            weakref.WeakKeyDictionary()
        )

    def clear(self) -> None:
        self._tokens_by_req.clear()

    def prune_to_batch(self, batch: Any) -> None:
        if batch.reqs is None:
            self._tokens_by_req.clear()
            return

        active_reqs = set(batch.reqs)
        for req in list(self._tokens_by_req):
            if req not in active_reqs:
                self._tokens_by_req.pop(req, None)

    def _stream_for_req(
        self, req: Any, *, error_prefix: Optional[str] = None
    ) -> list[int]:
        stream = self._tokens_by_req.setdefault(req, [])
        output_ids = list(req.output_ids)
        if output_ids:
            common_len = min(len(stream), len(output_ids))
            if stream[:common_len] != output_ids[:common_len]:
                prefix = error_prefix or "DVR output prefix"
                raise RuntimeError(
                    f"{prefix} diverged from materialized output ids: "
                    f"rid={req.rid}, tracked_tail={stream[-8:]}, "
                    f"req_tail={output_ids[-8:]}, tracked_len={len(stream)}, "
                    f"req_output_len={len(output_ids)}."
                )
            if len(output_ids) > len(stream):
                stream.extend(int(token_id) for token_id in output_ids[len(stream) :])
        return stream

    def observed_output_len(self, req: Any) -> int:
        """Return the best known client-visible output length for final repair."""

        return len(self._stream_for_req(req))

    def _prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        error_prefix: Optional[str] = None,
    ) -> Optional[list[int]]:
        """Return an explicitly owned DVR replay prefix."""

        origin_input_ids = list(req.origin_input_ids)
        output_len = seq_len - len(origin_input_ids)
        if output_len <= 0:
            return origin_input_ids[:seq_len]

        stream = self._stream_for_req(req, error_prefix=error_prefix)
        if len(stream) >= output_len:
            return origin_input_ids + stream[:output_len]

        if error_prefix is not None:
            raise RuntimeError(
                f"{error_prefix} replay prefix is not yet owned by DVR: "
                f"rid={req.rid}, origin_tokens={len(req.origin_input_ids)}, "
                f"req_output_tokens={len(req.output_ids)}, "
                f"tracked_output_tokens={len(stream)}, seq_len={seq_len}."
            )
        return None

    def request_output_prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
        *,
        error_prefix: str,
    ) -> list[int]:
        """Return a client-visible output prefix for DVR replay."""

        return self._prefix_token_ids(
            req,
            seq_len,
            error_prefix=error_prefix,
        )

    def try_request_output_prefix_token_ids(
        self,
        req: Any,
        seq_len: int,
    ) -> Optional[list[int]]:
        """Best-effort client-visible output prefix reconstruction."""

        return self._prefix_token_ids(
            req,
            seq_len,
        )

    def _append_tokens(
        self,
        req: Any,
        token_ids,
        *,
        error_prefix: str,
    ) -> None:
        stream = self._stream_for_req(req, error_prefix=error_prefix)
        stream.extend(int(token_id) for token_id in token_ids)

    def append_batch_output_tokens(
        self,
        batch: Any,
        tokens_per_req,
        *,
        base_seq_lens_cpu: Optional[list[int]] = None,
        error_prefix: str = "DVR output prefix",
    ) -> None:
        """Advance the client-visible output stream for all active requests."""

        self.prune_to_batch(batch)
        if base_seq_lens_cpu is None and getattr(batch, "seq_lens", None) is not None:
            base_seq_lens_cpu = (
                batch.seq_lens_cpu.tolist()
                if getattr(batch, "seq_lens_cpu", None) is not None
                else batch.seq_lens.detach().cpu().tolist()
            )

        if base_seq_lens_cpu is None:
            base_seq_lens_cpu = [None] * len(batch.reqs)

        for req, base_seq_len, token_ids in zip(
            batch.reqs, base_seq_lens_cpu, tokens_per_req, strict=True
        ):
            stream = self._stream_for_req(req, error_prefix=error_prefix)
            if base_seq_len is not None:
                required_len = max(0, int(base_seq_len) - len(req.origin_input_ids))
                if len(stream) < required_len:
                    raise RuntimeError(
                        f"{error_prefix} is behind the batch logical length: "
                        f"rid={req.rid}, tracked={len(stream)}, "
                        f"required={required_len}."
                    )
            stream.extend(int(token_id) for token_id in token_ids)


@dataclass
class DVRAcceptedOutputRows:
    """Accepted tokens in the same per-request rows scheduler will emit."""

    base_seq_lens_cpu: Optional[list[int]]
    accept_lens_cpu: list[int]
    token_ids_per_req: list[list[int]]

    @classmethod
    def from_flat_tokens(
        cls,
        *,
        batch: Any,
        output_tokens: torch.Tensor,
        accept_lens,
        tokens_per_req: Optional[int] = None,
        base_seq_lens_cpu: Optional[list[int]] = None,
    ) -> "DVRAcceptedOutputRows":
        if base_seq_lens_cpu is None and getattr(batch, "seq_lens", None) is not None:
            base_seq_lens_cpu = (
                batch.seq_lens_cpu.tolist()
                if getattr(batch, "seq_lens_cpu", None) is not None
                else batch.seq_lens.detach().cpu().tolist()
            )
        if base_seq_lens_cpu is not None:
            base_seq_lens_cpu = [int(seq_len) for seq_len in base_seq_lens_cpu]

        if torch.is_tensor(accept_lens):
            accept_lens_cpu = [int(x) for x in accept_lens.detach().cpu().tolist()]
        else:
            accept_lens_cpu = [int(x) for x in accept_lens]
        token_ids = output_tokens.detach().cpu().reshape(-1).tolist()

        token_ids_per_req = []
        if tokens_per_req is None:
            offset = 0
            for accept_len in accept_lens_cpu:
                end = offset + accept_len
                token_ids_per_req.append([int(x) for x in token_ids[offset:end]])
                offset = end
        else:
            for req_i, accept_len in enumerate(accept_lens_cpu):
                start = req_i * tokens_per_req
                end = start + accept_len
                token_ids_per_req.append([int(x) for x in token_ids[start:end]])

        return cls(
            base_seq_lens_cpu=base_seq_lens_cpu,
            accept_lens_cpu=accept_lens_cpu,
            token_ids_per_req=token_ids_per_req,
        )

    def append_to_prefix(
        self,
        prefix: DVRPendingOutputPrefix,
        batch: Any,
        *,
        error_prefix: str = "DVR output prefix",
    ) -> None:
        prefix.append_batch_output_tokens(
            batch,
            self.token_ids_per_req,
            base_seq_lens_cpu=self.base_seq_lens_cpu,
            error_prefix=error_prefix,
        )


@contextmanager
def _dvr_causal_verify_cuda_graph_metadata(
    model_runner, attn_backend, forward_mode, spec_info, fallback_custom_mask=None
):
    """Build DVR target-verify graph metadata without selecting custom masks."""

    old_custom_mask = getattr(spec_info, "custom_mask", None)
    should_clear_custom_mask = (
        (
            model_runner.spec_algorithm.is_dvr_self_draft()
            or getattr(model_runner, "enable_dvr_target_verify_cuda_graph", False)
        )
        and forward_mode.is_target_verify()
        and spec_info is not None
    )
    if (
        should_clear_custom_mask
        and old_custom_mask is None
        and fallback_custom_mask is not None
    ):
        # DVR target verify is a topk=1 chain, so the real attention mask is
        # causal. Some graph metadata builders still read custom_mask.shape;
        # provide the graph buffer only for that shape bookkeeping.
        spec_info.custom_mask = fallback_custom_mask
    if should_clear_custom_mask:
        spec_info.custom_mask = None
    try:
        yield
    finally:
        if should_clear_custom_mask:
            spec_info.custom_mask = old_custom_mask
            backends = [attn_backend]
            full_attn_backend = getattr(attn_backend, "full_attn_backend", None)
            if full_attn_backend is not None:
                backends.append(full_attn_backend)
            for backend in backends:
                metadata = getattr(backend, "forward_metadata", None)
                if metadata is None:
                    continue
                if hasattr(metadata, "custom_mask"):
                    metadata.custom_mask = None
                if hasattr(metadata, "mask_indptr"):
                    metadata.mask_indptr = None


class DVRTargetVerifyMixin:
    """DVR target-verify CUDA graph fixups shared by DVR draft variants."""

    @classmethod
    def from_eagle_verify_input(cls, verify_input: EagleVerifyInput):
        """Preserve EAGLE draft metadata while swapping in DVR verify hooks."""

        return cls(
            **{
                field.name: getattr(verify_input, field.name)
                for field in fields(EagleVerifyInput)
            }
        )

    def prepare_cuda_graph_replay_buffers(self, graph_runner, raw_num_token: int):
        if not graph_runner.capture_forward_mode.is_target_verify():
            return
        # The generic num_token_non_padded slot is enabled only for EP. GDN DVR
        # verify also needs the raw token count to mask padded graph rows.
        graph_runner.buffers.num_token_non_padded.fill_(raw_num_token)

    def cuda_graph_metadata_context(
        self,
        *,
        model_runner,
        attn_backend,
        forward_mode,
        fallback_custom_mask=None,
    ):
        return _dvr_causal_verify_cuda_graph_metadata(
            model_runner,
            attn_backend,
            forward_mode,
            self,
            fallback_custom_mask,
        )


class DVREagleVerifyInput(DVRTargetVerifyMixin, EagleVerifyInput):
    """DVR target verify with a standard EAGLE target-only sampler."""

    def _sampling_uniforms(self, candidates: torch.Tensor, batch):
        # EAGLE target-only verification has rejection/final-sampling coins
        # outside the normal sampler. Honor request-level sampling_seed here so
        # DVR-EAGLE sync/overlap comparisons are reproducible under sampling.
        from sglang.srt.speculative.dvr_worker import dvr_chain_uniform_samples

        return dvr_chain_uniform_samples(candidates, batch)


@dataclass
class DVRSelfDraftVerifyInput(DVRTargetVerifyMixin, EagleVerifyInput):
    """DVR verify input with classic chain speculative sampling.

    Shared EAGLE verification is target-only. DVR self-draft records the draft
    sampling distribution and must use the chain accept/reject kernel to keep
    the generated distribution and acceptance rate aligned with the reference
    branch.
    """

    draft_probs: Optional[torch.Tensor] = None

    def _sampling_fn_and_draft_probs(self, target_probs: torch.Tensor, batch):
        if self.draft_probs is None:
            return super()._sampling_fn_and_draft_probs(target_probs, batch)

        from sglang.srt.speculative.dvr_worker import chain_speculative_sampling

        return chain_speculative_sampling, self.draft_probs

    def _sampling_uniforms(self, candidates: torch.Tensor, batch):
        from sglang.srt.speculative.dvr_worker import dvr_chain_uniform_samples

        return dvr_chain_uniform_samples(candidates, batch)


@dataclass
class DVRVerifyOutput:
    """Compatibility view for DVR spec-v1 post-verify bookkeeping."""

    accept_tokens: torch.Tensor
    accept_indices: torch.Tensor
    num_correct_drafts_per_req_cpu: List[int]
    draft_extend_input: EagleDraftExtendInput
