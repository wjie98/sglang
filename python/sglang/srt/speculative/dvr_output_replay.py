from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dvr_logprob_repair import (
    defer_and_score_dvr_final_logprob_repairs,
)
from sglang.srt.speculative.dvr_scheduler_utils import (
    DVRFinalLogprobRepair,
    DVRReplayPrefixTracker,
)


def compact_output_token_rows(
    output_tokens: torch.Tensor | None,
    output_lens: torch.Tensor,
) -> Optional[list[list[int]]]:
    """Return per-request output token rows without padded tail tokens.

    Spec workers often keep accepted tokens in rectangular CUDA tensors.  Final
    logprob repair needs the client-visible emitted stream, not padded verify
    rows, so callers should pass only these compact CPU rows across DVR helpers.
    """

    if output_tokens is None:
        return None

    output_lens_cpu = output_lens.detach().cpu().tolist()
    output_tokens_cpu = output_tokens.detach().cpu().tolist()
    return [
        [int(token_id) for token_id in token_row[: int(output_len)]]
        for token_row, output_len in zip(
            output_tokens_cpu, output_lens_cpu, strict=True
        )
    ]


class DVRClientOutputReplayTracker:
    """Tracks the client-visible output stream for DVR final logprob repair.

    DVR-EAGLE has two token streams during overlap verify: the verifier stream
    used to replay target GDN state, and the output stream that the scheduler
    will publish to the client.  Keep this distinction inside DVR so the worker
    does not need to expose request/output materialization timing details.
    """

    def __init__(self) -> None:
        self.replay_prefix = DVRReplayPrefixTracker()

    def seed_from_target_extend(
        self,
        *,
        batch: ScheduleBatch,
        next_token_ids: torch.Tensor,
    ) -> None:
        """Record target EXTEND's first client-visible output token."""

        if batch.reqs is None or next_token_ids is None:
            return

        next_token_ids_cpu = next_token_ids.detach().cpu().tolist()
        self.replay_prefix.append_batch_output_tokens(
            batch,
            [[token_id] for token_id in next_token_ids_cpu],
            initialize_from_req_output=True,
        )

    def advance_from_compact_rows(
        self,
        *,
        batch: ScheduleBatch,
        compact_output_token_ids_per_req: list[list[int]],
        error_prefix: str,
    ) -> None:
        """Append this verify step's compact client-visible output rows."""

        if batch.forward_mode.is_idle() or batch.reqs is None:
            return

        self.replay_prefix.prune_to_batch(batch)

        seq_lens_cpu = (
            batch.seq_lens_cpu.tolist()
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().cpu().tolist()
        )

        for req, seq_len, compact_output_token_ids in zip(
            batch.reqs,
            seq_lens_cpu,
            compact_output_token_ids_per_req,
            strict=True,
        ):
            prompt_len = len(req.origin_input_ids)
            stream = self.replay_prefix.stream_for_req(
                req,
                initialize_from_req_output=True,
            )
            # In spec-v2 overlap, the model-side seq_len can lag the
            # client-visible output stream by one result. Do not truncate a
            # prefix already learned from Req.output_ids/tracker state.
            prefix_output_len = max(0, int(seq_len) - prompt_len, len(stream))
            prefix_ids = self.replay_prefix.request_output_prefix_token_ids(
                req,
                prompt_len + prefix_output_len,
                error_prefix=error_prefix,
            )
            # Overlap can compute final repair before the prefill/previous
            # decode result is materialized into Req.output_ids. Seed from the
            # best known prefix, then append this verify's compact output rows.
            stream[:] = prefix_ids[prompt_len:]
            stream.extend(compact_output_token_ids)

    def defer_and_score_final_logprob_repairs(
        self,
        *,
        batch: ScheduleBatch,
        target_worker: Any,
        linear_state_ctx: Any,
        base_seq_lens_cpu: list[int],
        accept_lens_cpu: list[int],
        compact_output_token_ids_per_req: list[list[int]],
        error_prefix: str,
    ) -> Optional[list[Optional[DVRFinalLogprobRepair]]]:
        """Advance output replay and run unified final logprob repair."""

        self.advance_from_compact_rows(
            batch=batch,
            compact_output_token_ids_per_req=compact_output_token_ids_per_req,
            error_prefix=f"{error_prefix} output replay prefix",
        )
        return defer_and_score_dvr_final_logprob_repairs(
            batch=batch,
            target_worker=target_worker,
            replay_prefix=self.replay_prefix,
            linear_state_ctx=linear_state_ctx,
            base_seq_lens_cpu=base_seq_lens_cpu,
            accept_lens_cpu=accept_lens_cpu,
            compact_output_token_ids_per_req=compact_output_token_ids_per_req,
            error_prefix=error_prefix,
        )
