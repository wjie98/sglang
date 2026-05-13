from __future__ import annotations

import torch


def chain_speculative_sampling(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool,
) -> None:
    """Classic chain speculative sampling for top-1 self draft.

    The first candidate is the anchor token that has already been sampled by the
    target model. Each following candidate is accepted with min(1, p / q). The
    last accepted slot is overwritten with the bonus token sampled from the
    residual distribution, matching the layout used by EAGLE's verify output.
    """

    del retrive_next_token, retrive_next_sibling
    del threshold_single, threshold_acc, deterministic

    batch_size, num_slots = candidates.shape
    for bid in range(batch_size):
        root_index = retrive_index[bid, 0]
        accept_index[bid, 0] = root_index
        last_index = root_index
        prob_row = 0
        num_accepted = 0
        all_accepted = True

        for step in range(1, num_slots):
            draft_token = candidates[bid, step]
            p = target_probs[bid, prob_row, draft_token]
            q = draft_probs[bid, prob_row, draft_token]
            if uniform_samples[bid, step - 1] * q < p:
                predicts[last_index] = draft_token
                num_accepted += 1
                prob_row = step
                last_index = retrive_index[bid, step]
                accept_index[bid, num_accepted] = last_index
            else:
                all_accepted = False
                break

        accept_token_num[bid] = num_accepted
        final_probs = target_probs[bid, prob_row]
        if not all_accepted:
            final_probs = torch.clamp(final_probs - draft_probs[bid, prob_row], min=0)

        norm = final_probs.sum()
        if norm <= 0:
            final_token = torch.argmax(target_probs[bid, prob_row])
        else:
            cdf = torch.cumsum(final_probs, dim=0)
            final_token = torch.searchsorted(cdf, uniform_samples_for_final_sampling[bid] * norm)
            final_token = torch.clamp(final_token, max=final_probs.numel() - 1)

        predicts[last_index] = final_token.to(predicts.dtype)
