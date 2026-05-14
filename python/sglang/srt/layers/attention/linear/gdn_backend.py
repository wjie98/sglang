from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )
else:
    FLA_CHUNK_SIZE = 64

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer if either decode or prefill selected it
        if decode_backend.is_flashinfer() or prefill_backend.is_flashinfer():
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def recurrent_state_from_qkvg_beta(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor,
        token_count: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if is_cpu():
            raise NotImplementedError("DVR GDN recurrent post-verify is GPU-only.")

        if token_count is None:
            _, final_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            return final_state.to(initial_state.dtype, copy=False)

        if isinstance(token_count, int):
            token_count = torch.full(
                (q.shape[0],), token_count, dtype=torch.long, device=q.device
            )
        token_count = token_count.to(device=q.device, dtype=torch.long)
        if torch.all(token_count == token_count[0]):
            end = int(token_count[0].item())
            if end == 0:
                return initial_state
            _, final_state = fused_recurrent_gated_delta_rule(
                q=q[:, :end],
                k=k[:, :end],
                v=v[:, :end],
                g=g[:, :end],
                beta=beta[:, :end],
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            return final_state.to(initial_state.dtype, copy=False)

        states = []
        for i, end in enumerate(token_count.tolist()):
            if end == 0:
                states.append(initial_state[i : i + 1])
                continue
            _, final_state = fused_recurrent_gated_delta_rule(
                q=q[i : i + 1, :end],
                k=k[i : i + 1, :end],
                v=v[i : i + 1, :end],
                g=g[i : i + 1, :end],
                beta=beta[i : i + 1, :end],
                initial_state=initial_state[i : i + 1],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            states.append(final_state)
        return torch.cat(states, dim=0).to(initial_state.dtype, copy=False)

    def chunkwise_boundary_state_from_qkvg_beta(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        boundary_token_count: int = FLA_CHUNK_SIZE,
    ) -> torch.Tensor:
        if is_cpu():
            raise NotImplementedError("DVR GDN chunkwise post-verify is GPU-only.")
        if boundary_token_count % FLA_CHUNK_SIZE != 0:
            raise ValueError(
                f"boundary_token_count must align to {FLA_CHUNK_SIZE}, "
                f"got {boundary_token_count}."
            )
        if boundary_token_count == 0:
            return state_pool[state_indices]
        if boundary_token_count > q.shape[1]:
            raise ValueError(
                f"boundary_token_count={boundary_token_count} exceeds saved "
                f"qkvg beta length {q.shape[1]}."
            )

        _, _, h = chunk_gated_delta_rule(
            q=q[:, :boundary_token_count],
            k=k[:, :boundary_token_count],
            v=v[:, :boundary_token_count],
            g=g[:, :boundary_token_count],
            beta=beta[:, :boundary_token_count],
            initial_state=state_pool,
            initial_state_indices=state_indices,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        return h[:, -1].to(state_pool.dtype, copy=False)

    def chunkwise_state_from_qkvg_beta(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        token_count: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        if is_cpu():
            raise NotImplementedError("DVR GDN chunkwise post-verify is GPU-only.")
        if isinstance(token_count, int):
            token_count = torch.full(
                (q.shape[0],), token_count, dtype=torch.long, device=q.device
            )
        token_count = token_count.to(device=q.device, dtype=torch.long)

        states = []
        for i, end in enumerate(token_count.tolist()):
            if end == 0:
                states.append(state_pool[state_indices[i : i + 1]])
                continue
            _, _, h = chunk_gated_delta_rule(
                q=q[i : i + 1, :end],
                k=k[i : i + 1, :end],
                v=v[i : i + 1, :end],
                g=g[i : i + 1, :end],
                beta=beta[i : i + 1, :end],
                initial_state=state_pool,
                initial_state_indices=state_indices[i : i + 1],
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            states.append(h[:, -1])
        return torch.cat(states, dim=0).to(state_pool.dtype, copy=False)


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

    def _cache_dvr_extend_verified_qkvg_beta(
        self,
        forward_batch: ForwardBatch,
        mamba_cache_params: MambaPool.SpeculativeState,
        cache_indices: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ):
        if getattr(mamba_cache_params, "dvr_q_state_cache", None) is None:
            return
        if (
            forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
            or self.forward_metadata.query_start_loc is None
        ):
            return

        query = query.reshape(query.shape[1], query.shape[2], query.shape[3])
        key = key.reshape(key.shape[1], key.shape[2], key.shape[3])
        value = value.reshape(value.shape[1], value.shape[2], value.shape[3])
        g = g.reshape(-1, g.shape[-1])
        beta = beta.reshape(-1, beta.shape[-1])
        query_start_loc = self.forward_metadata.query_start_loc

        for req_i, (prefix_len, extend_len) in enumerate(
            zip(
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                strict=True,
            )
        ):
            seq_len = prefix_len + extend_len
            boundary = (seq_len // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
            num_verified_tokens = seq_len - boundary
            dst = cache_indices[req_i].to(torch.long)
            mamba_cache_params.dvr_qkvg_beta_pos[dst] = num_verified_tokens
            if num_verified_tokens == 0:
                continue

            write_start = max(prefix_len, boundary)
            write_end = seq_len
            if write_start >= write_end:
                continue

            src_start = int(query_start_loc[req_i].item()) + (write_start - prefix_len)
            src_end = src_start + (write_end - write_start)
            cols = torch.arange(
                write_start - boundary,
                write_end - boundary,
                dtype=torch.long,
                device=cache_indices.device,
            )
            # Seed the DVR rolling window with the prompt/extend tail after the
            # latest chunk boundary. Later target verify appends draft_token
            # rows to this same qkvg_beta window.
            mamba_cache_params.dvr_q_state_cache[dst, cols] = query[src_start:src_end]
            mamba_cache_params.dvr_k_state_cache[dst, cols] = key[src_start:src_end]
            mamba_cache_params.dvr_v_state_cache[dst, cols] = value[src_start:src_end]
            mamba_cache_params.dvr_g_state_cache[dst, cols] = g[src_start:src_end]
            mamba_cache_params.dvr_beta_state_cache[dst, cols] = beta[src_start:src_end]

    def update_dvr_state_after_verify(
        self,
        *,
        live_indices: torch.Tensor,
        boundary_indices: torch.Tensor,
        accepted_tokens: torch.Tensor,
        accepted_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Commit DVR GDN state after target verify.

        The worker owns request-level bookkeeping, but the tensor lifecycle is
        GDN-specific: q/k/v/g/beta windows, live temporal state, chunk-boundary
        state, and conv windows must move together.
        """

        mamba_cache = self.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        pos_before = mamba_cache.dvr_qkvg_beta_pos[0, live_indices].to(torch.long)
        pos_after = pos_before + accepted_tokens
        crossing = pos_after >= FLA_CHUNK_SIZE

        # Rebuild live tail state from the deterministic chunk-boundary state
        # plus the saved q/k/v/g/beta window. This repairs the state dirtied by
        # draft/verify and lets the next self-decode continue from the last
        # accepted token.
        for layer_idx in range(mamba_cache.temporal.shape[0]):
            tail_state = self.kernel_dispatcher.chunkwise_state_from_qkvg_beta(
                mamba_cache.dvr_q_state_cache[layer_idx, live_indices],
                mamba_cache.dvr_k_state_cache[layer_idx, live_indices],
                mamba_cache.dvr_v_state_cache[layer_idx, live_indices],
                mamba_cache.dvr_g_state_cache[layer_idx, live_indices],
                mamba_cache.dvr_beta_state_cache[layer_idx, live_indices],
                state_pool=mamba_cache.temporal[layer_idx],
                state_indices=boundary_indices,
                token_count=pos_after,
            )
            mamba_cache.temporal[layer_idx, live_indices] = tail_state

        fused_mamba_state_scatter_with_mask(
            mamba_cache.conv[0],
            mamba_cache.intermediate_conv_window[0],
            live_indices,
            pos_before + accepted_steps,
        )

        if crossing.any():
            commit_step = torch.where(
                crossing,
                FLA_CHUNK_SIZE - pos_before - 1,
                torch.full_like(pos_before, -1),
            )
            for layer_idx in range(mamba_cache.temporal.shape[0]):
                boundary_state = (
                    self.kernel_dispatcher.chunkwise_boundary_state_from_qkvg_beta(
                        mamba_cache.dvr_q_state_cache[layer_idx, live_indices],
                        mamba_cache.dvr_k_state_cache[layer_idx, live_indices],
                        mamba_cache.dvr_v_state_cache[layer_idx, live_indices],
                        mamba_cache.dvr_g_state_cache[layer_idx, live_indices],
                        mamba_cache.dvr_beta_state_cache[layer_idx, live_indices],
                        state_pool=mamba_cache.temporal[layer_idx],
                        state_indices=boundary_indices,
                        boundary_token_count=FLA_CHUNK_SIZE,
                    )
                )
                mamba_cache.temporal[layer_idx, boundary_indices] = boundary_state

            fused_mamba_state_scatter_with_mask(
                mamba_cache.conv[0],
                mamba_cache.intermediate_conv_window[0],
                boundary_indices,
                commit_step,
            )

            new_pos = pos_after - FLA_CHUNK_SIZE
            for req_i, slot in enumerate(live_indices.tolist()):
                if not bool(crossing[req_i].item()):
                    continue
                remain = int(new_pos[req_i].item())
                if remain > 0:
                    for cache in (
                        mamba_cache.dvr_q_state_cache,
                        mamba_cache.dvr_k_state_cache,
                        mamba_cache.dvr_v_state_cache,
                        mamba_cache.dvr_g_state_cache,
                        mamba_cache.dvr_beta_state_cache,
                    ):
                        cache[:, slot, :remain] = cache[
                            :, slot, FLA_CHUNK_SIZE : FLA_CHUNK_SIZE + remain
                        ].clone()
            pos_after = torch.where(crossing, new_pos, pos_after)

        mamba_cache.dvr_qkvg_beta_pos[:, live_indices] = pos_after.to(torch.int32)
        return crossing

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            if getattr(mamba_cache_params, "dvr_q_state_cache", None) is not None:
                # DVR verifies a fixed linear window
                # (verified_token + draft_token + padding_token), so use the
                # same causal conv path as ordinary extend instead of the tree
                # update helper used by generic EAGLE target verify.
                mixed_qkv = causal_conv1d_fn(
                    mixed_qkv.transpose(0, 1),
                    layer.conv_weights,
                    layer.bias,
                    activation=layer.activation,
                    conv_states=conv_states,
                    has_initial_state=has_initial_states,
                    cache_indices=cache_indices[:batch_size],
                    query_start_loc=query_start_loc,
                    seq_lens_cpu=[draft_token_num] * batch_size,
                ).transpose(0, 1)[:seq_len]
            else:
                mixed_qkv_reshaped = mixed_qkv.view(
                    batch_size, draft_token_num, -1
                ).transpose(1, 2)
                mixed_qkv_processed = causal_conv1d_update(
                    mixed_qkv_reshaped,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    layer.activation,
                    conv_state_indices=cache_indices[:batch_size],
                    intermediate_conv_window=intermediate_conv_window_cache,
                    intermediate_state_indices=intermediate_state_indices[:batch_size],
                    retrieve_next_token=retrieve_next_token,
                    retrieve_next_sibling=retrieve_next_sibling,
                    retrieve_parent_token=retrieve_parent_token,
                )
                mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            if getattr(mamba_cache_params, "dvr_q_state_cache", None) is not None:
                dvr_indices = cache_indices[:batch_size].to(torch.long)
                pos = mamba_cache_params.dvr_qkvg_beta_pos[dvr_indices].to(torch.long)
                cols = pos.unsqueeze(1) + torch.arange(
                    draft_token_num, dtype=torch.long, device=pos.device
                ).unsqueeze(0)
                rows = dvr_indices.unsqueeze(1).expand(-1, draft_token_num)
                mamba_cache_params.dvr_q_state_cache[rows, cols] = query.reshape(
                    batch_size, draft_token_num, layer.num_q_heads, layer.head_q_dim
                )
                mamba_cache_params.dvr_k_state_cache[rows, cols] = key.reshape(
                    batch_size, draft_token_num, layer.num_k_heads, layer.head_k_dim
                )
                mamba_cache_params.dvr_v_state_cache[rows, cols] = value.reshape(
                    batch_size, draft_token_num, layer.num_v_heads, layer.head_v_dim
                )
                mamba_cache_params.dvr_g_state_cache[rows, cols] = g.reshape(
                    batch_size, draft_token_num, layer.num_v_heads
                )
                mamba_cache_params.dvr_beta_state_cache[rows, cols] = beta.reshape(
                    batch_size, draft_token_num, layer.num_v_heads
                )
                out, _, _ = chunk_gated_delta_rule(
                    q=mamba_cache_params.dvr_q_state_cache[dvr_indices],
                    k=mamba_cache_params.dvr_k_state_cache[dvr_indices],
                    v=mamba_cache_params.dvr_v_state_cache[dvr_indices],
                    g=mamba_cache_params.dvr_g_state_cache[dvr_indices],
                    beta=mamba_cache_params.dvr_beta_state_cache[dvr_indices],
                    initial_state=ssm_states,
                    initial_state_indices=dvr_indices,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )
                gather_index = cols[:, :, None, None].expand(
                    batch_size, draft_token_num, layer.num_v_heads, layer.head_v_dim
                )
                core_attn_out = out.gather(1, gather_index).reshape(
                    1, batch_size * draft_token_num, layer.num_v_heads, layer.head_v_dim
                )
            else:
                core_attn_out = self.kernel_dispatcher.target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    intermediate_states_buffer=intermediate_state_cache,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=forward_batch.spec_info.draft_token_num,
                    retrieve_parent_token=retrieve_parent_token,
                )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            self._cache_dvr_extend_verified_qkvg_beta(
                forward_batch,
                mamba_cache_params,
                cache_indices,
                query,
                key,
                value,
                g,
                beta,
            )
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out
