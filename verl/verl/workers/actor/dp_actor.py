# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.gradient import (
    compute_gradient_redundancy,
    compute_per_sample_grads_for_lm_head,
    AdvantageDecomposer,
    DecompositionConfig,
    collapse_token_level_advantage,
    expand_token_level_advantage,
)
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        # Initialize advantage decomposer if enabled
        self._init_advantage_decomposer()

    def _init_advantage_decomposer(self):
        """Initialize the advantage decomposer if enabled in config.

        The decomposer uses gradient geometry to balance consolidation and exploration
        channels in the advantage computation. This can help break rank collapse in GRPO.
        """
        adv_decomp_cfg = self.config.get("advantage_decomposition", {})

        if not adv_decomp_cfg.get("enabled", False):
            self.advantage_decomposer = None
            self._cached_decomposer_state = None
            return

        # Create decomposer configuration
        decomp_config = DecompositionConfig(
            enabled=True,
            rho=adv_decomp_cfg.get("rho", 0.5),
            k_min=adv_decomp_cfg.get("k_min", 1),
            k_max=adv_decomp_cfg.get("k_max", 64),
            use_relu_exploration=adv_decomp_cfg.get("use_relu_exploration", True),
            alpha_min=adv_decomp_cfg.get("alpha_min", 0.0),
            alpha_max=adv_decomp_cfg.get("alpha_max", 1.0),
            normalize_final=adv_decomp_cfg.get("normalize_final", False),
            chunk_size=adv_decomp_cfg.get("chunk_size", 64),
            use_fp32_gram=adv_decomp_cfg.get("use_fp32_gram", True),
            distributed_gram=adv_decomp_cfg.get("distributed_gram", False),
            log_metrics=adv_decomp_cfg.get("log_metrics", True),
            log_eigenspectrum=adv_decomp_cfg.get("log_eigenspectrum", False),
        )

        self.advantage_decomposer = AdvantageDecomposer(decomp_config)

        # Cache for delayed decomposition (uses previous step's geometry)
        # This allows efficient decomposition without extra forward pass
        self._cached_decomposer_state = {
            "projection_matrix": None,  # P_k from previous step
            "alpha_t": None,            # mixing coefficient from previous step
            "r_eff": None,              # effective rank from previous step
            "k_t": None,                # subspace dimension from previous step
            "initialized": False,
        }

        if torch.distributed.get_rank() == 0:
            print(f"[AdvantageDecomposer] Initialized with config: rho={decomp_config.rho}, "
                  f"k_min={decomp_config.k_min}, k_max={decomp_config.k_max}, "
                  f"use_relu_exploration={decomp_config.use_relu_exploration}")

    def _apply_advantage_decomposition(
        self,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        proxy_result,
        metrics_dict: dict,
    ) -> torch.Tensor:
        """Apply gradient-guided advantage decomposition.

        This method decomposes the token-level advantages into consolidation and
        exploration channels based on gradient geometry, then recombines them
        with an auto-balanced mixing coefficient.

        Args:
            advantages: Token-level advantages (B, seq_len).
            response_mask: Response mask (B, seq_len).
            proxy_result: Result from gradient proxy computation.
            metrics_dict: Dictionary to store decomposition metrics.

        Returns:
            Adjusted token-level advantages (B, seq_len).
        """
        if self.advantage_decomposer is None:
            return advantages

        B = advantages.shape[0]
        device = advantages.device
        dtype = advantages.dtype

        grads = proxy_result.grads_flat  # (B, D)

        # Collapse token-level advantages to per-sample advantages
        # Use response_mask to compute weighted average per sample
        adv_per_sample = collapse_token_level_advantage(advantages, response_mask)  # (B,)

        # Perform decomposition
        try:
            result = self.advantage_decomposer.decompose(
                gradient_features=grads,
                advantages=adv_per_sample,
            )

            # Update cached state for next iteration (if using delayed mode)
            # This is useful when we want to avoid extra computation
            self._cached_decomposer_state["initialized"] = True
            self._cached_decomposer_state["alpha_t"] = result.alpha_t
            self._cached_decomposer_state["r_eff"] = result.metrics.get("adv_decomp/participation_ratio")
            self._cached_decomposer_state["k_t"] = result.k_t

            # Expand back to token-level
            adjusted_advantages = expand_token_level_advantage(
                result.advantage_final, response_mask
            )

            # Log metrics
            if self.advantage_decomposer.config.log_metrics:
                decomp_metrics = result.to_dict()
                metrics_dict.update(decomp_metrics)

                # Also log the change in advantage statistics
                orig_mean = adv_per_sample.mean().item()
                orig_std = adv_per_sample.std().item()
                new_mean = result.advantage_final.mean().item()
                new_std = result.advantage_final.std().item()

                metrics_dict.update({
                    "adv_decomp/orig_adv_mean": orig_mean,
                    "adv_decomp/orig_adv_std": orig_std,
                    "adv_decomp/change_mean": new_mean - orig_mean,
                    "adv_decomp/change_std_ratio": new_std / (orig_std + 1e-8),
                })

            return adjusted_advantages

        except Exception as e:
            # If decomposition fails, fall back to original advantages
            if torch.distributed.get_rank() == 0:
                import traceback
                print(f"[AdvantageDecomposer] Error during decomposition: {e}")
                traceback.print_exc()
            metrics_dict["adv_decomp/error"] = 1.0
            return advantages

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, return_for_grad_analysis=False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            If return_for_grad_analysis=True, also returns:
            logits: # (bs, response_len, vocab_size) - with grad_fn
            hidden_states: # (bs, response_len, hidden_dim) - lm_head input
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                # For gradient analysis, we need hidden_states
                if return_for_grad_analysis:
                    extra_args["output_hidden_states"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                logits_for_grad = None
                hidden_states_for_grad = None

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

                    if return_for_grad_analysis:
                        # Keep logits with grad_fn for gradient analysis
                        logits_for_grad = logits  # (bsz, response_length, vocab_size)
                        # Get last hidden state (input to lm_head)
                        # hidden_states is a tuple of (n_layers + 1) tensors
                        # Last one is the final layer output before lm_head
                        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
                            last_hidden = output.hidden_states[-1]  # (bsz, seqlen, hidden_dim)
                            hidden_states_for_grad = last_hidden[:, -response_length - 1 : -1, :].detach()

                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if return_for_grad_analysis:
                return entropy, log_probs, logits_for_grad, hidden_states_for_grad
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include uid for gradient redundancy analysis
        if "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # Check if gradient redundancy logging is enabled
                    grad_red_cfg = self.config.get("gradient_redundancy", {})
                    log_grad_red = grad_red_cfg.get("enabled", False) and micro_batch.meta_info.get(
                        "log_grad_redundancy", True
                    )

                    # Check if advantage decomposition is enabled
                    adv_decomp_cfg = self.config.get("advantage_decomposition", {})
                    use_adv_decomp = adv_decomp_cfg.get("enabled", False) and self.advantage_decomposer is not None

                    # Determine if we need gradient analysis (for either redundancy logging or decomposition)
                    need_grad_analysis = (log_grad_red or use_adv_decomp) and not self.use_remove_padding and not self.use_fused_kernels

                    # all return: (bsz, response_length)
                    # When gradient redundancy is enabled, also get logits and hidden_states
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    if need_grad_analysis:
                        # Use special forward that returns logits and hidden_states for gradient analysis
                        forward_result = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            return_for_grad_analysis=True,
                        )
                        entropy, log_prob, logits_for_grad, hidden_states_for_grad = forward_result
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )
                        logits_for_grad = None
                        hidden_states_for_grad = None

                    # ========================================================================
                    # Shared Proxy Gradient Computation
                    # ========================================================================
                    # Compute gradient proxy ONCE and share between:
                    # 1. Advantage decomposition (if enabled)
                    # 2. Gradient redundancy logging (if enabled)
                    # This avoids duplicate computation and reduces memory usage.
                    shared_proxy_result = None
                    shared_group_ids = None

                    if need_grad_analysis and logits_for_grad is not None and hidden_states_for_grad is not None:
                        try:
                            # Get group_ids for gradient analysis (shared between both uses)
                            uid_field = micro_batch.non_tensor_batch.get("uid", None)
                            if uid_field is None:
                                shared_group_ids = torch.arange(
                                    advantages.shape[0], device=advantages.device, dtype=torch.long
                                )
                            else:
                                _, inverse_indices = np.unique(uid_field, return_inverse=True)
                                shared_group_ids = torch.as_tensor(inverse_indices, device=advantages.device, dtype=torch.long)

                            # Compute per-sample log_prob sum as proxy loss
                            # This gives us gradient direction without computing full policy loss
                            per_sample_logprob = (log_prob * response_mask).sum(dim=-1)  # (B,)

                            # Compute per-sample gradients (ONCE for both decomposition and redundancy)
                            shared_proxy_result = compute_per_sample_grads_for_lm_head(
                                per_sample_loss=-per_sample_logprob,  # negative because we maximize log_prob
                                logits=logits_for_grad,
                                hidden_states=hidden_states_for_grad,
                                response_mask=response_mask,
                                group_ids=shared_group_ids,
                            )

                            # Free intermediate tensors immediately after proxy computation
                            del per_sample_logprob

                        except Exception as e:
                            if torch.distributed.get_rank() == 0:
                                import traceback
                                print(f"[GradProxy] Error computing shared gradient proxy: {e}")
                                traceback.print_exc()
                            micro_batch_metrics["grad_proxy/error"] = 1.0

                    # ========================================================================
                    # Advantage Decomposition (before policy loss computation)
                    # ========================================================================
                    if use_adv_decomp and shared_proxy_result is not None:
                        try:
                            # Apply advantage decomposition using shared proxy result
                            advantages = self._apply_advantage_decomposition(
                                advantages=advantages,
                                response_mask=response_mask,
                                proxy_result=shared_proxy_result,
                                metrics_dict=micro_batch_metrics,
                            )
                        except Exception as e:
                            if torch.distributed.get_rank() == 0:
                                import traceback
                                print(f"[AdvDecomp] Error applying advantage decomposition: {e}")
                                traceback.print_exc()
                            micro_batch_metrics["adv_decomp/error"] = 1.0

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    # Optional: log gradient redundancy for research (using shared proxy result)
                    if log_grad_red:
                        if shared_proxy_result is not None:
                            try:
                                red_metrics = compute_gradient_redundancy(
                                    shared_proxy_result,
                                    advantages=collapse_token_level_advantage(advantages, response_mask),
                                    eps=grad_red_cfg.get("eps", 1e-9),
                                )

                                micro_batch_metrics.update(
                                    {
                                        "grad_redundancy/rin_mean": red_metrics["rin_mean"].detach().item(),
                                        "grad_redundancy/rin_raw_mean": red_metrics["rin_raw_mean"].detach().item(),
                                        "grad_redundancy/rout": red_metrics["rout"].detach().item(),
                                        "grad_redundancy/neff_in_mean": red_metrics["neff_in_mean"].detach().item(),
                                        "grad_redundancy/neff_out": red_metrics["neff_out"].detach().item(),
                                        "grad_redundancy/pr_in_mean": red_metrics["pr_in_mean"].detach().item(),
                                        "grad_redundancy/pr_total": red_metrics["pr_total"].detach().item(),
                                    }
                                )

                                if grad_red_cfg.get("debug_logging", False):
                                    micro_batch_metrics.update(
                                        {
                                            "grad_redundancy/debug/group_ids_len": shared_group_ids.numel() if shared_group_ids is not None else 0,
                                            "grad_redundancy/debug/unique_groups": shared_group_ids.unique().numel() if shared_group_ids is not None else 0,
                                            "grad_redundancy/debug/missing_grads": shared_proxy_result.missing_grads,
                                            "grad_redundancy/debug/grad_norm_mean": shared_proxy_result.grad_norms.mean().item(),
                                            "grad_redundancy/debug/grad_norm_max": shared_proxy_result.grad_norms.max().item(),
                                        }
                                    )
                                    print(f"[GRAD_DEBUG] rin_mean={red_metrics['rin_mean'].item():.4f}, "
                                          f"rin_raw_mean={red_metrics['rin_raw_mean'].item():.4f}, "
                                          f"pr_in={red_metrics['pr_in_mean'].item():.2f}, "
                                          f"pr_total={red_metrics['pr_total'].item():.2f}, "
                                          f"grad_norm_mean={shared_proxy_result.grad_norms.mean().item():.4f}, "
                                          f"missing={shared_proxy_result.missing_grads}")

                            except Exception as e:  # noqa: BLE001
                                micro_batch_metrics["grad_redundancy/error"] = 1.0
                                if grad_red_cfg.get("debug_logging", False):
                                    import traceback
                                    print(f"[GRAD_DEBUG] Error: {e}")
                                    traceback.print_exc()
                        else:
                            # Gradient analysis not available
                            if grad_red_cfg.get("debug_logging", False):
                                print("[GRAD_DEBUG] Gradient analysis not available (use_remove_padding=False and use_fused_kernels=False required)")
                            micro_batch_metrics["grad_redundancy/not_available"] = 1.0

                    # ========================================================================
                    # Memory Cleanup Before Backward
                    # ========================================================================
                    # Free large intermediate tensors to reduce memory pressure during backward
                    if logits_for_grad is not None:
                        del logits_for_grad
                        logits_for_grad = None
                    if hidden_states_for_grad is not None:
                        del hidden_states_for_grad
                        hidden_states_for_grad = None
                    if shared_proxy_result is not None:
                        del shared_proxy_result
                        shared_proxy_result = None
                    if shared_group_ids is not None:
                        del shared_group_ids
                        shared_group_ids = None
                    # Force CUDA memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
