from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import Tensor

if TYPE_CHECKING:
    from .proxy import ProxyResult

__all__ = ["DecompositionConfig", "DecompositionResult", "AdvantageDecomposer", "create_decomposer", "decompose_advantage"]


@dataclass
class DecompositionConfig:
    enabled: bool = False
    rho: float = 0.5
    k_min: int = 1
    k_max: int = 64
    use_relu_exploration: bool = True
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    normalize_final: bool = False
    chunk_size: int = 64
    use_fp32_gram: bool = True
    distributed_gram: bool = False
    log_metrics: bool = True
    log_eigenspectrum: bool = False

    def __post_init__(self):
        assert 0 < self.rho <= 2.0
        assert self.k_min >= 1
        assert self.k_max >= self.k_min
        assert 0 <= self.alpha_min <= self.alpha_max <= 1.0


@dataclass
class DecompositionResult:
    advantage_final: Tensor
    advantage_main: Tensor
    advantage_exp: Tensor
    alpha_t: float
    k_t: int
    eigenvalues: Optional[Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adv_decomp/alpha_t": self.alpha_t, "adv_decomp/k_t": self.k_t,
            "adv_decomp/adv_main_mean": self.advantage_main.mean().item(),
            "adv_decomp/adv_main_std": self.advantage_main.std().item(),
            "adv_decomp/adv_exp_mean": self.advantage_exp.mean().item(),
            "adv_decomp/adv_exp_std": self.advantage_exp.std().item(),
            "adv_decomp/adv_final_mean": self.advantage_final.mean().item(),
            "adv_decomp/adv_final_std": self.advantage_final.std().item(),
        } | self.metrics


class AdvantageDecomposer:
    def __init__(self, config: DecompositionConfig):
        self.config = config
        self._cached_projection: Optional[Tensor] = None
        self._cached_alpha: Optional[float] = None

    def compute_gram_matrix(self, grads: Tensor, chunk_size: Optional[int] = None) -> Tensor:
        B, D = grads.shape
        chunk_size = chunk_size or self.config.chunk_size
        device = grads.device
        compute_dtype = torch.float32 if self.config.use_fp32_gram else grads.dtype
        grads_c = grads.to(compute_dtype)

        if B <= chunk_size * 2:
            return (grads_c @ grads_c.T / B).to(grads.dtype)

        gram = torch.zeros(B, B, device=device, dtype=compute_dtype)
        for i in range(0, B, chunk_size):
            ie = min(i + chunk_size, B)
            gi = grads_c[i:ie]
            for j in range(0, B, chunk_size):
                je = min(j + chunk_size, B)
                gram[i:ie, j:je] = gi @ grads_c[j:je].T / B
        return gram.to(grads.dtype)

    def compute_gram_matrix_distributed(self, grads_local: Tensor, process_group: Optional[dist.ProcessGroup] = None) -> Tensor:
        if not dist.is_initialized() or not self.config.distributed_gram:
            return self.compute_gram_matrix(grads_local)
        ws = dist.get_world_size(process_group)
        if ws == 1:
            return self.compute_gram_matrix(grads_local)
        B_local, D = grads_local.shape
        device, dtype = grads_local.device, grads_local.dtype
        grads_list = [torch.empty(B_local, D, device=device, dtype=dtype) for _ in range(ws)]
        dist.all_gather(grads_list, grads_local, group=process_group)
        return self.compute_gram_matrix(torch.cat(grads_list, dim=0))

    @staticmethod
    def compute_participation_ratio(gram: Tensor, eps: float = 1e-10) -> float:
        trace = torch.diagonal(gram).sum()
        return ((trace ** 2) / ((gram ** 2).sum() + eps)).item()

    def compute_adaptive_k(self, pr: float, B: int) -> int:
        k_raw = int(round(self.config.rho * pr))
        return max(self.config.k_min, min(k_raw, self.config.k_max, B - 1))

    @staticmethod
    def compute_projection_matrix(eigenvectors: Tensor, k: int) -> Tensor:
        V_k = eigenvectors[:, -k:]
        return V_k @ V_k.T

    def decompose(
        self,
        gradient_features: Tensor,
        advantages: Tensor,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> DecompositionResult:
        B = advantages.shape[0]
        device, dtype = advantages.device, advantages.dtype
        advantages = advantages.view(-1)
        assert advantages.shape[0] == B
        metrics: Dict[str, float] = {}

        gram = self.compute_gram_matrix_distributed(gradient_features, process_group) if self.config.distributed_gram else self.compute_gram_matrix(gradient_features)
        gram = (gram + gram.T) / 2

        gram_f32 = gram.float()
        pr = self.compute_participation_ratio(gram_f32)
        k_t = self.compute_adaptive_k(pr, B)

        try:
            if k_t < B and B >= 3 * k_t:
                eigenvalues, eigenvectors = torch.lobpcg(gram_f32, k=k_t, largest=True)
            else:
                eigenvalues_all, eigenvectors_all = torch.linalg.eigh(gram_f32)
                eigenvalues = eigenvalues_all[-k_t:]
                eigenvectors = eigenvectors_all[:, -k_t:]
        except RuntimeError:
            gram_f32 = gram_f32 + 1e-6 * torch.eye(B, device=device, dtype=torch.float32)
            eigenvalues_all, eigenvectors_all = torch.linalg.eigh(gram_f32)
            eigenvalues = eigenvalues_all[-k_t:]
            eigenvectors = eigenvectors_all[:, -k_t:]
            metrics["adv_decomp/eigh_fallback"] = 1.0
        order = torch.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        eigenvectors, eigenvalues = eigenvectors.to(dtype), eigenvalues.to(dtype)

        signed_updates = advantages.unsqueeze(-1) * gradient_features
        update_sum_norm_sq = torch.linalg.vector_norm(signed_updates.sum(dim=0)).square()
        update_norms_sq = torch.linalg.vector_norm(signed_updates, dim=-1).square().sum()
        neff = (update_sum_norm_sq / (update_norms_sq + 1e-9)).item()
        alpha_t = max(self.config.alpha_min, min(1.0 - neff, self.config.alpha_max))

        if self.config.log_metrics:
            metrics["adv_decomp/participation_ratio"] = pr
            metrics["adv_decomp/neff"] = neff

        P_k = self.compute_projection_matrix(eigenvectors, k_t)
        I_minus_P_k = torch.eye(B, device=device, dtype=dtype) - P_k

        advantage_main = P_k @ advantages
        advantage_exp = I_minus_P_k @ (torch.relu(advantages) if self.config.use_relu_exploration else advantages)
        advantage_final = advantage_main + alpha_t * advantage_exp

        if self.config.normalize_final:
            orig_std, final_std = advantages.std(), advantage_final.std()
            if final_std > 1e-8:
                advantage_final = advantage_final * (orig_std / final_std)

        if self.config.log_metrics:
            metrics["adv_decomp/eigenvalue_max"] = eigenvalues[-1].item()
            metrics["adv_decomp/eigenvalue_min"] = eigenvalues[0].item()
            metrics["adv_decomp/eigenvalue_ratio"] = eigenvalues[-1].item() / (eigenvalues[0].item() + 1e-10)
            main_c = (advantage_main ** 2).sum().item()
            exp_c = (advantage_exp ** 2).sum().item()
            total_c = main_c + alpha_t ** 2 * exp_c + 1e-10
            metrics["adv_decomp/main_contrib_ratio"] = main_c / total_c
            metrics["adv_decomp/exp_contrib_ratio"] = (alpha_t ** 2 * exp_c) / total_c
            if self.config.use_relu_exploration:
                metrics["adv_decomp/positive_adv_ratio"] = (advantages > 0).float().mean().item()

        return DecompositionResult(
            advantage_final=advantage_final, advantage_main=advantage_main, advantage_exp=advantage_exp,
            alpha_t=alpha_t, k_t=k_t,
            eigenvalues=eigenvalues if self.config.log_eigenspectrum else None, metrics=metrics,
        )

    def decompose_from_proxy(self, proxy_result: "ProxyResult", advantages: Tensor, process_group: Optional[dist.ProcessGroup] = None) -> DecompositionResult:
        grads = proxy_result.grads_flat
        return self.decompose(grads, advantages, process_group)


def create_decomposer(config: Union[DecompositionConfig, Dict[str, Any], None]) -> Optional[AdvantageDecomposer]:
    if config is None:
        return None
    if isinstance(config, dict):
        if not config.get("enabled", False):
            return None
        config = DecompositionConfig(**config)
    if not config.enabled:
        return None
    return AdvantageDecomposer(config)


def decompose_advantage(
    gradient_features: Tensor,
    advantages: Tensor,
    config: Optional[DecompositionConfig] = None,
    **kwargs,
) -> Tuple[Tensor, Dict[str, float]]:
    if config is None:
        config = DecompositionConfig(enabled=True)
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    result = AdvantageDecomposer(config).decompose(gradient_features, advantages)
    return result.advantage_final, result.to_dict()


def expand_token_level_advantage(advantage_per_sample: Tensor, response_mask: Tensor) -> Tensor:
    return advantage_per_sample.unsqueeze(-1) * response_mask


def collapse_token_level_advantage(advantage_token_level: Tensor, response_mask: Tensor) -> Tensor:
    masked_sum = (advantage_token_level * response_mask).sum(dim=-1)
    return masked_sum / response_mask.sum(dim=-1).clamp(min=1.0)


def normalize_gradients(grads: Tensor, eps: float = 1e-8) -> Tensor:
    norms = torch.linalg.norm(grads, dim=-1, keepdim=True)
    return grads / (norms + eps)
