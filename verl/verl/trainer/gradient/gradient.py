from __future__ import annotations
from typing import Dict, Optional
import torch
from .proxy import ProxyResult

_CONFIG = {
    "use_random_projection": False,
    "projection_dim": 2048,
    "compute_pr_total": True,
    "max_samples_for_pr": 128,
}


def set_config(**kwargs):
    _CONFIG.update(kwargs)


def _random_project(grads: torch.Tensor, target_dim: int, seed: int = 42) -> torch.Tensor:
    N, D = grads.shape
    if D <= target_dim:
        return grads
    device, dtype = grads.device, grads.dtype
    gen = torch.Generator(device='cpu').manual_seed(seed)
    scale = (3.0 / target_dim) ** 0.5
    chunk_size = min(D, 65536)
    result = torch.zeros(N, target_dim, device=device, dtype=dtype)
    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        chunk_d = end - start
        rv = torch.rand(chunk_d, target_dim, generator=gen)
        proj = torch.zeros(chunk_d, target_dim, dtype=dtype)
        proj[rv < 1/6] = -scale
        proj[rv > 5/6] = scale
        proj = proj.to(device)
        result += torch.mm(grads[:, start:end].float(), proj.float()).to(dtype)
        del proj
    return result


def _pairwise_pos_cos(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.size(0) < 2:
        return x.new_tensor(0.0)
    sim = torch.triu(x @ x.T, diagonal=1)
    return torch.clamp_min(sim, 0.0).sum()


def _pairwise_raw_cos(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.size(0) < 2:
        return x.new_tensor(0.0)
    return torch.triu(x @ x.T, diagonal=1).sum()


def _neff(updates: torch.Tensor, eps: float) -> torch.Tensor:
    if updates.size(0) == 0:
        return updates.new_tensor(0.0)
    num = torch.linalg.norm(torch.sum(updates, dim=0), ord=2) ** 2
    denom = torch.sum(torch.linalg.norm(updates, dim=-1) ** 2) + eps
    return num / denom


def _pr_fast(grads: torch.Tensor, eps: float) -> torch.Tensor:
    m, d = grads.shape
    if m < 2:
        return grads.new_tensor(1.0)
    device = grads.device
    max_samples = _CONFIG["max_samples_for_pr"]
    if m > max_samples:
        idx = torch.randperm(m, device=device)[:max_samples]
        grads = grads[idx]
        m = max_samples
    try:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        chunk_size = min(d, 32768)
        gram = torch.zeros(m, m, device=device, dtype=torch.float32)
        for start in range(0, d, chunk_size):
            end = min(start + chunk_size, d)
            gc = grads[:, start:end].to(compute_dtype)
            gram += torch.mm(gc, gc.T).float()
            del gc
        gram = gram / m
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gram = torch.zeros(m, m, device=device, dtype=torch.float32)
            rc, cc = max(1, m // 4), min(d, 16384)
            for i in range(0, m, rc):
                ie = min(i + rc, m)
                for j in range(i, m, rc):
                    je = min(j + rc, m)
                    blk = torch.zeros(ie - i, je - j, device=device, dtype=torch.float32)
                    for k in range(0, d, cc):
                        ke = min(k + cc, d)
                        gi, gj = grads[i:ie, k:ke].float(), grads[j:je, k:ke].float()
                        blk += torch.mm(gi, gj.T)
                        del gi, gj
                    gram[i:ie, j:je] = blk / m
                    if i != j:
                        gram[j:je, i:ie] = blk.T / m
                    del blk
            torch.cuda.empty_cache()
        else:
            raise
    eigs = torch.clamp_min(torch.linalg.eigvalsh(gram), 0.0)
    pr = (eigs.sum() ** 2) / ((eigs ** 2).sum() + eps)
    del gram, eigs
    return pr


def compute_redundancy(
    proxy_result: ProxyResult,
    advantages: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> Dict[str, torch.Tensor]:
    grads = proxy_result.grads_flat
    group_ids = proxy_result.group_ids
    device = grads.device

    if _CONFIG["use_random_projection"]:
        proj_dim = _CONFIG["projection_dim"]
        if grads.shape[1] > proj_dim:
            grads = _random_project(grads, proj_dim)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    unique_ids = torch.unique(group_ids)
    rin_list, rin_raw_list, neff_in_list, pr_in_list, prompt_means, prompt_updates = [], [], [], [], [], []

    for gid in unique_ids:
        mask = group_ids == gid
        g = grads[mask]
        if g.numel() == 0:
            continue
        norms = torch.linalg.norm(g, dim=-1, keepdim=True)
        g_hat = g / (norms + eps)
        G = g.size(0)
        sum_pos = _pairwise_pos_cos(g_hat, eps)
        sum_raw = _pairwise_raw_cos(g_hat, eps)
        rin = 2 * sum_pos / (G * (G - 1)) if G > 1 else torch.zeros((), device=device)
        rin_raw = 2 * sum_raw / (G * (G - 1)) if G > 1 else torch.zeros((), device=device)
        rin_list.append(rin)
        rin_raw_list.append(rin_raw)
        u = g if advantages is None else advantages[mask].reshape(-1, 1) * g
        neff_in_list.append(_neff(u, eps))
        pr_in_list.append(_pr_fast(g, eps))
        prompt_means.append(g.mean(dim=0, keepdim=True))
        prompt_updates.append(u.mean(dim=0, keepdim=True))
        del g_hat, norms

    if len(prompt_means) == 0:
        zero, one = grads.new_tensor(0.0), grads.new_tensor(1.0)
        return {k: zero for k in ["rin_per_prompt", "rin_mean", "rin_raw_mean", "rin_median", "rin_p90", "rin_p95", "rout", "neff_in_per_prompt", "neff_in_mean", "neff_in_median", "neff_in_p10", "neff_out"]} | {"pr_in_mean": one, "pr_total": one}

    rin_t = torch.stack(rin_list)
    rin_raw_t = torch.stack(rin_raw_list)
    neff_in_t = torch.stack(neff_in_list)
    pr_in_t = torch.stack(pr_in_list)

    pm = torch.cat(prompt_means, dim=0)
    pu = torch.cat(prompt_updates, dim=0)
    mn = torch.linalg.norm(pm, dim=-1, keepdim=True)
    p_hat = pm / (mn + eps)
    sum_pos_out = _pairwise_pos_cos(p_hat, eps)
    B = pm.size(0)
    rout = 2 * sum_pos_out / (B * (B - 1)) if B > 1 else grads.new_tensor(0.0)
    neff_out = _neff(pu, eps)
    pr_total = _pr_fast(grads, eps) if _CONFIG["compute_pr_total"] else grads.new_tensor(1.0)

    del p_hat, mn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def q(t, qv):
        return torch.quantile(t.float(), torch.tensor(qv, device=t.device))

    return {
        "rin_per_prompt": rin_t, "rin_mean": rin_t.mean(), "rin_raw_mean": rin_raw_t.mean(),
        "rin_median": q(rin_t, 0.5), "rin_p90": q(rin_t, 0.9), "rin_p95": q(rin_t, 0.95),
        "rout": rout, "neff_in_per_prompt": neff_in_t, "neff_in_mean": neff_in_t.mean(),
        "neff_in_median": q(neff_in_t, 0.5), "neff_in_p10": q(neff_in_t, 0.1), "neff_out": neff_out,
        "pr_in_mean": pr_in_t.mean(), "pr_total": pr_total,
    }


def compute_loss_redundancy(
    per_sample_loss: torch.Tensor,
    group_ids: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
    per_token_loss: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> Dict[str, torch.Tensor]:
    device = per_sample_loss.device
    unique_ids = torch.unique(group_ids)
    var_in_list, mean_losses = [], []

    for gid in unique_ids:
        mask = group_ids == gid
        lg = per_sample_loss[mask]
        if lg.numel() < 2:
            continue
        var_in_list.append(lg.var())
        mean_losses.append(lg.mean())

    if len(var_in_list) == 0:
        zero = per_sample_loss.new_tensor(0.0)
        return {"loss_var_in_mean": zero, "loss_var_out": zero, "loss_cv_in_mean": zero, "loss_neff_in": zero, "loss_neff_out": zero}

    var_in_t = torch.stack(var_in_list)
    mean_t = torch.stack(mean_losses)
    var_out = mean_t.var() if len(mean_t) > 1 else per_sample_loss.new_tensor(0.0)

    cv_in_list, neff_in_list = [], []
    for gid in unique_ids:
        mask = group_ids == gid
        lg = per_sample_loss[mask]
        if lg.numel() < 2:
            continue
        cv_in_list.append(lg.std() / (lg.mean().abs() + eps))
        neff_in_list.append(lg.sum() ** 2 / ((lg ** 2).sum() + eps))

    cv_in_t = torch.stack(cv_in_list) if cv_in_list else per_sample_loss.new_tensor([0.0])
    neff_in_t = torch.stack(neff_in_list) if neff_in_list else per_sample_loss.new_tensor([0.0])
    neff_out = mean_t.sum() ** 2 / ((mean_t ** 2).sum() + eps) if len(mean_t) > 0 else per_sample_loss.new_tensor(0.0)

    return {
        "loss_var_in_mean": var_in_t.mean(), "loss_var_out": var_out,
        "loss_cv_in_mean": cv_in_t.mean(), "loss_neff_in": neff_in_t.mean(), "loss_neff_out": neff_out,
    }


# Backward compatibility aliases
compute_gradient_redundancy = compute_redundancy
compute_loss_based_redundancy = compute_loss_redundancy
set_grad_redundancy_config = set_config
