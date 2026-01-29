from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
import torch
from torch import nn


@dataclass
class ProxyResult:
    grads_flat: torch.Tensor
    group_ids: torch.Tensor
    grad_norms: torch.Tensor
    param_sizes: List[int]
    param_names: List[str]
    missing_grads: int


class GradientProxy:
    def compute(self, losses: torch.Tensor, params: Sequence[nn.Parameter], group_ids: torch.Tensor, retain_graph: bool = False) -> ProxyResult:
        raise NotImplementedError


class BackwardLastLayerProxy(GradientProxy):
    def __init__(self, create_graph: bool = False):
        self.create_graph = create_graph

    def compute(self, losses: torch.Tensor, params: Sequence[nn.Parameter], group_ids: torch.Tensor, retain_graph: bool = False) -> ProxyResult:
        assert losses.dim() == 1 and losses.shape[0] == group_ids.shape[0]
        param_sizes = [p.numel() for p in params]
        param_names = [getattr(p, "_name", f"param_{i}") for i, p in enumerate(params)]
        grads_flat, grad_norms = [], []
        missing_count = 0
        for idx, loss in enumerate(losses):
            g_list = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph or (idx + 1 < losses.shape[0]), create_graph=self.create_graph)
            missing_count += sum(1 for g in g_list if g is None)
            replaced = [torch.zeros_like(p).detach() if g is None else g for g, p in zip(g_list, params, strict=True)]
            flat = torch.cat([g.reshape(-1) for g in replaced])
            grads_flat.append(flat)
            grad_norms.append(torch.linalg.norm(flat))
        return ProxyResult(torch.stack(grads_flat), group_ids.detach(), torch.stack(grad_norms), param_sizes, param_names, missing_count)


class ManualLmHeadProxy(GradientProxy):
    def __init__(self):
        self.saved_hidden_states = None
        self._hook_handle = None

    def register_hook(self, lm_head_module: nn.Module):
        def hook_fn(module, input, output):
            self.saved_hidden_states = input[0].detach().clone()
        self._hook_handle = lm_head_module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def compute_from_logits(self, per_sample_loss: torch.Tensor, logits: torch.Tensor, response_mask: torch.Tensor, group_ids: torch.Tensor, hidden_states: Optional[torch.Tensor] = None) -> ProxyResult:
        if hidden_states is None:
            hidden_states = self.saved_hidden_states
        if hidden_states is None:
            raise ValueError("hidden_states not provided")
        N, device, dtype = per_sample_loss.shape[0], per_sample_loss.device, hidden_states.dtype
        vocab_size, hidden_dim = logits.shape[2], hidden_states.shape[2]
        grads_flat_list, grad_norms_list, missing_count = [], [], 0

        for i in range(N):
            mask_i, h_i = response_mask[i], hidden_states[i]
            valid_idx = mask_i.bool()
            if valid_idx.sum().item() == 0:
                grad_w = torch.zeros(vocab_size * hidden_dim, device=device, dtype=dtype)
                missing_count += 1
            else:
                try:
                    grad_logits_full = torch.autograd.grad(per_sample_loss[i], logits, retain_graph=True, allow_unused=True)[0]
                    if grad_logits_full is None:
                        grad_w = torch.zeros(vocab_size * hidden_dim, device=device, dtype=dtype)
                        missing_count += 1
                    else:
                        grad_w = (grad_logits_full[i][valid_idx].T @ h_i[valid_idx]).reshape(-1).to(dtype)
                except Exception:
                    grad_w = torch.zeros(vocab_size * hidden_dim, device=device, dtype=dtype)
                    missing_count += 1
            grads_flat_list.append(grad_w)
            grad_norms_list.append(torch.linalg.norm(grad_w.float()).to(dtype))

        return ProxyResult(torch.stack(grads_flat_list), group_ids.detach(), torch.stack(grad_norms_list), [vocab_size * hidden_dim], ["lm_head.weight"], missing_count)


def compute_per_sample_grads_for_lm_head(per_sample_loss: torch.Tensor, logits: torch.Tensor, hidden_states: torch.Tensor, response_mask: torch.Tensor, group_ids: torch.Tensor, advantages: Optional[torch.Tensor] = None, rewards: Optional[torch.Tensor] = None) -> ProxyResult:
    proxy = ManualLmHeadProxy()
    result = proxy.compute_from_logits(per_sample_loss, logits, response_mask, group_ids, hidden_states)
    _check_env_and_enable_save()
    if _SAVE_CONFIG["enabled"] and not _SAVE_CONFIG["saved"]:
        extra = {"per_sample_loss": per_sample_loss}
        if advantages is not None:
            extra["advantages"] = advantages
        if rewards is not None:
            extra["rewards"] = rewards
        save_proxy_result(result, extra_data=extra)
    return result


def select_last_layer_parameters(model: nn.Module, fallback_to_lm_head: bool = True, extra_modules: Iterable[str] | None = None) -> List[nn.Parameter]:
    candidates = ["layers", "model.layers", "transformer.layers", "decoder.layers", "gpt_neox.layers", "model.decoder.layers"]

    def _resolve(obj, dotted: str):
        cur = obj
        for part in dotted.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    layer_module = None
    for cand in candidates:
        maybe = _resolve(model, cand)
        if maybe is not None and isinstance(maybe, (list, tuple, nn.ModuleList)) and len(maybe) > 0:
            layer_module = maybe[-1]
            break

    params: List[nn.Parameter] = []
    if layer_module is not None:
        params.extend(list(layer_module.parameters()))
    if not params and fallback_to_lm_head:
        for head in ["lm_head", "output", "embed_out"]:
            if hasattr(model, head):
                params.extend(list(getattr(model, head).parameters()))
                break
    if extra_modules:
        for name in extra_modules:
            mod = _resolve(model, name)
            if mod is not None:
                params.extend(list(mod.parameters()))

    seen, unique = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            unique.append(p)
    if not unique:
        raise ValueError("No parameters found")
    return unique


def build_gradient_proxy(method: str = "backward_last_layer", **kwargs) -> GradientProxy:
    method = method.lower()
    if method in {"backward", "backward_last_layer"}:
        return BackwardLastLayerProxy(**kwargs)
    raise ValueError(f"Unknown method: {method}")


_SAVE_CONFIG = {
    "enabled": False, "save_dir": "/tmp/proxy_data", "save_step": 0, "current_step": -1,
    "saved": False, "save_extra_data": True, "accumulate_batch": True, "accumulated_data": [],
    "num_micro_batches_per_batch": 8,
}


def _check_env_and_enable_save():
    if _SAVE_CONFIG["enabled"]:
        return
    if os.environ.get("PROXY_SAVE_ENABLED", "0") == "1":
        _SAVE_CONFIG["enabled"] = True
        _SAVE_CONFIG["save_dir"] = os.environ.get("PROXY_SAVE_DIR", "/tmp/proxy_data")
        _SAVE_CONFIG["save_step"] = int(os.environ.get("PROXY_SAVE_STEP", "0"))
        _SAVE_CONFIG["num_micro_batches_per_batch"] = int(os.environ.get("PROXY_NUM_MICRO_BATCHES", "8"))
        os.makedirs(_SAVE_CONFIG["save_dir"], exist_ok=True)
        return
    trigger = "/tmp/proxy_save_trigger.txt"
    if os.path.exists(trigger):
        try:
            with open(trigger) as f:
                cfg = dict(line.split("=", 1) for line in f.read().strip().split("\n") if "=" in line)
            _SAVE_CONFIG["enabled"] = True
            _SAVE_CONFIG["save_dir"] = cfg.get("PROXY_SAVE_DIR", "/tmp/proxy_data")
            _SAVE_CONFIG["save_step"] = int(cfg.get("PROXY_SAVE_STEP", "0"))
            _SAVE_CONFIG["num_micro_batches_per_batch"] = int(cfg.get("PROXY_NUM_MICRO_BATCHES", "8"))
            os.makedirs(_SAVE_CONFIG["save_dir"], exist_ok=True)
        except Exception:
            pass


_check_env_and_enable_save()


def enable_proxy_save(save_dir: str = "/tmp/proxy_data", save_step: int = 0):
    os.makedirs(save_dir, exist_ok=True)
    _SAVE_CONFIG.update({"enabled": True, "save_dir": save_dir, "save_step": save_step, "current_step": -1, "saved": False})


def disable_proxy_save():
    _SAVE_CONFIG["enabled"] = False


def accumulate_proxy_result(proxy_result: ProxyResult, extra_data: Optional[dict] = None):
    if not _SAVE_CONFIG["enabled"] or _SAVE_CONFIG["saved"]:
        return
    data = {"grads_flat": proxy_result.grads_flat.detach().cpu().float(), "group_ids": proxy_result.group_ids.detach().cpu(), "grad_norms": proxy_result.grad_norms.detach().cpu().float(), "param_sizes": proxy_result.param_sizes, "param_names": proxy_result.param_names}
    if extra_data:
        for k, v in extra_data.items():
            data[k] = v.detach().cpu().float() if isinstance(v, torch.Tensor) else v
    _SAVE_CONFIG["accumulated_data"].append(data)
    if len(_SAVE_CONFIG["accumulated_data"]) >= _SAVE_CONFIG["num_micro_batches_per_batch"]:
        flush_accumulated_proxy()


def flush_accumulated_proxy(force: bool = False):
    if not force and (not _SAVE_CONFIG["enabled"] or _SAVE_CONFIG["saved"] or not _SAVE_CONFIG["accumulated_data"]):
        return
    _SAVE_CONFIG["current_step"] += 1
    if not force and _SAVE_CONFIG["current_step"] != _SAVE_CONFIG["save_step"]:
        _SAVE_CONFIG["accumulated_data"] = []
        return

    acc = _SAVE_CONFIG["accumulated_data"]
    all_grads, all_gids, all_norms, all_extra = [], [], [], {}
    gid_offset = 0
    for d in acc:
        all_grads.append(d["grads_flat"])
        offset_ids = d["group_ids"] + gid_offset
        all_gids.append(offset_ids)
        gid_offset = offset_ids.max().item() + 1
        all_norms.append(d["grad_norms"])
        for k, v in d.items():
            if k not in ["grads_flat", "group_ids", "grad_norms", "param_sizes", "param_names"]:
                all_extra.setdefault(k, []).append(v)

    save_data = {"grads_flat": torch.cat(all_grads), "group_ids": torch.cat(all_gids), "grad_norms": torch.cat(all_norms), "param_sizes": acc[0]["param_sizes"], "param_names": acc[0]["param_names"], "missing_grads": 0, "num_micro_batches": len(acc)}
    for k, vl in all_extra.items():
        save_data[k] = torch.cat(vl) if isinstance(vl[0], torch.Tensor) else vl

    rank = 0
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        pass
    torch.save(save_data, os.path.join(_SAVE_CONFIG["save_dir"], f"proxy_result_rank{rank}.pt"))
    _SAVE_CONFIG["saved"] = True
    _SAVE_CONFIG["accumulated_data"] = []


def save_proxy_result(proxy_result: ProxyResult, extra_data: Optional[dict] = None, force: bool = False):
    if not force and (not _SAVE_CONFIG["enabled"] or _SAVE_CONFIG["saved"]):
        return
    if _SAVE_CONFIG["accumulate_batch"] and not force:
        accumulate_proxy_result(proxy_result, extra_data)
        return
    _SAVE_CONFIG["current_step"] += 1
    if not force and _SAVE_CONFIG["current_step"] != _SAVE_CONFIG["save_step"]:
        return
    save_data = {"grads_flat": proxy_result.grads_flat.detach().cpu().float(), "group_ids": proxy_result.group_ids.detach().cpu(), "grad_norms": proxy_result.grad_norms.detach().cpu().float(), "param_sizes": proxy_result.param_sizes, "param_names": proxy_result.param_names, "missing_grads": proxy_result.missing_grads}
    if extra_data:
        for k, v in extra_data.items():
            save_data[k] = v.detach().cpu().float() if isinstance(v, torch.Tensor) else v
    torch.save(save_data, os.path.join(_SAVE_CONFIG["save_dir"], "proxy_result.pt"))
    _SAVE_CONFIG["saved"] = True


def load_proxy_result(save_dir: str = "/tmp/proxy_data") -> dict:
    path = os.path.join(save_dir, "proxy_result.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No proxy result at {path}")
    return torch.load(path, map_location="cpu")


def get_proxy_save_config() -> dict:
    return _SAVE_CONFIG.copy()
