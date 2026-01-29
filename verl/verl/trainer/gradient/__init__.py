from .gradient import compute_redundancy, compute_loss_redundancy, set_config
from .gradient import compute_gradient_redundancy, compute_loss_based_redundancy, set_grad_redundancy_config
from .proxy import (
    BackwardLastLayerProxy, GradientProxy, ManualLmHeadProxy, ProxyResult,
    build_gradient_proxy, compute_per_sample_grads_for_lm_head, select_last_layer_parameters,
    enable_proxy_save, disable_proxy_save, save_proxy_result, load_proxy_result,
    get_proxy_save_config, accumulate_proxy_result, flush_accumulated_proxy,
)
from .advantage_decomposition import (
    AdvantageDecomposer, DecompositionConfig, DecompositionResult,
    create_decomposer, decompose_advantage, expand_token_level_advantage,
    collapse_token_level_advantage, normalize_gradients,
)

__all__ = [
    "BackwardLastLayerProxy", "GradientProxy", "ManualLmHeadProxy", "ProxyResult",
    "build_gradient_proxy", "compute_per_sample_grads_for_lm_head", "select_last_layer_parameters",
    "enable_proxy_save", "disable_proxy_save", "save_proxy_result", "load_proxy_result",
    "get_proxy_save_config", "accumulate_proxy_result", "flush_accumulated_proxy",
    "compute_redundancy", "compute_loss_redundancy", "set_config",
    "compute_gradient_redundancy", "compute_loss_based_redundancy", "set_grad_redundancy_config",
    "AdvantageDecomposer", "DecompositionConfig", "DecompositionResult",
    "create_decomposer", "decompose_advantage", "expand_token_level_advantage",
    "collapse_token_level_advantage", "normalize_gradients",
]
