<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2026-blue" alt="NeurIPS 2026">
  <a href="https://arxiv.org/abs/2606.05800"><img src="https://img.shields.io/badge/arXiv-2606.05800-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License: Apache-2.0">
</p>

# SALT: When More Rollouts Don't Help in Group-Based Policy Optimization and How to Make Them Matter

English | [中文](README_zh.md)

Our work is accepted at **NeurIPS 2026 Main Track** as a poster.

> **TL;DR:** SALT makes additional RLVR rollouts matter by diagnosing signed low-rank gradient redundancy and adaptively reweighting dominant and spectral-tail update channels.

## Overview

Group-based policy optimization methods sample multiple responses for each prompt and construct relative learning signals within the group. Increasing the rollout group size, however, does not necessarily produce a stronger update: per-sample policy-gradient features can concentrate in a low-rank geometry, while opposing signed directions cancel during aggregation.

SALT addresses this by:

- Diagnosing rollout effectiveness with **participation ratio (PR)** and the **effective sample size of signed updates**
- Estimating a **dominant spectral subspace** from mini-batch LM-head gradient geometry
- Decomposing group-relative coefficients into **dominant** and **spectral-tail** channels
- Adaptively reweighting the spectral-tail channel when signed cancellation is severe
- Acting as a plug-in for group-based RLVR objectives without changing the reward model or rollout sampler

## Project Structure

```
SALT/
├── README.md
├── README_zh.md
├── environment.sh
└── verl/
    ├── verl/trainer/gradient/
    │   ├── advantage_decomposition.py   # SALT spectral decomposition and reweighting
    │   ├── gradient.py                  # Gradient-redundancy diagnostics
    │   └── proxy.py                     # LM-head per-sample gradient proxy
    ├── verl/workers/actor/
    │   └── dp_actor.py                  # SALT integration in the actor update
    └── verl/trainer/config/actor/
        └── actor.yaml                   # SALT configuration
```

## Quick Start

### Installation

```bash
git clone https://github.com/Chang-pw/SALT.git
cd SALT
source environment.sh
```

FlashAttention should be installed separately with a wheel or source build compatible with the local CUDA and PyTorch versions.

### Configuration

SALT is disabled by default. Enable it in the actor configuration:

```yaml
actor_rollout_ref:
  model:
    use_remove_padding: false
  actor:
    advantage_decomposition:
      enabled: true
      rho: 1.0
      k_min: 1
      k_max: 64
```

For a matched GRPO or DAPO baseline, keep `advantage_decomposition.enabled: false` and leave the remaining training configuration unchanged.

### Data

Prepare the training and evaluation datasets in the Parquet format expected by verl, then set the corresponding `data.train_files` and `data.val_files` paths in the experiment configuration.

### Training

Run the desired GRPO- or DAPO-style verl recipe after applying the configuration above:

```bash
cd verl
sh recipe/dapo/test_dapo_7b_math.sh
```

The same backbone recipe can be used for the baseline and SALT by toggling only the SALT configuration.

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable SALT advantage reweighting |
| `rho` | `1.0` | Scale the participation ratio when selecting the dominant-subspace dimension |
| `k_min` | `1` | Minimum dominant-subspace dimension |
| `k_max` | `64` | Maximum dominant-subspace dimension |
| `use_relu_exploration` | `true` | Apply the positive-part operator to spectral-tail coefficients |
| `alpha_min` | `0.0` | Minimum adaptive spectral-tail mixing coefficient |
| `alpha_max` | `1.0` | Maximum adaptive spectral-tail mixing coefficient |
| `use_fp32_gram` | `true` | Compute the Gram matrix in FP32 |
| `log_metrics` | `true` | Log PR, effective sample size, mixing coefficient, and selected dimension |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{chang2026saltrolloutsdonthelp,
  title={SALT: When More Rollouts Don't Help in Group-Based Policy Optimization and How to Make Them Matter},
  author={Powei Chang and Jinpeng Zhang and Chaoqun Sun and MiniWell Tsao and Lianrui Li and Jianxiang Xiang and Chenyu Wang and Yukang Gao and Dongying Kong},
  year={2026},
  eprint={2606.05800},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.05800},
}
```

## Acknowledgments

This implementation is built on [verl](https://github.com/volcengine/verl).

## License

The vendored verl implementation retains its Apache-2.0 license and notices. See [verl/LICENSE](verl/LICENSE) and [verl/Notice.txt](verl/Notice.txt) for details.
