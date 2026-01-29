<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2026-blue" alt="NeurIPS 2026">
  <a href="https://arxiv.org/abs/2606.05800"><img src="https://img.shields.io/badge/arXiv-2606.05800-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License: Apache-2.0">
</p>

# SALT：当更多 Rollout 无法改善组式策略优化，以及如何让它们真正发挥作用

[English](README.md) | 中文

我们的工作已被 **NeurIPS 2026 Main Track** 接收为 Poster。

> **TL;DR：** SALT 通过诊断带符号的低秩梯度冗余，并自适应重加权主导通道与谱尾通道，使新增的 RLVR rollout 能够转化为更有效的策略更新。

## 方法概览

组式策略优化方法会针对每个 prompt 采样多个回答，并在组内构造相对学习信号。然而，增加 rollout 数量并不一定带来更强的更新：逐样本策略梯度特征可能集中在低秩几何结构中，同时带相反符号的更新方向会在聚合时相互抵消。

SALT 的核心包括：

- 使用 **参与率（Participation Ratio, PR）** 和 **带符号更新的有效样本量** 诊断 rollout 的有效性
- 根据 mini-batch 的 LM-head 梯度几何估计 **主导谱子空间**
- 将组相对系数分解为 **主导通道** 与 **谱尾通道**
- 当带符号抵消严重时，自适应增强谱尾通道
- 作为组式 RLVR 目标的即插即用组件，无需修改奖励模型或 rollout 采样器

## 项目结构

```
SALT/
├── README.md
├── README_zh.md
├── environment.sh
└── verl/
    ├── verl/trainer/gradient/
    │   ├── advantage_decomposition.py   # SALT 谱分解与重加权
    │   ├── gradient.py                  # 梯度冗余诊断
    │   └── proxy.py                     # LM-head 逐样本梯度代理
    ├── verl/workers/actor/
    │   └── dp_actor.py                  # 在 actor 更新中集成 SALT
    └── verl/trainer/config/actor/
        └── actor.yaml                   # SALT 配置
```

## 快速开始

### 安装

```bash
git clone https://github.com/Chang-pw/SALT.git
cd SALT
source environment.sh
```

FlashAttention 需要根据本机 CUDA 与 PyTorch 版本，单独安装兼容的 wheel 或从源码编译。

### 配置

SALT 默认关闭。在 actor 配置中启用：

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

如需运行严格匹配的 GRPO 或 DAPO 基线，只需保持 `advantage_decomposition.enabled: false`，其余训练配置不变。

### 数据

请按照 verl 所需格式准备 Parquet 训练集与评测集，并在实验配置中设置对应的 `data.train_files` 和 `data.val_files` 路径。

### 训练

应用上述配置后，运行所需的 GRPO 或 DAPO 风格 verl recipe：

```bash
cd verl
sh recipe/dapo/test_dapo_7b_math.sh
```

通过切换 SALT 配置，同一套骨干训练 recipe 可用于运行基线与 SALT。

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `false` | 启用 SALT advantage 重加权 |
| `rho` | `1.0` | 根据参与率选择主导子空间维度时的缩放系数 |
| `k_min` | `1` | 主导子空间的最小维度 |
| `k_max` | `64` | 主导子空间的最大维度 |
| `use_relu_exploration` | `true` | 对谱尾系数应用正部算子 |
| `alpha_min` | `0.0` | 自适应谱尾混合系数的下界 |
| `alpha_max` | `1.0` | 自适应谱尾混合系数的上界 |
| `use_fp32_gram` | `true` | 使用 FP32 计算 Gram 矩阵 |
| `log_metrics` | `true` | 记录 PR、有效样本量、混合系数及所选维度 |

## 引用

如果这项工作对你有帮助，请引用：

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

## 致谢

本实现基于 [verl](https://github.com/volcengine/verl) 构建。

## 许可证

仓库内置的 verl 实现保留其 Apache-2.0 许可证及声明。详见 [verl/LICENSE](verl/LICENSE) 和 [verl/Notice.txt](verl/Notice.txt)。
