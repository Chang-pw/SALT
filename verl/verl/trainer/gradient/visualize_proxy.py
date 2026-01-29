import argparse
import os
import glob
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_data(save_dir: str) -> dict:
    rank_files = sorted(glob.glob(os.path.join(save_dir, "proxy_result_rank*.pt")))
    if rank_files:
        all_grads, all_gids, all_norms, all_extra = [], [], [], {}
        gid_offset = 0
        for rf in rank_files:
            d = torch.load(rf, map_location="cpu")
            all_grads.append(d['grads_flat'])
            offset_ids = d['group_ids'] + gid_offset
            all_gids.append(offset_ids)
            gid_offset = offset_ids.max().item() + 1
            all_norms.append(d['grad_norms'])
            for k, v in d.items():
                if k not in ['grads_flat', 'group_ids', 'grad_norms', 'param_sizes', 'param_names', 'missing_grads', 'num_micro_batches']:
                    all_extra.setdefault(k, []).append(v)
        combined = {'grads_flat': torch.cat(all_grads), 'group_ids': torch.cat(all_gids), 'grad_norms': torch.cat(all_norms), 'param_sizes': d['param_sizes'], 'param_names': d['param_names']}
        for k, vl in all_extra.items():
            if isinstance(vl[0], torch.Tensor):
                combined[k] = torch.cat(vl)
        return combined
    path = os.path.join(save_dir, "proxy_result.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No proxy result at {path}")
    return torch.load(path, map_location="cpu")


def pca_visualization(grads: np.ndarray, group_ids: np.ndarray, advantages: Optional[np.ndarray] = None, rewards: Optional[np.ndarray] = None, output_dir: str = ".", n_components: int = 2):
    os.makedirs(output_dir, exist_ok=True)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    grads_n = grads / norms

    pca = PCA(n_components=min(n_components, grads.shape[0], grads.shape[1]))
    grads_pca = pca.fit_transform(grads_n)

    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    cmap = plt.cm.get_cmap('tab20', n_groups)
    g2c = {gid: cmap(i) for i, gid in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(12, 10))
    for gid in unique_groups:
        mask = group_ids == gid
        ax.scatter(grads_pca[mask, 0], grads_pca[mask, 1], c=[g2c[gid]], label=f'Prompt {gid}', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        if mask.sum() > 1:
            idx = np.where(mask)[0]
            for j in range(len(idx)):
                for k in range(j+1, len(idx)):
                    ax.plot([grads_pca[idx[j], 0], grads_pca[idx[k], 0]], [grads_pca[idx[j], 1], grads_pca[idx[k], 1]], c=g2c[gid], alpha=0.2, linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title('PCA of Per-Sample Gradients (by prompt)')
    if n_groups <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_by_prompt.png'), dpi=150, bbox_inches='tight')
    plt.close()

    if advantages is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        sc = ax.scatter(grads_pca[:, 0], grads_pca[:, 1], c=advantages, cmap='RdYlGn', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        plt.colorbar(sc, ax=ax, label='Advantage')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('PCA of Per-Sample Gradients (by advantage)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_by_advantage.png'), dpi=150, bbox_inches='tight')
        plt.close()

    if rewards is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        sc = ax.scatter(grads_pca[:, 0], grads_pca[:, 1], c=rewards, cmap='viridis', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        plt.colorbar(sc, ax=ax, label='Reward')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('PCA of Per-Sample Gradients (by reward)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_by_reward.png'), dpi=150, bbox_inches='tight')
        plt.close()

    n_comp = min(50, grads.shape[0], grads.shape[1])
    pca_full = PCA(n_components=n_comp)
    pca_full.fit(grads_n)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, n_comp + 1), pca_full.explained_variance_ratio_, alpha=0.7, label='Individual')
    ax.plot(range(1, n_comp + 1), cumsum, 'r-', marker='o', markersize=4, label='Cumulative')
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90%')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'), dpi=150, bbox_inches='tight')
    plt.close()

    cos_sim = grads_n @ grads_n.T
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    boundaries = [0]
    for gid in unique_groups:
        boundaries.append(boundaries[-1] + (group_ids == gid).sum())
    for b in boundaries[1:-1]:
        ax.axhline(y=b-0.5, color='black', linewidth=1)
        ax.axvline(x=b-0.5, color='black', linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    ax.set_title('Gradient Cosine Similarity Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    within_sims, across_sims = [], []
    for i in range(len(group_ids)):
        for j in range(i+1, len(group_ids)):
            sim = cos_sim[i, j]
            (within_sims if group_ids[i] == group_ids[j] else across_sims).append(sim)

    fig, ax = plt.subplots(figsize=(10, 6))
    if within_sims:
        ax.hist(within_sims, bins=50, alpha=0.7, label=f'Within-prompt (n={len(within_sims)})', density=True)
    if across_sims:
        ax.hist(across_sims, bins=50, alpha=0.7, label=f'Across-prompt (n={len(across_sims)})', density=True)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Gradient Cosine Similarities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if within_sims:
        ax.axvline(x=np.mean(within_sims), color='blue', linestyle='--', alpha=0.7)
    if across_sims:
        ax.axvline(x=np.mean(across_sims), color='orange', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return pca, grads_pca


def tsne_visualization(grads: np.ndarray, group_ids: np.ndarray, advantages: Optional[np.ndarray] = None, output_dir: str = ".", perplexity: int = 30):
    os.makedirs(output_dir, exist_ok=True)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    grads_n = grads / norms
    perplexity = min(perplexity, len(grads) - 1)
    grads_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(grads_n)

    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    cmap = plt.cm.get_cmap('tab20', n_groups)
    g2c = {gid: cmap(i) for i, gid in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(12, 10))
    for gid in unique_groups:
        mask = group_ids == gid
        ax.scatter(grads_tsne[mask, 0], grads_tsne[mask, 1], c=[g2c[gid]], label=f'Prompt {gid}', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE of Per-Sample Gradients (by prompt)')
    if n_groups <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_by_prompt.png'), dpi=150, bbox_inches='tight')
    plt.close()
    return grads_tsne


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/tmp/proxy_data')
    parser.add_argument('--output_dir', type=str, default='/tmp/proxy_vis')
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--perplexity', type=int, default=30)
    args = parser.parse_args()

    data = load_data(args.save_dir)
    grads = data['grads_flat'].detach().numpy()
    group_ids = data['group_ids'].detach().numpy()
    advantages = data.get('advantages')
    if advantages is not None:
        advantages = advantages.detach().numpy()
    rewards = data.get('rewards')
    if rewards is not None:
        rewards = rewards.detach().numpy()

    pca_visualization(grads, group_ids, advantages, rewards, args.output_dir)
    if args.tsne:
        tsne_visualization(grads, group_ids, advantages, args.output_dir, args.perplexity)


if __name__ == '__main__':
    main()
