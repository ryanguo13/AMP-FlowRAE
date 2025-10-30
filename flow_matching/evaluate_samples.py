#!/usr/bin/env python3
"""
评估 Flow Matching 生成的 latent vectors 质量

1. 解码回 ESM-3 embeddings
2. 检查条件控制效果（charge, hydrophobicity, length）
3. 可视化生成的 latent space 分布
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import sys

# 添加 rae 模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "rae"))
from model import RAE


def load_samples(samples_dir: Path):
    """加载采样的 latent vectors 和对应条件"""
    latents = np.load(samples_dir / "sampled_latents.npy")
    conditions = pd.read_csv(samples_dir / "sampled_conditions.csv")
    
    print(f"Loaded {len(latents)} sampled latent vectors")
    print(f"Latent shape: {latents.shape}")
    print(f"Conditions: {list(conditions.columns)}")
    
    return latents, conditions


def decode_latents(latents, rae_checkpoint, device):
    """使用 RAE 解码器将 latent vectors 解码回 ESM-3 embeddings"""
    print(f"\nLoading RAE decoder from {rae_checkpoint}...")
    
    ckpt = torch.load(rae_checkpoint, map_location=device)
    
    # 从 checkpoint 推断维度
    encoder_weight = ckpt["model_state_dict"]["encoder.0.weight"]
    esm_dim = encoder_weight.shape[1]  # input dim
    latent_dim = latents.shape[1]
    
    print(f"ESM dim: {esm_dim}, Latent dim: {latent_dim}")
    
    rae = RAE(esm_dim=esm_dim, latent_dim=latent_dim).to(device)
    rae.load_state_dict(ckpt["model_state_dict"])
    rae.eval()
    
    # 解码
    print(f"Decoding {len(latents)} latent vectors...")
    with torch.no_grad():
        z = torch.from_numpy(latents).float().to(device)
        x_recon = rae.decoder(z)
        embeddings = x_recon.cpu().numpy()
    
    print(f"Decoded embeddings shape: {embeddings.shape}")
    return embeddings


def compute_physicochemical_props(sequences):
    """
    计算序列的理化性质（如果有序列）
    这里是占位符 - 实际使用时需要先将 embedding 解码回序列
    """
    # TODO: 在 Phase 4 实现从 ESM-3 embedding 反向解码到序列后，计算真实的理化性质
    pass


def analyze_condition_correlation(latents, conditions):
    """
    分析条件与 latent space 的相关性
    
    使用线性回归检查每个 latent 维度与条件的相关性
    """
    print("\n=== Condition-Latent Correlation Analysis ===")
    
    correlations = {}
    
    for cond_name in ["charge", "hydrophobicity", "length"]:
        cond_values = conditions[cond_name].values
        
        # 对每个 latent 维度计算与该条件的相关性
        dim_correlations = []
        for dim in range(latents.shape[1]):
            latent_dim = latents[:, dim]
            corr, pval = pearsonr(latent_dim, cond_values)
            dim_correlations.append(abs(corr))  # 使用绝对值
        
        max_corr = max(dim_correlations)
        avg_corr = np.mean(dim_correlations)
        
        correlations[cond_name] = {
            "max": max_corr,
            "avg": avg_corr,
            "dim_corrs": dim_correlations
        }
        
        print(f"{cond_name:15s}: max_corr={max_corr:.3f}, avg_corr={avg_corr:.3f}")
    
    return correlations


def plot_latent_distribution(latents, conditions, output_dir):
    """可视化生成的 latent space 分布"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print("\n=== Visualizing Latent Distribution ===")
    
    # PCA 降维到 2D
    print("Running PCA...")
    pca = PCA(n_components=2)
    latents_2d_pca = pca.fit_transform(latents)
    
    # t-SNE 降维到 2D
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)-1))
    latents_2d_tsne = tsne.fit_transform(latents)
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, cond_name in enumerate(["charge", "hydrophobicity", "length"]):
        cond_values = conditions[cond_name].values
        
        # PCA
        ax = axes[0, i]
        scatter = ax.scatter(
            latents_2d_pca[:, 0],
            latents_2d_pca[:, 1],
            c=cond_values,
            cmap="viridis",
            alpha=0.6,
            s=20
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"PCA colored by {cond_name}")
        plt.colorbar(scatter, ax=ax)
        
        # t-SNE
        ax = axes[1, i]
        scatter = ax.scatter(
            latents_2d_tsne[:, 0],
            latents_2d_tsne[:, 1],
            c=cond_values,
            cmap="viridis",
            alpha=0.6,
            s=20
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE colored by {cond_name}")
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plot_path = output_dir / "latent_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved plot to {plot_path}")
    plt.close()


def plot_condition_histograms(conditions, output_dir):
    """绘制生成的条件分布"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, cond_name in enumerate(["charge", "hydrophobicity", "length"]):
        ax = axes[i]
        ax.hist(conditions[cond_name], bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel(cond_name)
        ax.set_ylabel("Count")
        ax.set_title(f"{cond_name} distribution")
        ax.axvline(conditions[cond_name].mean(), color="red", linestyle="--", 
                   label=f"Mean: {conditions[cond_name].mean():.2f}")
        ax.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "condition_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved plot to {plot_path}")
    plt.close()


def main(args):
    # 设备
    if torch.backends.mps.is_available() and not args.cpu:
        device = "mps"
    elif torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    samples_dir = Path(args.samples_dir)
    output_dir = samples_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    # 加载采样结果
    latents, conditions = load_samples(samples_dir)
    
    # 统计信息
    print(f"\n=== Latent Statistics ===")
    print(f"Mean: {latents.mean():.4f}")
    print(f"Std: {latents.std():.4f}")
    print(f"Min: {latents.min():.4f}")
    print(f"Max: {latents.max():.4f}")
    
    # 条件与 latent 的相关性
    correlations = analyze_condition_correlation(latents, conditions)
    
    # 保存相关性结果
    corr_results = pd.DataFrame({
        cond_name: {
            "max_correlation": corr_data["max"],
            "avg_correlation": corr_data["avg"]
        }
        for cond_name, corr_data in correlations.items()
    }).T
    
    corr_path = output_dir / "condition_correlations.csv"
    corr_results.to_csv(corr_path)
    print(f"\n✅ Saved correlations to {corr_path}")
    
    # 可视化
    print("\n=== Generating Visualizations ===")
    plot_condition_histograms(conditions, output_dir)
    plot_latent_distribution(latents, conditions, output_dir)
    
    # 如果提供了 RAE checkpoint，解码回 embedding
    if args.rae_checkpoint:
        embeddings = decode_latents(latents, args.rae_checkpoint, device)
        
        # 保存解码的 embeddings
        emb_path = output_dir / "decoded_embeddings.npy"
        np.save(emb_path, embeddings)
        print(f"✅ Saved decoded embeddings to {emb_path}")
        
        # 统计
        print(f"\n=== Decoded Embedding Statistics ===")
        print(f"Mean: {embeddings.mean():.4f}")
        print(f"Std: {embeddings.std():.4f}")
        print(f"Min: {embeddings.min():.4f}")
        print(f"Max: {embeddings.max():.4f}")
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sampled latent vectors")
    
    parser.add_argument("--samples-dir", type=str, default="outputs/samples",
                        help="Directory containing sampled_latents.npy and sampled_conditions.csv")
    parser.add_argument("--rae-checkpoint", type=str, default=None,
                        help="Path to RAE checkpoint (optional, for decoding)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    
    args = parser.parse_args()
    main(args)


