#!/usr/bin/env python3
"""
RAE 评估脚本 - Phase 2

评估 RAE 性能:
1. 重构误差 (MSE)
2. Latent 空间分析 (PCA, t-SNE)
3. 属性分布可分性
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from model import RAE
from dataset import get_dataloaders


@torch.no_grad()
def extract_latents(model, loader, device):
    """提取所有 latent vectors"""
    model.eval()
    all_latents = []
    all_embeddings = []
    all_recons = []
    
    for batch in tqdm(loader, desc="Extracting latents"):
        batch = batch.to(device)
        recon, z = model(batch)
        
        all_latents.append(z.cpu().numpy())
        all_embeddings.append(batch.cpu().numpy())
        all_recons.append(recon.cpu().numpy())
    
    return {
        "latents": np.vstack(all_latents),
        "embeddings": np.vstack(all_embeddings),
        "recons": np.vstack(all_recons),
    }


def compute_reconstruction_metrics(embeddings, recons):
    """计算重构指标"""
    mse = np.mean((embeddings - recons) ** 2)
    mae = np.mean(np.abs(embeddings - recons))
    
    # Per-sample MSE
    per_sample_mse = np.mean((embeddings - recons) ** 2, axis=1)
    
    return {
        "mse": mse,
        "mae": mae,
        "per_sample_mse": per_sample_mse,
    }


def visualize_latent_space(latents, metadata, output_dir):
    """可视化 latent 空间"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    # t-SNE (sample if too large)
    if len(latents) > 5000:
        print("Sampling for t-SNE...")
        indices = np.random.choice(len(latents), 5000, replace=False)
        latents_sample = latents[indices]
        metadata_sample = metadata.iloc[indices]
    else:
        latents_sample = latents
        metadata_sample = metadata
        indices = np.arange(len(latents))
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    latents_tsne = tsne.fit_transform(latents_sample)
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PCA - is_amp
    scatter = axes[0, 0].scatter(
        latents_pca[:, 0], latents_pca[:, 1],
        c=metadata["is_amp"], cmap="coolwarm", s=1, alpha=0.5
    )
    axes[0, 0].set_title("PCA - AMP vs Non-AMP")
    axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # PCA - charge
    scatter = axes[0, 1].scatter(
        latents_pca[:, 0], latents_pca[:, 1],
        c=metadata["charge"], cmap="viridis", s=1, alpha=0.5
    )
    axes[0, 1].set_title("PCA - Charge")
    axes[0, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # PCA - hydrophobicity
    scatter = axes[0, 2].scatter(
        latents_pca[:, 0], latents_pca[:, 1],
        c=metadata["hydrophobicity"], cmap="RdYlGn", s=1, alpha=0.5
    )
    axes[0, 2].set_title("PCA - Hydrophobicity")
    axes[0, 2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0, 2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(scatter, ax=axes[0, 2])
    
    # t-SNE - is_amp
    scatter = axes[1, 0].scatter(
        latents_tsne[:, 0], latents_tsne[:, 1],
        c=metadata_sample["is_amp"], cmap="coolwarm", s=1, alpha=0.5
    )
    axes[1, 0].set_title("t-SNE - AMP vs Non-AMP")
    axes[1, 0].set_xlabel("t-SNE 1")
    axes[1, 0].set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # t-SNE - charge
    scatter = axes[1, 1].scatter(
        latents_tsne[:, 0], latents_tsne[:, 1],
        c=metadata_sample["charge"], cmap="viridis", s=1, alpha=0.5
    )
    axes[1, 1].set_title("t-SNE - Charge")
    axes[1, 1].set_xlabel("t-SNE 1")
    axes[1, 1].set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=axes[1, 1])
    
    # t-SNE - hydrophobicity
    scatter = axes[1, 2].scatter(
        latents_tsne[:, 0], latents_tsne[:, 1],
        c=metadata_sample["hydrophobicity"], cmap="RdYlGn", s=1, alpha=0.5
    )
    axes[1, 2].set_title("t-SNE - Hydrophobicity")
    axes[1, 2].set_xlabel("t-SNE 1")
    axes[1, 2].set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_dir / "latent_space.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {output_dir / 'latent_space.png'}")
    
    # 保存降维结果
    np.savez(
        output_dir / "latent_projections.npz",
        pca=latents_pca,
        tsne=latents_tsne,
        tsne_indices=indices,
    )


def main(args):
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading model...")
    model = RAE(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    
    # 加载数据
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.npz_path,
        args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # 加载 metadata
    metadata_path = Path(args.npz_path).parent / "metadata_with_labels.csv"
    metadata = pd.read_csv(metadata_path)
    
    # 提取 latents
    splits = {
        "train": (train_loader, np.load(Path(args.splits_dir) / "train_indices.npy")),
        "val": (val_loader, np.load(Path(args.splits_dir) / "val_indices.npy")),
        "test": (test_loader, np.load(Path(args.splits_dir) / "test_indices.npy")),
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for split_name, (loader, indices) in splits.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {split_name.upper()} set")
        print(f"{'='*60}")
        
        # 提取
        data = extract_latents(model, loader, device)
        
        # 重构指标
        metrics = compute_reconstruction_metrics(data["embeddings"], data["recons"])
        print(f"\nReconstruction Metrics:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  MSE std: {metrics['per_sample_mse'].std():.6f}")
        
        # 保存 latents
        np.savez(
            output_dir / f"{split_name}_latents.npz",
            latents=data["latents"],
            embeddings=data["embeddings"],
            recons=data["recons"],
        )
        print(f"Saved latents to {output_dir / f'{split_name}_latents.npz'}")
        
        # 可视化 (只对 test set)
        if split_name == "test":
            split_metadata = metadata.iloc[indices]
            visualize_latent_space(
                data["latents"],
                split_metadata,
                output_dir / "visualizations"
            )
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument("--npz-path", type=str, default="data/embeddings/esm3_embeddings.npz")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--checkpoint", type=str, required=True)
    
    # 模型
    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    
    # 其他
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="outputs/rae_eval")
    
    args = parser.parse_args()
    main(args)


