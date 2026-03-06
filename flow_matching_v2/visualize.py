#!/usr/bin/env python3
"""
Visualization and Analysis Script

Generates comprehensive visualizations for tracking progress:
1. Training curves (loss)
2. Latent space distribution
3. Condition correlation analysis
4. Evaluation metrics comparison (before/after)
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_training_curves(log_dir, output_dir):
    """Plot training and validation loss curves"""
    print("📊 Plotting training curves...")

    # Try to read tensorboard logs or training history
    log_path = Path(log_dir) / "logs"

    # For now, create a placeholder - in practice you'd use tensorboard reader
    print("   (TensorBoard logs require tensorboard reader)")

    # Check if there's a history file
    history_path = Path(log_dir) / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.get("train_loss", []), label="Train Loss")
        ax.plot(history.get("val_loss", []), label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close()
        print(f"   ✅ Saved to {output_dir / 'training_curves.png'}")


def plot_latent_distribution(samples_dir, output_dir):
    """Visualize latent space distribution with PCA/t-SNE"""
    print("📊 Plotting latent space distribution...")

    latents_path = Path(samples_dir) / "sampled_latents.npy"
    conditions_path = Path(samples_dir) / "sampled_conditions.csv"

    if not latents_path.exists() or not conditions_path.exists():
        print("   ⚠️  Missing required files")
        return

    latents = np.load(latents_path)
    conditions = pd.read_csv(conditions_path)

    # PCA
    print("   Running PCA...")
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)

    # t-SNE
    print("   Running t-SNE...")
    perp = min(30, len(latents) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    latents_tsne = tsne.fit_transform(latents)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, cond_name in enumerate(["charge", "hydrophobicity", "length"]):
        cond_values = conditions[cond_name].values

        # PCA
        ax = axes[0, i]
        scatter = ax.scatter(
            latents_pca[:, 0], latents_pca[:, 1], c=cond_values, cmap="viridis", alpha=0.6, s=20
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title(f"PCA colored by {cond_name}")
        plt.colorbar(scatter, ax=ax)

        # t-SNE
        ax = axes[1, i]
        scatter = ax.scatter(
            latents_tsne[:, 0], latents_tsne[:, 1], c=cond_values, cmap="viridis", alpha=0.6, s=20
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE colored by {cond_name}")
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / "latent_distribution.png", dpi=150)
    plt.close()
    print(f"   ✅ Saved to {output_dir / 'latent_distribution.png'}")


def plot_condition_correlation(samples_dir, output_dir):
    """Plot condition correlation with actual generated properties"""
    print("📊 Plotting condition correlation...")

    conditions_path = Path(samples_dir) / "sampled_conditions.csv"
    properties_path = Path(samples_dir, "evaluation", "generated_properties.csv")

    if not conditions_path.exists():
        print("   ⚠️  Missing conditions file")
        return

    conditions = pd.read_csv(conditions_path)

    # If we have generated properties, compare target vs actual
    if properties_path.exists():
        props = pd.read_csv(properties_path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, cond_name in enumerate(["charge", "hydrophobicity", "length"]):
            ax = axes[i]
            target = conditions[cond_name].values[: len(props)]
            actual = props[cond_name].values

            ax.scatter(target, actual, alpha=0.5, s=20)

            # Add perfect correlation line
            min_val = min(target.min(), actual.min())
            max_val = max(target.max(), actual.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")

            # Calculate correlation
            corr, _ = pearsonr(target, actual)
            ax.set_xlabel(f"Target {cond_name}")
            ax.set_ylabel(f"Actual {cond_name}")
            ax.set_title(f"{cond_name} Control\nCorrelation: {corr:.3f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "condition_correlation.png", dpi=150)
        plt.close()
        print(f"   ✅ Saved to {output_dir / 'condition_correlation.png'}")
    else:
        # Just plot condition distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, cond_name in enumerate(["charge", "hydrophobicity", "length"]):
            ax = axes[i]
            ax.hist(conditions[cond_name], bins=30, alpha=0.7, edgecolor="black")
            ax.set_xlabel(cond_name)
            ax.set_ylabel("Count")
            ax.set_title(f"{cond_name} Distribution")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "condition_distributions.png", dpi=150)
        plt.close()
        print(f"   ✅ Saved to {output_dir / 'condition_distributions.png'}")


def plot_evaluation_comparison(eval_dir, output_dir):
    """Compare evaluation metrics before/after improvements"""
    print("📊 Plotting evaluation comparison...")

    # Look for evaluation results
    eval_v1_path = Path("outputs/evaluation/evaluation_results.json")
    eval_v2_path = Path(eval_dir) / "evaluation_results.json"

    metrics = ["novelty_rate", "mean_identity", "diversity"]
    labels = ["Novelty Rate", "Mean Identity", "Internal Diversity"]

    v1_data = {}
    v2_data = {}

    if eval_v1_path.exists():
        with open(eval_v1_path) as f:
            v1_data = json.load(f)

    if eval_v2_path.exists():
        with open(eval_v2_path) as f:
            v2_data = json.load(f)

    if not v1_data and not v2_data:
        print("   ⚠️  No evaluation results found")
        return

    # Prepare data for plotting
    v1_values = []
    v2_values = []

    if v1_data:
        v1_values = [
            v1_data.get("novelty", {}).get("novelty_rate", 0) * 100,
            v1_data.get("novelty", {}).get("mean_identity", 0),
            v1_data.get("diversity", {}).get("mean_identity", 0),
        ]

    if v2_data:
        v2_values = [
            v2_data.get("novelty", {}).get("novelty_rate", 0) * 100,
            v2_data.get("novelty", {}).get("mean_identity", 0),
            v2_data.get("diversity", {}).get("mean_identity", 0),
        ]

    if not v1_values and not v2_values:
        print("   ⚠️  No metrics to compare")
        return

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    if v1_values:
        ax.bar(x - width / 2, v1_values, width, label="V1 (Before)", alpha=0.8)
    if v2_values:
        ax.bar(x + width / 2, v2_values, width, label="V2 (After)", alpha=0.8)

    ax.set_ylabel("Value")
    ax.set_title("Evaluation Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (v1, v2) in enumerate(zip(v1_values, v2_values)):
        if v1:
            ax.text(i - width / 2, v1 + 1, f"{v1:.1f}", ha="center", fontsize=9)
        if v2:
            ax.text(i + width / 2, v2 + 1, f"{v2:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_comparison.png", dpi=150)
    plt.close()
    print(f"   ✅ Saved to {output_dir / 'evaluation_comparison.png'}")


def plot_identity_distribution(eval_dir, output_dir):
    """Plot identity distribution histogram"""
    print("📊 Plotting identity distribution...")

    eval_v1_path = Path("outputs/evaluation/evaluation_results.json")
    eval_v2_path = Path(eval_dir) / "evaluation_results.json"

    v1_dist = None
    v2_dist = None

    if eval_v1_path.exists():
        with open(eval_v1_path) as f:
            v1_data = json.load(f)
            v1_dist = v1_data.get("novelty", {}).get("identity_distribution", {})

    if eval_v2_path.exists():
        with open(eval_v2_path) as f:
            v2_data = json.load(f)
            v2_dist = v2_data.get("novelty", {}).get("identity_distribution", {})

    if not v1_dist and not v2_dist:
        print("   ⚠️  No identity distribution data found")
        return

    # Prepare data
    ranges = ["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", ">70%"]

    v1_values = [v1_dist.get(r, 0) for r in ranges] if v1_dist else [0] * 7
    v2_values = [v2_dist.get(r, 0) for r in ranges] if v2_dist else [0] * 7

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ranges))
    width = 0.35

    if any(v1_values):
        ax.bar(x - width / 2, v1_values, width, label="V1 (Before)", alpha=0.8)
    if any(v2_values):
        ax.bar(x + width / 2, v2_values, width, label="V2 (After)", alpha=0.8)

    ax.set_xlabel("Sequence Identity Range")
    ax.set_ylabel("Percentage of Generated Sequences")
    ax.set_title("Sequence Identity Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(ranges)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "identity_distribution.png", dpi=150)
    plt.close()
    print(f"   ✅ Saved to {output_dir / 'identity_distribution.png'}")


def plot_summary_dashboard(eval_dir, output_dir):
    """Create a comprehensive summary dashboard"""
    print("📊 Creating summary dashboard...")

    eval_v1_path = Path("outputs/evaluation/evaluation_results.json")
    eval_v2_path = Path(eval_dir) / "evaluation_results.json"

    # Load data
    v1_data = {}
    v2_data = {}

    if eval_v1_path.exists():
        with open(eval_v1_path) as f:
            v1_data = json.load(f)

    if eval_v2_path.exists():
        with open(eval_v2_path) as f:
            v2_data = json.load(f)

    # Create dashboard
    fig = plt.figure(figsize=(16, 12))

    # 1. Key metrics comparison (top row)
    ax1 = fig.add_subplot(2, 2, 1)
    metrics = ["Novelty Rate (%)", "Internal Diversity (%)"]
    v1_vals = [
        v1_data.get("novelty", {}).get("novelty_rate", 0) * 100,
        v1_data.get("diversity", {}).get("mean_identity", 0),
    ]
    v2_vals = [
        v2_data.get("novelty", {}).get("novelty_rate", 0) * 100,
        v2_data.get("diversity", {}).get("mean_identity", 0),
    ]

    x = np.arange(len(metrics))
    width = 0.35
    if any(v1_vals):
        ax1.bar(x - width / 2, v1_vals, width, label="V1", alpha=0.8, color="coral")
    if any(v2_vals):
        ax1.bar(x + width / 2, v2_vals, width, label="V2", alpha=0.8, color="steelblue")
    ax1.set_ylabel("Value")
    ax1.set_title("Key Metrics Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Condition control accuracy (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    conds = ["charge", "hydrophobicity", "length"]
    v1_corr = [
        v1_data.get("condition_accuracy", {}).get("correlations", {}).get(c, 0) for c in conds
    ]
    v2_corr = [
        v2_data.get("condition_accuracy", {}).get("correlations", {}).get(c, 0) for c in conds
    ]

    x = np.arange(len(conds))
    if any(v1_corr):
        ax2.bar(x - width / 2, v1_corr, width, label="V1", alpha=0.8, color="coral")
    if any(v2_corr):
        ax2.bar(x + width / 2, v2_corr, width, label="V2", alpha=0.8, color="steelblue")
    ax2.set_ylabel("Correlation")
    ax2.set_title("Condition Control Accuracy")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conds)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # 3. Identity distribution (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    ranges = ["<20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", ">70%"]

    v1_dist = v1_data.get("novelty", {}).get("identity_distribution", {})
    v2_dist = v2_data.get("novelty", {}).get("identity_distribution", {})

    v1_dist_vals = [v1_dist.get(r, 0) for r in ranges]
    v2_dist_vals = [v2_dist.get(r, 0) for r in ranges]

    x = np.arange(len(ranges))
    if any(v1_dist_vals):
        ax3.bar(x - width / 2, v1_dist_vals, width, label="V1", alpha=0.8, color="coral")
    if any(v2_dist_vals):
        ax3.bar(x + width / 2, v2_dist_vals, width, label="V2", alpha=0.8, color="steelblue")
    ax3.set_xlabel("Identity Range")
    ax3.set_ylabel("Percentage")
    ax3.set_title("Sequence Identity Distribution")
    ax3.set_xticks(x)
    ax3.set_xticklabels(ranges, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Summary text (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    # Generate summary text
    summary_text = "=== SUMMARY ===\n\n"

    if v1_data and v2_data:
        novelty_improvement = (
            v2_data.get("novelty", {}).get("novelty_rate", 0)
            - v1_data.get("novelty", {}).get("novelty_rate", 0)
        ) * 100
        diversity_improvement = v2_data.get("diversity", {}).get("mean_identity", 0) - v1_data.get(
            "diversity", {}
        ).get("mean_identity", 0)

        summary_text += f"V1 (Before):\n"
        summary_text += (
            f"  - Novelty Rate: {v1_data.get('novelty', {}).get('novelty_rate', 0) * 100:.1f}%\n"
        )
        summary_text += f"  - Internal Diversity: {v1_data.get('diversity', {}).get('mean_identity', 0):.2f}%\n\n"

        summary_text += f"V2 (After):\n"
        summary_text += (
            f"  - Novelty Rate: {v2_data.get('novelty', {}).get('novelty_rate', 0) * 100:.1f}%\n"
        )
        summary_text += f"  - Internal Diversity: {v2_data.get('diversity', {}).get('mean_identity', 0):.2f}%\n\n"

        summary_text += f"Improvements:\n"
        summary_text += f"  - Novelty: {novelty_improvement:+.1f}%\n"
        summary_text += f"  - Diversity: {diversity_improvement:+.2f}%\n"
    else:
        summary_text += "Run evaluation on both V1 and V2\nto see comparison."

    ax4.text(
        0.1,
        0.9,
        summary_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved to {output_dir / 'summary_dashboard.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=str, default="outputs/samples_v2")
    parser.add_argument("--eval-dir", type=str, default="outputs/evaluation_v2")
    parser.add_argument("--log-dir", type=str, default="outputs/flow_v2/logs")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Visualization and Analysis")
    print("=" * 60)

    # Generate all plots
    plot_training_curves(args.log_dir, output_dir)
    plot_latent_distribution(args.samples_dir, output_dir)
    plot_condition_correlation(args.samples_dir, output_dir)
    plot_evaluation_comparison(args.eval_dir, output_dir)
    plot_identity_distribution(args.eval_dir, output_dir)
    plot_summary_dashboard(args.eval_dir, output_dir)

    print(f"\n✅ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
