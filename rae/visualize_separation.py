#!/usr/bin/env python3
"""
改进的可视化 - 专门分析 AMP vs Non-AMP 分离度

针对数据不平衡问题：
- 下采样 AMP，平衡显示
- 计算分离度指标
- 训练简单分类器评估可分性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path


def visualize_balanced_tsne(latents, labels, output_path):
    """平衡采样后的 t-SNE 可视化"""
    
    # 找出 AMP 和 Non-AMP 索引
    amp_idx = np.where(labels == 1)[0]
    nonamp_idx = np.where(labels == 0)[0]
    
    print(f"Original counts - AMP: {len(amp_idx)}, Non-AMP: {len(nonamp_idx)}")
    
    # 下采样 AMP 到和 Non-AMP 相同数量
    np.random.seed(42)
    amp_sampled = np.random.choice(amp_idx, len(nonamp_idx), replace=False)
    
    # 合并
    balanced_idx = np.concatenate([amp_sampled, nonamp_idx])
    np.random.shuffle(balanced_idx)
    
    balanced_latents = latents[balanced_idx]
    balanced_labels = labels[balanced_idx]
    
    print(f"Balanced counts - AMP: {(balanced_labels==1).sum()}, Non-AMP: {(balanced_labels==0).sum()}")
    
    # t-SNE
    print("Running t-SNE on balanced data...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_tsne = tsne.fit_transform(balanced_latents)
    
    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 先画 AMP (背景)
    amp_mask = balanced_labels == 1
    ax.scatter(
        latents_tsne[amp_mask, 0], 
        latents_tsne[amp_mask, 1],
        c='red', s=20, alpha=0.3, label='AMP', edgecolors='none'
    )
    
    # 再画 Non-AMP (前景，更明显)
    nonamp_mask = balanced_labels == 0
    ax.scatter(
        latents_tsne[nonamp_mask, 0], 
        latents_tsne[nonamp_mask, 1],
        c='blue', s=50, alpha=0.8, label='Non-AMP', edgecolors='black', linewidths=0.5
    )
    
    ax.set_title(f"t-SNE - Balanced Sampling\n(AMP: {amp_mask.sum()}, Non-AMP: {nonamp_mask.sum()})", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc='best', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")


def compute_separation_metrics(latents, labels):
    """计算分离度指标"""
    
    # 1. 中心距离
    amp_latents = latents[labels == 1]
    nonamp_latents = latents[labels == 0]
    
    amp_center = amp_latents.mean(axis=0)
    nonamp_center = nonamp_latents.mean(axis=0)
    
    center_dist = np.linalg.norm(amp_center - nonamp_center)
    
    # 2. 类内方差
    amp_var = amp_latents.var(axis=0).mean()
    nonamp_var = nonamp_latents.var(axis=0).mean()
    
    # 3. Fisher 判别比 (类间距离 / 类内方差)
    fisher_ratio = center_dist / (np.sqrt(amp_var) + np.sqrt(nonamp_var))
    
    print(f"\n=== Separation Metrics ===")
    print(f"Center distance: {center_dist:.4f}")
    print(f"AMP variance: {amp_var:.4f}")
    print(f"Non-AMP variance: {nonamp_var:.4f}")
    print(f"Fisher ratio: {fisher_ratio:.4f}")
    
    return {
        "center_dist": center_dist,
        "amp_var": amp_var,
        "nonamp_var": nonamp_var,
        "fisher_ratio": fisher_ratio,
    }


def evaluate_classifier(train_latents, train_labels, test_latents, test_labels):
    """训练简单分类器评估可分性"""
    
    print(f"\n=== Classifier Evaluation ===")
    print(f"Train: AMP={train_labels.sum()}, Non-AMP={(train_labels==0).sum()}")
    print(f"Test: AMP={test_labels.sum()}, Non-AMP={(test_labels==0).sum()}")
    
    # 训练 Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(train_latents, train_labels)
    
    # 预测
    train_pred = clf.predict(train_latents)
    test_pred = clf.predict(test_latents)
    
    train_proba = clf.predict_proba(train_latents)[:, 1]
    test_proba = clf.predict_proba(test_latents)[:, 1]
    
    # 评估
    print("\n--- Train Set ---")
    print(classification_report(train_labels, train_pred, target_names=['Non-AMP', 'AMP'], digits=4))
    print(f"ROC-AUC: {roc_auc_score(train_labels, train_proba):.4f}")
    
    print("\n--- Test Set ---")
    print(classification_report(test_labels, test_pred, target_names=['Non-AMP', 'AMP'], digits=4))
    print(f"ROC-AUC: {roc_auc_score(test_labels, test_proba):.4f}")
    
    return clf


def main():
    output_dir = Path("outputs/rae_v1_eval/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载数据
    print("Loading data...")
    train_data = np.load("outputs/rae_v1_eval/train_latents.npz")
    test_data = np.load("outputs/rae_v1_eval/test_latents.npz")
    
    metadata = pd.read_csv("data/embeddings/metadata_with_labels.csv")
    train_idx = np.load("data/splits/train_indices.npy")
    test_idx = np.load("data/splits/test_indices.npy")
    
    train_latents = train_data["latents"]
    test_latents = test_data["latents"]
    
    train_labels = metadata.iloc[train_idx]["is_amp"].values
    test_labels = metadata.iloc[test_idx]["is_amp"].values
    
    # 1. 平衡采样 t-SNE 可视化
    print("\n" + "="*60)
    print("Visualizing with balanced sampling...")
    print("="*60)
    visualize_balanced_tsne(
        test_latents, 
        test_labels, 
        output_dir / "tsne_balanced.png"
    )
    
    # 2. 分离度指标
    print("\n" + "="*60)
    print("Computing separation metrics...")
    print("="*60)
    metrics = compute_separation_metrics(test_latents, test_labels)
    
    # 3. 分类器评估
    print("\n" + "="*60)
    print("Evaluating classifier...")
    print("="*60)
    clf = evaluate_classifier(train_latents, train_labels, test_latents, test_labels)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()

