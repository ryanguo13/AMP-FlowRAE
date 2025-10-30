#!/usr/bin/env python3
"""
分析生成的AMP序列质量

1. 计算理化性质
2. 活性预测
3. 与训练集对比
4. 生成可视化
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.sequence_properties import predict_amp_activity


def analyze_sequences(csv_path: str, output_dir: str):
    """分析生成的序列"""
    
    print(f"Loading sequences from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total sequences: {len(df)}")
    print(f"Unique sequences: {df['sequence'].nunique()}")
    
    # 计算所有序列的属性
    print("\nComputing properties for all sequences...")
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        seq = row['sequence']
        props = predict_amp_activity(seq)
        
        result = {
            'query_idx': row['query_idx'],
            'sequence': seq,
            'nn_distance': row['nn_distance'],
            'nn_source': row['nn_source'],
            **props
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # 保存完整结果
    output_path = Path(output_dir) / "generated_amps_analyzed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved analyzed results to {output_path}")
    
    # 统计分析
    print(f"\n{'='*60}")
    print("=== GENERATION QUALITY ANALYSIS ===")
    print(f"{'='*60}")
    
    print(f"\n📊 基本统计:")
    print(f"Total sequences: {len(results_df)}")
    print(f"Unique sequences: {results_df['sequence'].nunique()}")
    print(f"Average length: {results_df['length'].mean():.1f} ± {results_df['length'].std():.1f}")
    print(f"Length range: [{results_df['length'].min()}, {results_df['length'].max()}]")
    
    print(f"\n🔬 理化性质:")
    print(f"Charge: {results_df['charge'].mean():.2f} ± {results_df['charge'].std():.2f}")
    print(f"  Range: [{results_df['charge'].min():.1f}, {results_df['charge'].max():.1f}]")
    print(f"Hydrophobicity: {results_df['hydrophobicity'].mean():.2f} ± {results_df['hydrophobicity'].std():.2f}")
    print(f"  Range: [{results_df['hydrophobicity'].min():.2f}, {results_df['hydrophobicity'].max():.2f}]")
    print(f"Boman Index: {results_df['boman_index'].mean():.2f} ± {results_df['boman_index'].std():.2f}")
    print(f"Aliphatic Index: {results_df['aliphatic_index'].mean():.1f} ± {results_df['aliphatic_index'].std():.1f}")
    
    print(f"\n⭐ 活性预测:")
    print(f"Average activity score: {results_df['activity_score'].mean():.3f} ± {results_df['activity_score'].std():.3f}")
    
    high_activity = (results_df['activity_score'] > 0.8).sum()
    medium_activity = ((results_df['activity_score'] > 0.6) & (results_df['activity_score'] <= 0.8)).sum()
    low_activity = (results_df['activity_score'] <= 0.6).sum()
    
    print(f"High activity (>0.8):   {high_activity} ({high_activity/len(results_df)*100:.1f}%)")
    print(f"Medium activity (0.6-0.8): {medium_activity} ({medium_activity/len(results_df)*100:.1f}%)")
    print(f"Low activity (<0.6):    {low_activity} ({low_activity/len(results_df)*100:.1f}%)")
    
    print(f"\n🆕 新颖性:")
    print(f"Average NN distance: {results_df['nn_distance'].mean():.4f} ± {results_df['nn_distance'].std():.4f}")
    high_novelty = (results_df['nn_distance'] > 0.5).sum()
    print(f"High novelty (>0.5): {high_novelty} ({high_novelty/len(results_df)*100:.1f}%)")
    
    # 可视化
    print(f"\n📈 Generating visualizations...")
    create_visualizations(results_df, output_dir)
    
    # Top sequences
    print(f"\n🏆 Top 10 Generated AMPs (by activity score):")
    top10 = results_df.nlargest(10, 'activity_score')[
        ['sequence', 'activity_score', 'charge', 'hydrophobicity', 'length', 'nn_distance']
    ]
    print(top10.to_string(index=False))
    
    return results_df


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """生成可视化"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 属性分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    properties = [
        ('charge', 'Charge'),
        ('hydrophobicity', 'Hydrophobicity'),
        ('boman_index', 'Boman Index'),
        ('aliphatic_index', 'Aliphatic Index'),
        ('length', 'Length'),
        ('activity_score', 'Activity Score')
    ]
    
    for ax, (prop, label) in zip(axes.flat, properties):
        ax.hist(df[prop], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(df[prop].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[prop].mean():.2f}')
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{label} Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "property_distributions.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved property_distributions.png")
    plt.close()
    
    # 2. Activity score vs properties
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    scatter_props = [
        ('charge', 'Charge'),
        ('hydrophobicity', 'Hydrophobicity'),
        ('length', 'Length'),
        ('nn_distance', 'NN Distance (Novelty)')
    ]
    
    for ax, (prop, label) in zip(axes.flat, scatter_props):
        scatter = ax.scatter(df[prop], df['activity_score'], 
                            c=df['activity_score'], cmap='RdYlGn',
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Activity Score', fontsize=12)
        ax.set_title(f'Activity vs {label}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 相关系数
        corr = df[prop].corr(df['activity_score'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
        
        plt.colorbar(scatter, ax=ax, label='Activity Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / "activity_correlations.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved activity_correlations.png")
    plt.close()
    
    # 3. 来源分布
    fig, ax = plt.subplots(figsize=(10, 6))
    source_counts = df['nn_source'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
    wedges, texts, autotexts = ax.pie(source_counts, labels=source_counts.index, 
                                        autopct='%1.1f%%', colors=colors, 
                                        startangle=90, textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Nearest Neighbor Source Distribution', fontsize=14, fontweight='bold')
    plt.savefig(output_dir / "source_distribution.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved source_distribution.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze generated AMP sequences")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to decoded_sequences.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    results = analyze_sequences(args.input, args.output_dir)
    
    print(f"\n{'='*60}")
    print("✅ Analysis complete!")
    print(f"Results saved to {args.output_dir}/")
    print(f"{'='*60}")

