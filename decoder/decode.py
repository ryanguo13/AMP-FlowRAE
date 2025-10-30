#!/usr/bin/env python3
"""
完整解码流程：Latent → Embedding → Sequence

Pipeline:
1. Latent vectors (64)
2. → RAE Decoder → ESM-3 embeddings (1536)
3. → Nearest Neighbor Search → AMP sequences
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# 添加 rae 模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "rae"))
from model import RAE
from nearest_neighbor import load_decoder


def decode_latents_to_embeddings(
    latents: np.ndarray,
    rae_checkpoint: str,
    device: str = "cpu"
) -> np.ndarray:
    """
    Step 1: 使用 RAE Decoder 将 latent vectors 解码为 ESM-3 embeddings
    
    Args:
        latents: (N, 64) latent vectors
        rae_checkpoint: RAE 模型路径
        device: 计算设备
    
    Returns:
        embeddings: (N, 1536) ESM-3 embeddings
    """
    print(f"\n=== Step 1: Decoding latents to embeddings ===")
    print(f"Loading RAE from {rae_checkpoint}...")
    
    # 加载模型
    ckpt = torch.load(rae_checkpoint, map_location=device)
    
    # 推断维度
    encoder_weight = ckpt["model_state_dict"]["encoder.encoder.0.weight"]
    esm_dim = encoder_weight.shape[1]
    latent_dim = latents.shape[1]
    
    print(f"ESM dim: {esm_dim}, Latent dim: {latent_dim}")
    
    rae = RAE(input_dim=esm_dim, latent_dim=latent_dim).to(device)
    rae.load_state_dict(ckpt["model_state_dict"])
    rae.eval()
    
    # 解码
    print(f"Decoding {len(latents)} latent vectors...")
    with torch.no_grad():
        z = torch.from_numpy(latents).float().to(device)
        x_recon = rae.decoder(z)
        embeddings = x_recon.cpu().numpy()
    
    print(f"✅ Decoded to embeddings: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
    
    return embeddings


def decode_embeddings_to_sequences(
    embeddings: np.ndarray,
    reference_embeddings_path: str,
    metadata_path: str,
    splits_path: str,
    n_neighbors: int = 5,
    metric: str = "cosine"
) -> pd.DataFrame:
    """
    Step 2: 使用最近邻搜索将 embeddings 解码为序列
    
    Args:
        embeddings: (N, 1536) ESM-3 embeddings
        reference_embeddings_path: 训练集 embeddings 路径
        metadata_path: metadata.csv 路径
        splits_path: splits 目录
        n_neighbors: k-NN 的 k
        metric: 距离度量
    
    Returns:
        DataFrame with decoded sequences and metadata
    """
    print(f"\n=== Step 2: Searching nearest neighbors ===")
    
    # 加载 decoder
    decoder = load_decoder(
        embeddings_path=reference_embeddings_path,
        metadata_path=metadata_path,
        splits_path=splits_path,
        split="train",  # 只在训练集中搜索
        n_neighbors=n_neighbors,
        metric=metric
    )
    
    # 解码
    print(f"\nDecoding {len(embeddings)} embeddings to sequences...")
    results = decoder.decode(
        embeddings,
        return_distances=True,
        return_all_neighbors=True
    )
    
    # 转换为 DataFrame
    records = []
    for i, result in enumerate(results):
        nn = result['neighbors'][0]  # 最近的邻居
        record = {
            'query_idx': i,
            'sequence': nn['sequence'],
            'nn_distance': nn['distance'],
            'nn_idx': nn['idx'],
            'nn_id': nn['id'],
            'nn_source': nn['source'],
            'length': nn['length'],
            # Top-5 近邻距离（用于评估新颖性）
            'nn_distances_top5': ','.join([f"{n['distance']:.4f}" for n in result['neighbors'][:5]])
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    print(f"✅ Decoded {len(df)} sequences")
    print(f"\n=== Decoding Statistics ===")
    print(f"Average NN distance: {df['nn_distance'].mean():.4f} ± {df['nn_distance'].std():.4f}")
    print(f"Min distance: {df['nn_distance'].min():.4f}")
    print(f"Max distance: {df['nn_distance'].max():.4f}")
    
    return df


def main(args):
    # 设备
    if torch.backends.mps.is_available() and not args.cpu:
        device = "mps"
    elif torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # 加载 latent vectors
    print(f"\nLoading latent vectors from {args.latents}...")
    latents = np.load(args.latents)
    print(f"Loaded {len(latents)} latent vectors, shape: {latents.shape}")
    
    # 加载对应的条件（如果有）
    conditions = None
    if args.conditions:
        conditions = pd.read_csv(args.conditions)
        print(f"Loaded conditions: {list(conditions.columns)}")
    
    # Step 1: Latent → Embedding
    embeddings = decode_latents_to_embeddings(
        latents=latents,
        rae_checkpoint=args.rae_checkpoint,
        device=device
    )
    
    # Step 2: Embedding → Sequence
    results_df = decode_embeddings_to_sequences(
        embeddings=embeddings,
        reference_embeddings_path=args.reference_embeddings,
        metadata_path=args.metadata,
        splits_path=args.splits_dir,
        n_neighbors=args.n_neighbors,
        metric=args.metric
    )
    
    # 合并条件信息（如果有）
    if conditions is not None:
        results_df = pd.concat([conditions, results_df], axis=1)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Results saved to {output_path}")
    print(f"\n=== Sample Results (first 3) ===")
    print(results_df[['sequence', 'nn_distance', 'length', 'nn_source']].head(3).to_string())
    
    # 额外统计
    print(f"\n=== Novelty Analysis ===")
    high_novelty = (results_df['nn_distance'] > 0.1).sum()
    medium_novelty = ((results_df['nn_distance'] > 0.05) & (results_df['nn_distance'] <= 0.1)).sum()
    low_novelty = (results_df['nn_distance'] <= 0.05).sum()
    
    print(f"High novelty (dist > 0.10): {high_novelty} ({high_novelty/len(results_df)*100:.1f}%)")
    print(f"Medium novelty (0.05-0.10): {medium_novelty} ({medium_novelty/len(results_df)*100:.1f}%)")
    print(f"Low novelty (dist < 0.05):  {low_novelty} ({low_novelty/len(results_df)*100:.1f}%)")
    
    print(f"\n🎉 Decoding complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode latent vectors to AMP sequences")
    
    # 输入
    parser.add_argument("--latents", type=str, required=True,
                        help="Path to latent vectors (.npy)")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Path to conditions CSV (optional)")
    
    # 模型
    parser.add_argument("--rae-checkpoint", type=str, required=True,
                        help="Path to RAE checkpoint")
    
    # 参考数据（用于最近邻搜索）
    parser.add_argument("--reference-embeddings", type=str,
                        default="data/embeddings/esm3_embeddings_normalized.npz",
                        help="Path to reference embeddings")
    parser.add_argument("--metadata", type=str,
                        default="data/embeddings/metadata.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--splits-dir", type=str,
                        default="data/splits",
                        help="Path to splits directory")
    
    # 最近邻参数
    parser.add_argument("--n-neighbors", type=int, default=5,
                        help="Number of nearest neighbors to consider")
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "manhattan"],
                        help="Distance metric")
    
    # 输出
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    
    args = parser.parse_args()
    main(args)

