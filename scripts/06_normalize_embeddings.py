#!/usr/bin/env python3
"""
归一化 embeddings - 修复 Phase 1 的数据问题

ESM-3 embeddings 未归一化，范围 [-11970, 1359]
这会导致 RAE 训练失败

解决方案：标准化到 zero mean, unit variance
"""

import numpy as np
from pathlib import Path


def main():
    data_dir = Path("data/embeddings")
    input_path = data_dir / "esm3_embeddings.npz"
    output_path = data_dir / "esm3_embeddings_normalized.npz"
    
    # 加载
    print(f"Loading {input_path}...")
    data = np.load(input_path)
    embeddings = data["embeddings"]
    sequences = data["sequences"]
    indices = data["indices"]
    
    print(f"Original stats:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    
    # 标准化 (zero mean, unit variance)
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    embeddings_norm = (embeddings - mean) / (std + 1e-8)
    
    print(f"\nNormalized stats:")
    print(f"  Mean: {embeddings_norm.mean():.4f}")
    print(f"  Std: {embeddings_norm.std():.4f}")
    print(f"  Min: {embeddings_norm.min():.4f}")
    print(f"  Max: {embeddings_norm.max():.4f}")
    
    # 保存
    np.savez(
        output_path,
        embeddings=embeddings_norm.astype(np.float32),
        sequences=sequences,
        indices=indices,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    
    print(f"\n✅ Saved to {output_path}")
    print(f"   (mean/std saved for inverse transform)")


if __name__ == "__main__":
    main()


