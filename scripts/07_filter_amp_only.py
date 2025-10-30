#!/usr/bin/env python3
"""
过滤 AMP-only 数据 - 去掉 non-AMP

策略：
1. 过滤掉 non-AMP (456 条)
2. 保留 AMP (10,491 条)
3. 重新归一化（基于 AMP only）
4. 更新 splits
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    data_dir = Path("data")
    emb_dir = data_dir / "embeddings"
    splits_dir = data_dir / "splits"
    
    # 加载数据
    print("Loading data...")
    data = np.load(emb_dir / "esm3_embeddings.npz")
    embeddings = data["embeddings"]
    sequences = data["sequences"]
    indices = data["indices"]
    
    metadata = pd.read_csv(emb_dir / "metadata_with_labels.csv")
    
    print(f"Original data: {len(metadata)} sequences")
    print(metadata["source"].value_counts())
    
    # 过滤 AMP only
    amp_mask = metadata["source"] != "non-AMP"
    amp_indices = metadata.index[amp_mask].values
    
    amp_embeddings = embeddings[amp_mask]
    amp_sequences = sequences[amp_mask]
    amp_metadata = metadata[amp_mask].copy()
    
    print(f"\nFiltered to AMP only: {len(amp_metadata)} sequences")
    print(amp_metadata["source"].value_counts())
    
    # 重新索引 (重要！因为我们删除了一些行)
    amp_metadata = amp_metadata.reset_index(drop=True)
    amp_metadata["embedding_idx"] = np.arange(len(amp_metadata))
    
    # 统计原始 embeddings
    print(f"\n=== Original Embeddings Stats ===")
    print(f"Mean: {amp_embeddings.mean():.4f}")
    print(f"Std:  {amp_embeddings.std():.4f}")
    print(f"Min:  {amp_embeddings.min():.4f}")
    print(f"Max:  {amp_embeddings.max():.4f}")
    
    # 归一化（基于 AMP only）
    print(f"\n=== Normalizing (AMP only) ===")
    mean = amp_embeddings.mean(axis=0, keepdims=True)
    std = amp_embeddings.std(axis=0, keepdims=True)
    amp_embeddings_norm = (amp_embeddings - mean) / (std + 1e-8)
    
    print(f"Normalized:")
    print(f"Mean: {amp_embeddings_norm.mean():.4f}")
    print(f"Std:  {amp_embeddings_norm.std():.4f}")
    print(f"Min:  {amp_embeddings_norm.min():.4f}")
    print(f"Max:  {amp_embeddings_norm.max():.4f}")
    
    # 保存原始 embeddings (AMP only, 未归一化)
    output_path_raw = emb_dir / "esm3_embeddings_amp_only.npz"
    np.savez(
        output_path_raw,
        embeddings=amp_embeddings.astype(np.float32),
        sequences=amp_sequences,
        indices=np.arange(len(amp_embeddings)),
    )
    print(f"\n✅ Saved raw embeddings to {output_path_raw}")
    
    # 保存归一化 embeddings (AMP only)
    output_path_norm = emb_dir / "esm3_embeddings_amp_only_normalized.npz"
    np.savez(
        output_path_norm,
        embeddings=amp_embeddings_norm.astype(np.float32),
        sequences=amp_sequences,
        indices=np.arange(len(amp_embeddings)),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    print(f"✅ Saved normalized embeddings to {output_path_norm}")
    
    # 保存 metadata
    output_meta = emb_dir / "metadata_amp_only.csv"
    amp_metadata.to_csv(output_meta, index=False)
    print(f"✅ Saved metadata to {output_meta}")
    
    # 更新 splits（过滤现有 splits）
    print(f"\n=== Updating splits ===")
    
    # 创建 old_idx -> new_idx 的映射
    # 因为我们删除了 non-AMP，原始索引不再对应
    old_to_new = {}
    for new_idx, old_idx in enumerate(amp_indices):
        old_to_new[old_idx] = new_idx
    
    splits_dir_new = data_dir / "splits_amp_only"
    splits_dir_new.mkdir(exist_ok=True)
    
    for split_name in ["train", "val", "test"]:
        # 读取原始 split CSV（包含原始索引）
        split_csv = splits_dir / f"{split_name}.csv"
        split_df = pd.read_csv(split_csv)
        
        # 过滤掉 non-AMP
        split_df_amp = split_df[split_df["source"] != "non-AMP"].copy()
        
        # 更新 embedding_idx（映射到新的索引）
        split_df_amp["embedding_idx"] = split_df_amp.index.map(
            lambda old_idx: amp_metadata.index[amp_metadata["id"] == split_df_amp.loc[old_idx, "id"]].values[0]
        )
        
        # 重置索引
        split_df_amp = split_df_amp.reset_index(drop=True)
        split_df_amp["idx"] = np.arange(len(split_df_amp))
        
        # 保存
        split_df_amp.to_csv(splits_dir_new / f"{split_name}.csv", index=False)
        np.save(splits_dir_new / f"{split_name}_indices.npy", split_df_amp["embedding_idx"].values)
        
        print(f"  {split_name}: {len(split_df_amp)} sequences (removed {len(split_df) - len(split_df_amp)} non-AMP)")
    
    print(f"\n✅ Saved splits to {splits_dir_new}/")
    
    # 最终统计
    print(f"\n=== Summary ===")
    print(f"Original:  10,947 sequences (10,491 AMP + 456 Non-AMP)")
    print(f"Filtered:  {len(amp_metadata)} sequences (100% AMP)")
    print(f"Removed:   456 Non-AMP sequences")
    print(f"\nFiles created:")
    print(f"  - {output_path_raw}")
    print(f"  - {output_path_norm}")
    print(f"  - {output_meta}")
    print(f"  - {splits_dir_new}/")


if __name__ == "__main__":
    main()

