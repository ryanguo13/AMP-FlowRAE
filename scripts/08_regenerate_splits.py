#!/usr/bin/env python3
"""
重新生成 splits - 基于已过滤的 AMP-only 数据

简单直接，不依赖旧文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    # 读取当前的 metadata（已过滤，只有 AMP）
    metadata_path = Path("data/embeddings/metadata.csv")
    meta = pd.read_csv(metadata_path)
    
    print(f"Total sequences: {len(meta)}")
    print(f"Sources: {meta['source'].value_counts()}")
    
    # 分层划分 (按 source)
    train_idx, temp_idx = train_test_split(
        meta.index, test_size=0.2, stratify=meta["source"], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=meta.loc[temp_idx, "source"], random_state=42
    )
    
    train_df = meta.loc[train_idx]
    val_df = meta.loc[val_idx]
    test_df = meta.loc[test_idx]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(meta)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(meta)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(meta)*100:.1f}%)")
    
    # 保存
    splits_dir = Path("data/splits")
    splits_dir.mkdir(exist_ok=True)
    
    # embedding_idx 就是 row index（因为数据已经过滤且重新索引）
    np.save(splits_dir / "train_indices.npy", train_df.index.values)
    np.save(splits_dir / "val_indices.npy", val_df.index.values)
    np.save(splits_dir / "test_indices.npy", test_df.index.values)
    
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    
    print(f"\n✅ Splits saved to {splits_dir}/")
    
    # 验证
    print(f"\n=== 验证 ===")
    print(f"train_indices: max={train_df.index.max()}, len={len(train_df)}")
    print(f"metadata: len={len(meta)}")
    if train_df.index.max() < len(meta):
        print("✅ 索引正常！")
    else:
        print("❌ 索引超出")


if __name__ == "__main__":
    main()


