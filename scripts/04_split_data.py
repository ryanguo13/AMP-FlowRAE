#!/usr/bin/env python3
"""
数据划分脚本 - Phase 1 收尾

按 source 分层，80/10/10 划分 train/val/test
避免数据泄露，确保每个 source 在各 split 中比例一致
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    # 路径
    data_dir = Path("data")
    metadata_path = data_dir / "embeddings" / "metadata.csv"
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # 读取 metadata
    df = pd.read_csv(metadata_path)
    print(f"Total sequences: {len(df)}")
    print(f"Sources: {df['source'].value_counts()}")

    # 分层划分 (按 source)
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["source"], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=df.loc[temp_idx, "source"], random_state=42
    )

    # 验证
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    test_df = df.loc[test_idx]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    print(f"\nSource distribution:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {split_name}:")
        for src, count in split_df["source"].value_counts().items():
            print(f"    {src}: {count} ({count/len(split_df)*100:.1f}%)")

    # 保存索引 (只保存 embedding_idx，最轻量)
    np.save(splits_dir / "train_indices.npy", train_df["embedding_idx"].values)
    np.save(splits_dir / "val_indices.npy", val_df["embedding_idx"].values)
    np.save(splits_dir / "test_indices.npy", test_df["embedding_idx"].values)

    # 同时保存 CSV (方便调试)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"\n✅ Saved splits to {splits_dir}/")


if __name__ == "__main__":
    main()



