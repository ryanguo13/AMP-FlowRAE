#!/usr/bin/env python3
"""
计算条件标签 - Phase 2 准备

计算简单但实用的条件：
- is_amp: 0/1 binary label
- charge: net charge at pH 7
- hydrophobicity: Kyte-Doolittle scale average
- length: already in metadata

这些条件将用于 Phase 3 Conditional Flow Matching
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Net charge at pH 7 (simplified)
CHARGE = {
    "K": +1, "R": +1,  # positive
    "D": -1, "E": -1,  # negative
}


def compute_charge(seq: str) -> int:
    """计算 net charge at pH 7"""
    return sum(CHARGE.get(aa, 0) for aa in seq)


def compute_hydrophobicity(seq: str) -> float:
    """计算 Kyte-Doolittle 平均疏水性"""
    values = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    return np.mean(values) if values else 0.0


def main():
    # 路径
    data_dir = Path("data")
    metadata_path = data_dir / "embeddings" / "metadata.csv"
    output_path = data_dir / "embeddings" / "metadata_with_labels.csv"

    # 读取
    df = pd.read_csv(metadata_path)
    print(f"Processing {len(df)} sequences...")

    # 计算标签（如果列不存在）
    if "is_amp" not in df.columns:
        df["is_amp"] = df["source"].apply(lambda x: 0 if "non-amp" in x.lower() else 1)
    if "charge" not in df.columns:
        df["charge"] = df["sequence"].apply(compute_charge)
    if "hydrophobicity" not in df.columns:
        df["hydrophobicity"] = df["sequence"].apply(compute_hydrophobicity)

    # 统计
    print(f"\nLabel statistics:")
    print(f"  AMP / Non-AMP: {df['is_amp'].sum()} / {(~df['is_amp'].astype(bool)).sum()}")
    print(f"  Charge: mean={df['charge'].mean():.2f}, std={df['charge'].std():.2f}, "
          f"range=[{df['charge'].min()}, {df['charge'].max()}]")
    print(f"  Hydrophobicity: mean={df['hydrophobicity'].mean():.2f}, "
          f"std={df['hydrophobicity'].std():.2f}, "
          f"range=[{df['hydrophobicity'].min():.2f}, {df['hydrophobicity'].max():.2f}]")
    print(f"  Length: mean={df['length'].mean():.1f}, std={df['length'].std():.1f}, "
          f"range=[{df['length'].min()}, {df['length'].max()}]")

    # 保存到两个位置（保持向后兼容）
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    
    # 同时更新原始 metadata.csv（原地更新）
    df.to_csv(metadata_path, index=False)
    print(f"✅ Updated {metadata_path}")


if __name__ == "__main__":
    main()

