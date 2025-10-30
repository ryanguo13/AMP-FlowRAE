#!/usr/bin/env python3
"""
Validate Phase 1 output data quality.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("🔍 Validating Phase 1 outputs...\n")
    
    # Check processed data
    print("📊 Step 1: Processed sequences")
    dedup_csv = Path('data/processed/dedup_sequences.csv')
    
    if not dedup_csv.exists():
        print("   ❌ dedup_sequences.csv not found")
        return
    
    df = pd.read_csv(dedup_csv)
    print(f"   ✓ Loaded {len(df)} sequences")
    print(f"   ✓ Length range: {df['length'].min()}-{df['length'].max()} aa")
    print(f"   ✓ Sources: {df['source'].nunique()}")
    print(f"\n   Distribution:")
    print(df['source'].value_counts().to_string(header=False))
    
    # Check embeddings
    print(f"\n📊 Step 2: ESM-3 embeddings")
    emb_file = Path('data/embeddings/esm3_embeddings.npz')
    
    if not emb_file.exists():
        print("   ⏳ esm3_embeddings.npz not found (still processing?)")
        return
    
    data = np.load(emb_file)
    embeddings = data['embeddings']
    
    print(f"   ✓ Embedding shape: {embeddings.shape}")
    print(f"   ✓ Expected: ({len(df)}, 1536)  # ESM3-sm dimension")
    print(f"   ✓ Data type: {embeddings.dtype}")
    print(f"   ✓ File size: {emb_file.stat().st_size / 1e6:.1f} MB")
    
    # Statistics
    print(f"\n📊 Embedding statistics:")
    print(f"   Mean: {embeddings.mean():.4f}")
    print(f"   Std:  {embeddings.std():.4f}")
    print(f"   Min:  {embeddings.min():.4f}")
    print(f"   Max:  {embeddings.max():.4f}")
    
    # Sanity checks
    print(f"\n✅ Validation checks:")
    
    # Check 1: No NaN/Inf
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    print(f"   {'✓' if not has_nan else '❌'} No NaN values")
    print(f"   {'✓' if not has_inf else '❌'} No Inf values")
    
    # Check 2: Dimensions match
    dims_match = embeddings.shape == (len(df), 1536)
    print(f"   {'✓' if dims_match else '❌'} Dimensions match dataset (1536 dims)")
    
    # Check 3: Not all zeros
    not_zeros = (embeddings != 0).any()
    print(f"   {'✓' if not_zeros else '❌'} Contains non-zero values")
    
    # Check 4: Reasonable value range
    reasonable = (embeddings.min() > -100) and (embeddings.max() < 100)
    print(f"   {'✓' if reasonable else '❌'} Values in reasonable range")
    
    if all([not has_nan, not has_inf, dims_match, not_zeros, reasonable]):
        print(f"\n🎉 All checks passed! Phase 1 complete.")
    else:
        print(f"\n⚠️  Some checks failed. Review the data.")


if __name__ == '__main__':
    main()

