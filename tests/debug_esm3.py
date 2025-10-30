#!/usr/bin/env python3
"""
Debug ESM3 encoding to see what's actually happening.
"""

import torch
import pandas as pd
from pathlib import Path


def debug_single_sequence():
    """Test encoding a single real sequence from your dataset."""
    
    print("🔍 Debugging ESM3 encoding...\n")
    
    # Load your actual data
    data_file = Path('data/processed/dedup_sequences.csv')
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    
    # Get first 5 sequences
    test_seqs = df['sequence'].head(5).tolist()
    
    print(f"📊 Testing {len(test_seqs)} sequences from your dataset\n")
    
    # Load model
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    
    # Use MPS like your actual run
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🖥️  Device: {device}\n")
    
    model = ESM3.from_pretrained("esm3-sm-open-v1", device=torch.device(device))
    model.eval()
    
    print(f"✓ Model loaded\n")
    print("=" * 70)
    
    # Test each sequence
    for i, seq in enumerate(test_seqs):
        print(f"\n🧪 Sequence {i+1}: {seq[:30]}{'...' if len(seq) > 30 else ''}")
        print(f"   Length: {len(seq)} aa")
        
        try:
            protein = ESMProtein(sequence=seq)
            print(f"   ✓ ESMProtein created")
            
            with torch.no_grad():
                encoded = model.encode(protein)
            
            # ===== 关键诊断信息 =====
            print(f"\n   📋 Encoded object type: {type(encoded)}")
            print(f"   📋 Encoded attributes: {dir(encoded)}")
            print(f"\n   🔬 encoded.sequence type: {type(encoded.sequence)}")
            print(f"   🔬 encoded.sequence value: {encoded.sequence}")
            
            if encoded.sequence is not None:
                print(f"   🔬 encoded.sequence shape: {encoded.sequence.shape}")
                print(f"   🔬 encoded.sequence dtype: {encoded.sequence.dtype}")
                print(f"   🔬 encoded.sequence device: {encoded.sequence.device}")
                
                # Try mean
                try:
                    emb = encoded.sequence.mean(dim=0)
                    print(f"\n   ✅ Mean pooling succeeded!")
                    print(f"   ✅ Embedding shape: {emb.shape}")
                    print(f"   ✅ Embedding dtype: {emb.dtype}")
                except Exception as e:
                    print(f"\n   ❌ Mean pooling failed: {e}")
                    print(f"      Full error: {repr(e)}")
            else:
                print(f"   ❌ encoded.sequence is None!")
            
        except Exception as e:
            print(f"\n   ❌ Encoding failed: {e}")
            print(f"      Full error: {repr(e)}")
            import traceback
            print("\n   Full traceback:")
            traceback.print_exc()
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    debug_single_sequence()

