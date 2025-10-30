#!/usr/bin/env python3
"""
Test the fixed extraction script with a few sequences.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys


def load_esm3_model(device='cpu'):
    """Load ESM-3 model."""
    
    print(f"🔧 Loading ESM-3 model on {device}...")
    
    from esm.models.esm3 import ESM3
    
    device_obj = torch.device(device)
    model = ESM3.from_pretrained("esm3-sm-open-v1", device=device_obj)
    model.eval()
    
    print(f"   ✓ ESM-3 loaded successfully (1536 dimensions)")
    
    return model


def extract_embeddings_batch(sequences: list[str], model, device='cpu', batch_size=8):
    """Extract embeddings using ESM-3.
    
    Uses the correct API: encode() -> forward() -> output.embeddings
    """
    
    from esm.sdk.api import ESMProtein
    
    embeddings = []
    failed = 0
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i+batch_size]
        
        for seq in batch_seqs:
            try:
                # Step 1: Create ESMProtein and tokenize
                protein = ESMProtein(sequence=seq)
                encoded = model.encode(protein)
                
                # Step 2: Forward pass to get embeddings
                with torch.no_grad():
                    # Add batch dimension
                    sequence_tokens = encoded.sequence.unsqueeze(0)  # [1, seq_len]
                    
                    # Forward pass
                    output = model.forward(sequence_tokens=sequence_tokens)
                    
                    # Extract embeddings: [1, seq_len, 1536]
                    # Mean pool over sequence dimension
                    emb = output.embeddings[0].mean(dim=0)  # [1536]
                    
                    embeddings.append(emb.cpu().numpy())
                    
            except Exception as e:
                # Handle failed sequences
                failed += 1
                # Append zero vector as placeholder
                embeddings.append(np.zeros(1536))  # ESM3-sm has 1536 dims
                if failed <= 5:  # Only show first few errors
                    print(f"\n⚠️  Failed sequence {i//batch_size}/{len(sequences)//batch_size}: {str(e)}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} sequences failed, filled with zeros")
    
    return np.array(embeddings)


def main():
    """Test with first 10 sequences."""
    
    input_csv = Path('data/processed/dedup_sequences.csv')
    
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        sys.exit(1)
    
    # Load ONLY first 10 sequences for testing
    df = pd.read_csv(input_csv)
    sequences = df['sequence'].head(10).tolist()
    
    print(f"📊 Testing with {len(sequences)} sequences")
    print(f"   Length range: {df['length'].head(10).min()} - {df['length'].head(10).max()} aa\n")
    
    # Force CPU
    device = 'cpu'
    batch_size = 4
    
    print(f"🖥️  Device: {device} (batch_size={batch_size})\n")
    
    # Load ESM-3 model
    model = load_esm3_model(device=device)
    
    # Extract embeddings
    embeddings = extract_embeddings_batch(sequences, model, device, batch_size)
    
    print(f"\n✅ Test completed!")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Expected: (10, 1536)")
    
    # Check for zeros (failed sequences)
    zero_count = (embeddings.sum(axis=1) == 0).sum()
    print(f"   Zero embeddings: {zero_count}/10")
    
    if zero_count == 0 and embeddings.shape == (10, 1536):
        print(f"\n🎉 SUCCESS! All sequences encoded correctly!")
        print(f"   Sample embedding stats:")
        print(f"   Mean: {embeddings[0].mean():.2f}")
        print(f"   Std: {embeddings[0].std():.2f}")
        print(f"   Min: {embeddings[0].min():.2f}")
        print(f"   Max: {embeddings[0].max():.2f}")
    else:
        print(f"\n❌ FAILED: {zero_count} sequences returned zeros")


if __name__ == '__main__':
    main()

