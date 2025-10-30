#!/usr/bin/env python3
"""
Extract ESM-3 embeddings for all sequences.

Uses ESM3 (esm3-sm-open-v1) with batched inference.
Supports: CUDA (GPU) > MPS (Mac GPU) > CPU
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys


def load_esm3_model(device='cpu'):
    """Load ESM-3 model from local or auto-download."""
    
    print(f"🔧 Loading ESM-3 model on {device}...")
    
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, GenerationConfig
        
        device_obj = torch.device(device)
        
        # Try 1: Load from local complete model directory
        local_model_dir = Path("models/esm3_sm_open_v1")
        if local_model_dir.exists() and (local_model_dir / "config.json").exists():
            print(f"   Loading from local: {local_model_dir}")
            model = ESM3.from_pretrained(str(local_model_dir), device=device_obj)
        else:
            # Try 2: Auto-download (will cache to ~/.cache/esm/)
            print(f"   Local model incomplete, downloading esm3-sm-open-v1...")
            print(f"   (Will cache to ~/.cache/esm/ for future use)")
            model = ESM3.from_pretrained("esm3-sm-open-v1", device=device_obj)
        
        model.eval()
        
        print(f"   ✓ ESM-3 loaded successfully (1536 dimensions)")
        
        return model
        
    except ImportError as e:
        print(f"❌ ESM-3 import failed: {e}")
        print("   Make sure you have installed esm:")
        print("   uv pip install esm")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print(f"   Error details: {str(e)}")
        print(f"\n   Troubleshooting:")
        print(f"   1. Check internet connection (model will auto-download)")
        print(f"   2. Or set up complete local model directory")
        sys.exit(1)


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
    # Paths
    data_processed = Path('data/processed')
    data_embeddings = Path('data/embeddings')
    data_embeddings.mkdir(exist_ok=True, parents=True)
    
    input_csv = data_processed / 'dedup_sequences.csv'
    output_file = data_embeddings / 'esm3_embeddings.npz'
    
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        print("   Run 01_parse_fasta.py and 02_deduplicate.py first")
        sys.exit(1)
    
    # Load sequences
    df = pd.read_csv(input_csv)
    sequences = df['sequence'].tolist()
    
    print(f"📊 Dataset: {len(sequences)} sequences")
    print(f"   Length range: {df['length'].min()} - {df['length'].max()} aa")
    
    # Setup device
    # Note: ESM3 has a bug with MPS (Mac GPU), forcing CPU for now
    if torch.cuda.is_available():
        device = 'cuda'
        batch_size = 32
    else:
        # Force CPU even if MPS available (ESM3 bug: "Unsupported device type: mps")
        device = 'cpu'
        batch_size = 4  # Smaller batch for CPU
        if torch.backends.mps.is_available():
            print(f"⚠️  MPS available but not supported by ESM3, using CPU")
    
    print(f"🖥️  Device: {device} (batch_size={batch_size})")
    
    # Load ESM-3 model
    model = load_esm3_model(device=device)
    
    # Extract embeddings
    embeddings = extract_embeddings_batch(sequences, model, device, batch_size)
    
    # Save embeddings
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        sequences=sequences,
        indices=df.index.values,
    )
    
    print(f"\n✅ Embeddings saved:")
    print(f"   File: {output_file}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Size: {output_file.stat().st_size / 1e6:.1f} MB")
    
    # Save metadata CSV with embedding indices
    df['embedding_idx'] = df.index
    meta_csv = data_embeddings / 'metadata.csv'
    df.to_csv(meta_csv, index=False)
    print(f"   Metadata: {meta_csv}")


if __name__ == '__main__':
    main()
