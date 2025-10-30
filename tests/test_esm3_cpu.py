#!/usr/bin/env python3
"""
Test ESM3 on CPU to see if it works.
"""

import torch


def test_cpu():
    """Test ESM3 on CPU."""
    
    print("🧪 Testing ESM3 on CPU...\n")
    
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    
    # FORCE CPU
    device = 'cpu'
    print(f"🖥️  Device: {device}\n")
    
    model = ESM3.from_pretrained("esm3-sm-open-v1", device=torch.device(device))
    model.eval()
    
    test_seq = "KLLKLLKLLK"
    protein = ESMProtein(sequence=test_seq)
    
    print(f"🧬 Test sequence: {test_seq}\n")
    print("=" * 70)
    
    # Method 1: Forward with sequence_tokens
    print("\n📋 Method 1: model.forward(sequence_tokens=...)")
    try:
        encoded = model.encode(protein)
        sequence_tokens = encoded.sequence.unsqueeze(0)  # [1, seq_len]
        
        with torch.no_grad():
            output = model.forward(sequence_tokens=sequence_tokens)
        
        print(f"   ✅ Forward succeeded on CPU!")
        print(f"   Output type: {type(output)}")
        
        # Find embeddings
        for attr in dir(output):
            if not attr.startswith('_'):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor) and val.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    print(f"\n   {attr}:")
                    print(f"      shape: {val.shape}")
                    print(f"      dtype: {val.dtype}")
                    
                    if len(val.shape) == 3:  # [batch, seq_len, hidden_dim]
                        print(f"      🎯 Per-residue embeddings!")
                        # Mean pool
                        pooled = val[0].mean(dim=0)  # Remove batch, average over seq
                        print(f"      Mean pooled: {pooled.shape}")
                        print(f"      Sample: {pooled[:5]}")
                        
                        print(f"\n      ✅✅✅ THIS IS WHAT WE NEED! ✅✅✅")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: logits()
    print("\n" + "=" * 70)
    print("\n📋 Method 2: model.logits()")
    try:
        encoded = model.encode(protein)
        
        with torch.no_grad():
            output = model.logits(encoded)
        
        print(f"   ✅ logits() succeeded on CPU!")
        
        for attr in dir(output):
            if not attr.startswith('_'):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor) and val.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    print(f"\n   {attr}: {val.shape}, {val.dtype}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    test_cpu()

