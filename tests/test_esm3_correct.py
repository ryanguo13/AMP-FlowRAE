#!/usr/bin/env python3
"""
Test the CORRECT way to get ESM3 embeddings.
"""

import torch


def test_correct_method():
    """Test the correct embedding extraction."""
    
    print("🧪 Testing CORRECT ESM3 embedding extraction...\n")
    
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    
    # Load model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = ESM3.from_pretrained("esm3-sm-open-v1", device=torch.device(device))
    model.eval()
    
    test_seq = "KLLKLLKLLK"
    protein = ESMProtein(sequence=test_seq)
    
    print(f"🧬 Test sequence: {test_seq}")
    print(f"   Length: {len(test_seq)} aa\n")
    print("=" * 70)
    
    # Step 1: Encode (tokenize)
    print("\n📋 Step 1: Tokenize with model.encode()")
    encoded = model.encode(protein)
    print(f"   encoded.sequence: {encoded.sequence}")
    print(f"   Shape: {encoded.sequence.shape}")
    print(f"   Dtype: {encoded.sequence.dtype}")
    
    # Step 2: Forward pass
    print("\n📋 Step 2: Forward pass with sequence_tokens")
    try:
        with torch.no_grad():
            # Add batch dimension and call forward
            sequence_tokens = encoded.sequence.unsqueeze(0)  # [1, seq_len]
            print(f"   Input shape: {sequence_tokens.shape}")
            
            output = model.forward(sequence_tokens=sequence_tokens)
        
        print(f"   ✅ Forward succeeded!")
        print(f"   Output type: {type(output)}")
        print(f"   Output attributes: {[a for a in dir(output) if not a.startswith('_')]}")
        
        # Check all attributes
        print("\n   📊 Output tensors:")
        for attr in dir(output):
            if not attr.startswith('_'):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor):
                    print(f"\n   {attr}:")
                    print(f"      shape: {val.shape}")
                    print(f"      dtype: {val.dtype}")
                    print(f"      device: {val.device}")
                    
                    if val.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        print(f"      ✅ FLOAT TENSOR!")
                        # Try to get per-residue embeddings
                        if len(val.shape) == 3:  # [batch, seq_len, hidden_dim]
                            print(f"      🎯 This looks like embeddings!")
                            print(f"         Batch size: {val.shape[0]}")
                            print(f"         Sequence length: {val.shape[1]}")
                            print(f"         Hidden dim: {val.shape[2]}")
                            
                            # Try mean pooling
                            pooled = val.mean(dim=1)  # [batch, hidden_dim]
                            print(f"      Mean pooled shape: {pooled.shape}")
                            print(f"      Sample values: {pooled[0, :5]}")
        
    except Exception as e:
        print(f"   ❌ Forward failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    
    # Step 3: Alternative - use logits with encoded protein
    print("\n📋 Step 3: Try logits() with encoded protein")
    try:
        with torch.no_grad():
            result = model.logits(encoded)
        
        print(f"   ✅ logits() succeeded!")
        print(f"   Result type: {type(result)}")
        
        for attr in dir(result):
            if not attr.startswith('_'):
                val = getattr(result, attr)
                if isinstance(val, torch.Tensor):
                    print(f"\n   {attr}:")
                    print(f"      shape: {val.shape}")
                    print(f"      dtype: {val.dtype}")
        
    except Exception as e:
        print(f"   ❌ logits() failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_correct_method()

