#!/usr/bin/env python3
"""
Test different ESM3 API methods to find the correct way to get embeddings.
"""

import torch


def test_esm3_api():
    """Test various ESM3 methods."""
    
    print("🧪 Testing ESM3 API methods...\n")
    
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    
    # Load model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🖥️  Device: {device}\n")
    
    model = ESM3.from_pretrained("esm3-sm-open-v1", device=torch.device(device))
    model.eval()
    
    # Test sequence
    test_seq = "KLLKLLKLLK"
    protein = ESMProtein(sequence=test_seq)
    
    print(f"🧬 Test sequence: {test_seq}\n")
    print("=" * 70)
    
    # Method 1: encode()
    print("\n📋 Method 1: model.encode()")
    with torch.no_grad():
        encoded = model.encode(protein)
    print(f"   Type: {type(encoded)}")
    print(f"   encoded.sequence: {encoded.sequence}")
    print(f"   encoded.sequence dtype: {encoded.sequence.dtype}")
    print(f"   encoded.sequence shape: {encoded.sequence.shape}")
    
    # Method 2: Check model methods
    print("\n📋 Model available methods:")
    methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
    relevant = [m for m in methods if any(kw in m.lower() for kw in ['forward', 'embed', 'encode', 'logit'])]
    for m in relevant:
        print(f"   - {m}")
    
    # Method 3: Try forward()
    print("\n📋 Method 2: model.forward()")
    try:
        with torch.no_grad():
            # Forward typically needs tokenized input
            output = model.forward(encoded)
        print(f"   Type: {type(output)}")
        print(f"   Output attributes: {dir(output)}")
        
        # Check for embeddings
        if hasattr(output, 'embeddings'):
            print(f"   output.embeddings shape: {output.embeddings.shape}")
            print(f"   output.embeddings dtype: {output.embeddings.dtype}")
        if hasattr(output, 'last_hidden_state'):
            print(f"   output.last_hidden_state shape: {output.last_hidden_state.shape}")
        if hasattr(output, 'hidden_states'):
            print(f"   output.hidden_states: {output.hidden_states}")
            
    except Exception as e:
        print(f"   ❌ Forward failed: {e}")
    
    # Method 4: Check encode output attributes
    print("\n📋 Method 3: encoded object attributes")
    attrs = [a for a in dir(encoded) if not a.startswith('_')]
    print(f"   Available: {attrs}")
    
    for attr in attrs:
        val = getattr(encoded, attr)
        if isinstance(val, torch.Tensor):
            print(f"\n   {attr}:")
            print(f"      type: {type(val)}")
            print(f"      dtype: {val.dtype}")
            print(f"      shape: {val.shape}")
            if val.dtype in [torch.float32, torch.float16]:
                print(f"      ✅ This is a float tensor! Might be embeddings.")
    
    # Method 5: Try embed() if exists
    print("\n📋 Method 4: Checking for embed-related methods")
    if hasattr(model, 'embed'):
        print("   Found: model.embed()")
        try:
            with torch.no_grad():
                emb = model.embed(protein)
            print(f"   Result type: {type(emb)}")
            print(f"   Result shape: {emb.shape if isinstance(emb, torch.Tensor) else 'N/A'}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    if hasattr(model, 'get_embeddings'):
        print("   Found: model.get_embeddings()")
        try:
            with torch.no_grad():
                emb = model.get_embeddings(protein)
            print(f"   Result type: {type(emb)}")
            print(f"   Result shape: {emb.shape if isinstance(emb, torch.Tensor) else 'N/A'}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    test_esm3_api()

