#!/usr/bin/env python3
"""
Test ESM3 forward and logits methods.
"""

import torch


def test_forward_logits():
    """Test forward and logits methods."""
    
    print("🧪 Testing ESM3 forward/logits methods...\n")
    
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    
    # Load model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = ESM3.from_pretrained("esm3-sm-open-v1", device=torch.device(device))
    model.eval()
    
    test_seq = "KLLKLLKLLK"
    protein = ESMProtein(sequence=test_seq)
    
    print(f"🧬 Test sequence: {test_seq}\n")
    print("=" * 70)
    
    # Test 1: logits()
    print("\n📋 Test 1: model.logits()")
    try:
        with torch.no_grad():
            result = model.logits(protein)
        
        print(f"   Type: {type(result)}")
        print(f"   Attributes: {dir(result)}")
        
        # Check all tensor attributes
        for attr in dir(result):
            if not attr.startswith('_'):
                val = getattr(result, attr)
                if isinstance(val, torch.Tensor):
                    print(f"\n   {attr}:")
                    print(f"      dtype: {val.dtype}")
                    print(f"      shape: {val.shape}")
                    print(f"      device: {val.device}")
                    
                    # If float tensor, show sample values
                    if val.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        print(f"      ✅ FLOAT TENSOR - This might be what we need!")
                        print(f"      Sample values: {val[0, :5]}")
                        
    except Exception as e:
        print(f"   ❌ logits() failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: forward with no args
    print("\n" + "=" * 70)
    print("\n📋 Test 2: model.forward() with no args")
    try:
        with torch.no_grad():
            # Check signature
            import inspect
            sig = inspect.signature(model.forward)
            print(f"   Signature: {sig}")
            
            # Try calling with different args
            result = model.forward()
            print(f"   Result type: {type(result)}")
            
    except Exception as e:
        print(f"   ❌ forward() failed: {e}")
    
    # Test 3: encoder
    print("\n" + "=" * 70)
    print("\n📋 Test 3: model.encoder")
    try:
        print(f"   Type: {type(model.encoder)}")
        print(f"   Methods: {[m for m in dir(model.encoder) if not m.startswith('_')][:10]}")
        
        # Try encoder forward
        encoded = model.encode(protein)
        print(f"\n   Trying encoder.forward()...")
        with torch.no_grad():
            # Check what encoder expects
            import inspect
            if hasattr(model.encoder, 'forward'):
                sig = inspect.signature(model.encoder.forward)
                print(f"   Signature: {sig}")
        
    except Exception as e:
        print(f"   ❌ encoder test failed: {e}")
    
    # Test 4: Check ESMProteinTensor usage
    print("\n" + "=" * 70)
    print("\n📋 Test 4: Using ESMProteinTensor with model")
    try:
        encoded = model.encode(protein)
        print(f"   Encoded type: {type(encoded)}")
        print(f"   Trying to pass encoded to model...")
        
        with torch.no_grad():
            # Maybe forward takes ESMProteinTensor
            output = model(encoded)
        
        print(f"   Output type: {type(output)}")
        
        for attr in dir(output):
            if not attr.startswith('_'):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor):
                    print(f"\n   {attr}: {val.shape}, {val.dtype}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: Check model.__call__
    print("\n" + "=" * 70)
    print("\n📋 Test 5: Direct model call with protein")
    try:
        with torch.no_grad():
            output = model(protein)
        
        print(f"   Output type: {type(output)}")
        
        for attr in dir(output):
            if not attr.startswith('_'):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor):
                    print(f"   {attr}: {val.shape}, {val.dtype}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    test_forward_logits()

