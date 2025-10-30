#!/usr/bin/env python3
"""
Test ESM3 model loading to verify the fix.
"""

import sys
from pathlib import Path


def test_esm3_loading():
    """Quick test of ESM3 model loading."""
    
    print("🧪 Testing ESM3 model loading...\n")
    
    # Test 1: Import
    print("1️⃣  Testing imports...")
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein
        print("   ✅ ESM library imports successful\n")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   Install with: uv pip install esm\n")
        return False
    
    # Test 2: Model loading
    print("2️⃣  Testing model loading...")
    print("   (This may take 1-2 minutes if downloading for first time)")
    try:
        import torch
        device = 'cpu'  # Use CPU for testing
        model = ESM3.from_pretrained("esm3-sm-open-v1", device=torch.device(device))
        model.eval()
        print(f"   ✅ Model loaded successfully on {device}\n")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}\n")
        return False
    
    # Test 3: Single sequence encoding
    print("3️⃣  Testing sequence encoding...")
    try:
        test_seq = "KLLKLLKLLK"  # Simple test peptide
        protein = ESMProtein(sequence=test_seq)
        
        import torch
        with torch.no_grad():
            encoded = model.encode(protein)
            embedding = encoded.sequence.mean(dim=0)
        
        print(f"   ✅ Encoded test sequence: {test_seq}")
        print(f"   ✅ Embedding shape: {embedding.shape}")
        print(f"   ✅ Expected: torch.Size([1536])\n")
        
        if embedding.shape[0] != 1536:
            print(f"   ⚠️  Warning: Expected 1536 dims, got {embedding.shape[0]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Encoding failed: {e}\n")
        return False
    
    # Success
    print("=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"   • ESM3 model: esm3-sm-open-v1")
    print(f"   • Embedding dims: 1536")
    print(f"   • Device: {device}")
    print(f"\n✅ Ready to run: python scripts/03_extract_esm3.py")
    
    return True


if __name__ == "__main__":
    success = test_esm3_loading()
    sys.exit(0 if success else 1)

