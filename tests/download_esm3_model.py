#!/usr/bin/env python3
"""
Download complete ESM3 model to local directory.

This ensures you have all necessary files (weights + config) for offline use.
"""

import sys
from pathlib import Path


def download_esm3_model():
    """Download ESM3 model and save to local directory."""
    
    print("🔽 Downloading ESM3 model (esm3-sm-open-v1)...")
    print("   This will download ~1-2 GB to local models/ directory")
    
    try:
        from esm.models.esm3 import ESM3
        
        # Download to temporary location (will use cache)
        print("\n📦 Step 1: Loading model (will use ~/.cache/esm/)...")
        model = ESM3.from_pretrained("esm3-sm-open-v1")
        
        # Save to local directory
        output_dir = Path("models/esm3_sm_open_v1")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Step 2: Saving complete model to {output_dir}...")
        model.save_pretrained(str(output_dir))
        
        print(f"\n✅ Success! ESM3 model saved to: {output_dir}")
        print(f"\n📁 Directory contents:")
        for f in sorted(output_dir.iterdir()):
            size_mb = f.stat().st_size / (1024**2) if f.is_file() else 0
            print(f"   - {f.name:30s} ({size_mb:.1f} MB)" if f.is_file() else f"   - {f.name}/")
        
        print(f"\n🎯 Next: Run embedding extraction script")
        print(f"   python scripts/03_extract_esm3.py")
        
    except ImportError as e:
        print(f"\n❌ Error: ESM library not installed")
        print(f"   {e}")
        print(f"\n   Install with: uv pip install esm")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print(f"\n   Possible issues:")
        print(f"   1. No internet connection")
        print(f"   2. HuggingFace Hub unavailable")
        print(f"   3. Insufficient disk space (~2 GB needed)")
        sys.exit(1)


if __name__ == "__main__":
    download_esm3_model()

