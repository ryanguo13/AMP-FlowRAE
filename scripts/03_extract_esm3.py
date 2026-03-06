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


def load_esm3_model(device='cuda'):
    """Load ESM-3 model from local or auto-download."""
    
    print(f"🔧 Loading ESM-3 model on {device}...")
    
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, GenerationConfig
        
        device_obj = torch.device(device)
        
        # Try 1: Load from local directories (support env override, absolute/relative, hyphen/underscore)
        import os
        repo_root = Path(__file__).resolve().parents[1]
        env_override = os.environ.get("AMP_FLOWRAE_ESM3_DIR")
        candidate_dirs = []
        if env_override:
            candidate_dirs.append(Path(env_override))
        candidate_dirs.extend([
            repo_root / "esm3-sm-open-v1",
            repo_root / "esm3_sm_open_v1",
            repo_root / "models" / "esm3-sm-open-v1",
            repo_root / "models" / "esm3_sm_open_v1",
            Path("./esm3-sm-open-v1"),
            Path("./esm3_sm_open_v1"),
            Path("models/esm3-sm-open-v1"),
            Path("models/esm3_sm_open_v1"),
        ])

        import os
        import shutil
        model = None
        
        # 首先检查 HuggingFace 缓存目录（优先级最高）
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        esm3_cache = hf_cache / "models--EvolutionaryScale--esm3-sm-open-v1" / "snapshots"
        if esm3_cache.exists():
            # 查找最新的 snapshot
            snapshots = sorted([s for s in esm3_cache.iterdir() if s.is_dir()], 
                             key=lambda x: x.stat().st_mtime, reverse=True)
            for snapshot in snapshots:
                if (snapshot / "config.json").exists() and (snapshot / "pytorch_model.bin").exists():
                    print(f"   Found cached model at: {snapshot}")
                    try:
                        model = ESM3.from_pretrained(str(snapshot), device=device_obj)
                        print(f"   ✓ Loaded from HuggingFace cache")
                        break
                    except Exception as cache_e:
                        print(f"   ⚠️  Cache load failed: {cache_e}")
        
        # 如果缓存加载失败，检查是否有 HFD 格式的模型（权重在 data/weights/）
        for d in candidate_dirs:
            if not d.exists():
                continue
            weight_file = d / "data" / "weights" / "esm3_sm_open_v1.pth"
            if weight_file.exists():
                print(f"   Found HFD format model at: {d}")
                # 尝试创建符合 ESM-3 期望的目录结构
                # ESM-3 期望权重文件在根目录或特定位置
                temp_model_dir = d.parent / "esm3-sm-open-v1-temp"
                try:
                    # 创建临时目录并复制/链接权重文件
                    temp_model_dir.mkdir(exist_ok=True, parents=True)
                    # 复制权重文件到根目录（如果不存在）
                    target_weight = temp_model_dir / "pytorch_model.bin"
                    if not target_weight.exists():
                        print(f"   Copying weights to temporary location...")
                        shutil.copy2(weight_file, target_weight)
                    # 复制 config.json（如果存在）
                    if (d / "config.json").exists():
                        shutil.copy2(d / "config.json", temp_model_dir / "config.json")
                    
                    # 尝试从临时目录加载
                    model = ESM3.from_pretrained(str(temp_model_dir), device=device_obj)
                    print(f"   ✓ Loaded from HFD format (via temp directory)")
                    break
                except Exception as hfd_e:
                    print(f"   ⚠️  Temp directory method failed: {hfd_e}")
                    # 清理临时目录
                    if temp_model_dir.exists():
                        try:
                            shutil.rmtree(temp_model_dir)
                        except:
                            pass
        
        # 如果 HFD 格式失败，尝试标准格式
        if model is None:
            for d in candidate_dirs:
                if not d.exists():
                    continue
                try:
                    has_config = (d / "config.json").exists()
                    has_hf_weights = (d / "model.safetensors").exists() or ((d / "pytorch_model.bin").exists() and (d / "pytorch_model.bin").stat().st_size > 1000)
                    if has_config or has_hf_weights:
                        print(f"   Loading from local: {d}")
                        model = ESM3.from_pretrained(str(d), device=device_obj)
                        break
                except Exception as inner_e:
                    print(f"   ⚠️  Local directory present but failed to load: {inner_e}")
                    model = None
        if model is None:
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
                    
                    # Forward pass with dtype-safe context on CUDA
                    # Use float16 instead of bfloat16 for better GPU compatibility
                    if torch.cuda.is_available():
                        from torch.amp import autocast
                        with autocast(device_type='cuda', dtype=torch.float16):
                            output = model.forward(sequence_tokens=sequence_tokens)
                    else:
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
