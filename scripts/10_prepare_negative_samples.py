#!/usr/bin/env python3
"""
负样本数据准备脚本

从 UniProt 提取人源/哺乳动物短肽（30-80 aa），生成 ESM-3 embeddings
处理 APOE 滑窗等负样本模板
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 动态导入以数字开头的模块
import importlib.util
extract_esm3_path = Path(__file__).parent / "03_extract_esm3.py"
spec = importlib.util.spec_from_file_location("extract_esm3", extract_esm3_path)
extract_esm3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_esm3)
load_esm3_model = extract_esm3.load_esm3_model
extract_embeddings_batch = extract_esm3.extract_embeddings_batch


def read_fasta(path):
    """读取 FASTA 文件"""
    sequences = []
    headers = []
    header = None
    seq_parts = []
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    sequences.append(''.join(seq_parts))
                    headers.append(header)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        
        if header is not None:
            sequences.append(''.join(seq_parts))
            headers.append(header)
    
    return headers, sequences


def extract_uniprot_short_peptides(
    uniprot_fasta,
    output_fasta,
    min_length=30,
    max_length=80,
    max_samples=100000,
    organism_keywords=['Homo sapiens', 'Mus musculus', 'Rattus', 'Bos taurus', 'Sus scrofa']
):
    """
    从 UniProt 提取人源/哺乳动物短肽片段
    
    Args:
        uniprot_fasta: UniProt FASTA 文件路径
        output_fasta: 输出 FASTA 文件路径
        min_length: 最小长度
        max_length: 最大长度
        max_samples: 最大样本数（下采样）
        organism_keywords: 物种关键词列表
    """
    print(f"📖 Reading UniProt FASTA: {uniprot_fasta}")
    headers, sequences = read_fasta(uniprot_fasta)
    
    print(f"   Total sequences: {len(sequences)}")
    
    # 筛选人源/哺乳动物序列
    filtered_headers = []
    filtered_sequences = []
    
    for h, s in zip(headers, sequences):
        # 检查 header 中是否包含目标物种关键词
        h_upper = h.upper()
        if any(kw.upper() in h_upper for kw in organism_keywords):
            # 检查长度
            if min_length <= len(s) <= max_length:
                filtered_headers.append(h)
                filtered_sequences.append(s)
    
    print(f"   Filtered sequences (length {min_length}-{max_length}): {len(filtered_sequences)}")
    
    # 下采样
    if len(filtered_sequences) > max_samples:
        indices = np.random.choice(len(filtered_sequences), max_samples, replace=False)
        filtered_headers = [filtered_headers[i] for i in indices]
        filtered_sequences = [filtered_sequences[i] for i in indices]
        print(f"   Downsampled to: {len(filtered_sequences)}")
    
    # 写入 FASTA
    print(f"💾 Writing to: {output_fasta}")
    with open(output_fasta, 'w') as f:
        for h, s in zip(filtered_headers, filtered_sequences):
            f.write(f">{h}\n{s}\n")
    
    return filtered_headers, filtered_sequences


def prepare_apoe_windows(apoe_fasta, output_fasta, window_sizes=[50, 60, 70], repeat=20):
    """
    准备 APOE 滑窗负样本（重复多次）
    
    Args:
        apoe_fasta: APOE FASTA 文件路径
        output_fasta: 输出 FASTA 文件路径
        window_sizes: 窗口大小列表
        repeat: 重复次数（训练时高权重）
    """
    print(f"📖 Reading APOE FASTA: {apoe_fasta}")
    headers, sequences = read_fasta(apoe_fasta)
    
    if len(sequences) == 0:
        print("   ⚠️  No sequences found!")
        return [], []
    
    # 使用第一个序列（通常是人类 APOE）
    seq = sequences[0]
    print(f"   Sequence length: {len(seq)}")
    
    # 提取 90-160 位（如果序列足够长）
    if len(seq) >= 160:
        seq_region = seq[89:160]  # 90-160 (1-indexed -> 0-indexed)
        print(f"   Using region 90-160: length {len(seq_region)}")
    else:
        seq_region = seq
        print(f"   Using full sequence")
    
    # 生成滑窗
    all_headers = []
    all_sequences = []
    
    for w in window_sizes:
        for i in range(len(seq_region) - w + 1):
            window_seq = seq_region[i:i+w]
            for r in range(repeat):
                all_headers.append(f"APOE_region90-160_pos{i+1}_{i+w}_w{w}_rep{r}")
                all_sequences.append(window_seq)
    
    print(f"   Generated {len(all_sequences)} windows")
    
    # 写入 FASTA
    print(f"💾 Writing to: {output_fasta}")
    with open(output_fasta, 'w') as f:
        for h, s in zip(all_headers, all_sequences):
            f.write(f">{h}\n{s}\n")
    
    return all_headers, all_sequences


def extract_embeddings_for_sequences(headers, sequences, output_npz, device='cuda', batch_size=32):
    """
    为序列提取 ESM-3 embeddings
    
    Args:
        headers: 序列 headers
        sequences: 序列列表
        output_npz: 输出 .npz 文件路径
        device: 设备
        batch_size: batch 大小
    """
    print(f"🔧 Extracting ESM-3 embeddings...")
    print(f"   Device: {device}, Batch size: {batch_size}")
    
    model = load_esm3_model(device)
    
    # 批量提取（注意参数顺序：sequences, model, device, batch_size）
    embeddings = extract_embeddings_batch(
        sequences, model, device=device, batch_size=batch_size
    )
    
    print(f"   Extracted {len(embeddings)} embeddings")
    print(f"   Embedding shape: {embeddings[0].shape}")
    
    # 保存
    print(f"💾 Saving to: {output_npz}")
    np.savez_compressed(output_npz, embeddings=embeddings)
    
    # 保存 metadata
    metadata_path = output_npz.replace('.npz', '_metadata.csv')
    df = pd.DataFrame({
        'header': headers,
        'sequence': sequences,
        'length': [len(s) for s in sequences]
    })
    df.to_csv(metadata_path, index=False)
    print(f"💾 Metadata saved to: {metadata_path}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniprot-fasta", type=str, default="data/raw/uniprot_sprot.fasta")
    parser.add_argument("--apoe-fasta", type=str, default="data/raw/apoe_full.fasta")
    parser.add_argument("--output-dir", type=str, default="data/negative_samples")
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-embeddings", action="store_true", help="只生成 FASTA，不提取 embeddings")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. 提取 UniProt 短肽
    print("\n" + "="*60)
    print("Step 1: Extract UniProt short peptides")
    print("="*60)
    uniprot_output = output_dir / "uniprot_short_peptides.fasta"
    uniprot_headers, uniprot_seqs = extract_uniprot_short_peptides(
        args.uniprot_fasta,
        uniprot_output,
        args.min_length,
        args.max_length,
        args.max_samples
    )
    
    # 2. 准备 APOE 滑窗
    print("\n" + "="*60)
    print("Step 2: Prepare APOE windows")
    print("="*60)
    apoe_output = output_dir / "apoe_windows.fasta"
    apoe_headers, apoe_seqs = prepare_apoe_windows(
        args.apoe_fasta,
        apoe_output
    )
    
    # 3. 提取 embeddings（如果未跳过）
    if not args.skip_embeddings:
        print("\n" + "="*60)
        print("Step 3: Extract ESM-3 embeddings")
        print("="*60)
        
        # UniProt embeddings
        uniprot_npz = output_dir / "uniprot_short_peptides_embeddings.npz"
        extract_embeddings_for_sequences(
            uniprot_headers, uniprot_seqs, str(uniprot_npz),
            device=args.device, batch_size=args.batch_size
        )
        
        # APOE embeddings
        apoe_npz = output_dir / "apoe_windows_embeddings.npz"
        extract_embeddings_for_sequences(
            apoe_headers, apoe_seqs, str(apoe_npz),
            device=args.device, batch_size=args.batch_size
        )
    
    print("\n" + "="*60)
    print("✅ Negative sample preparation complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"  - UniProt short peptides: {len(uniprot_seqs)} sequences")
    print(f"  - APOE windows: {len(apoe_seqs)} sequences")


if __name__ == "__main__":
    main()

