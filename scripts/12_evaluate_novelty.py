#!/usr/bin/env python3
"""
生成序列新颖度评估脚本

评估指标：
1. 新颖度：与 UniProt/DRAMP/APD3/DBAASP 的最大 identity < 70%
2. APOE 污染率：命中 APOE motif 的比例
3. 理化性质：长度、电荷、疏水性等分布
"""

import argparse
import subprocess
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sequence_properties import compute_charge, compute_hydrophobicity


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


def compute_novelty_with_mmseqs(query_fasta, db_path, output_dir, identity_threshold=0.70):
    """
    使用 MMseqs2 计算新颖度
    
    Returns:
        novelty_rate: 新颖度（identity < threshold 的比例）
        max_identities: 每个序列的最大 identity
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🔍 Computing novelty with MMseqs2...")
    
    # 创建查询数据库
    query_db = output_dir / "query_db"
    result_db = output_dir / "result_db"
    result_tsv = output_dir / "result.tsv"
    
    # 清理旧文件
    for f in [query_db, result_db, result_tsv]:
        if f.exists():
            subprocess.run(["rm", "-rf", str(f)], check=False)
    
    # 1. 创建查询数据库
    subprocess.run([
        "mmseqs/bin/mmseqs", "createdb",
        str(query_fasta),
        str(query_db)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. 搜索（使用更宽松的阈值以获取所有匹配）
    subprocess.run([
        "mmseqs/bin/mmseqs", "search",
        str(query_db),
        str(db_path),
        str(result_db),
        str(output_dir / "tmp"),
        "--min-seq-id", "0.0",  # 获取所有匹配
        "--threads", "16"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. 转换为 TSV
    subprocess.run([
        "mmseqs/bin/mmseqs", "convertalis",
        str(query_db),
        str(db_path),
        str(result_db),
        str(result_tsv),
        "--format-output", "query,target,pident"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 4. 解析结果
    headers, sequences = read_fasta(query_fasta)
    max_identities = {}
    
    if result_tsv.exists():
        with open(result_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query = parts[0]
                    identity = float(parts[2])
                    
                    if query not in max_identities or identity > max_identities[query]:
                        max_identities[query] = identity
    
    # 对于没有匹配的序列，identity = 0
    for h in headers:
        header_id = h.split()[0] if ' ' in h else h
        if header_id not in max_identities:
            max_identities[header_id] = 0.0
    
    # 计算新颖度
    novel_count = sum(1 for v in max_identities.values() if v < identity_threshold * 100)
    novelty_rate = novel_count / len(max_identities) if max_identities else 0.0
    
    print(f"   Novelty rate (identity < {identity_threshold*100}%): {novelty_rate*100:.1f}%")
    print(f"   Novel sequences: {novel_count}/{len(max_identities)}")
    
    return novelty_rate, max_identities


def compute_apoe_contamination(headers, sequences):
    """
    计算 APOE 污染率
    
    Returns:
        contamination_rate: 污染率
        contaminated_indices: 污染的序列索引
    """
    print(f"🔍 Computing APOE contamination...")
    
    # APOE motif patterns
    pattern1 = re.compile(r'LR.KLRK.LLR', re.IGNORECASE)
    pattern2a = re.compile(r'RLAVY', re.IGNORECASE)
    pattern2b = re.compile(r'PLVEDM', re.IGNORECASE)
    
    contaminated_indices = []
    
    for i, (h, s) in enumerate(zip(headers, sequences)):
        contaminated = False
        
        # Pattern 1
        if pattern1.search(s):
            contaminated = True
        
        # Pattern 2
        if not contaminated:
            match_a = pattern2a.search(s)
            match_b = pattern2b.search(s)
            if match_a and match_b:
                pos_a = match_a.start()
                pos_b = match_b.start()
                if abs(pos_a - pos_b) <= 15:
                    contaminated = True
        
        if contaminated:
            contaminated_indices.append(i)
    
    contamination_rate = len(contaminated_indices) / len(sequences) if sequences else 0.0
    
    print(f"   APOE contamination rate: {contamination_rate*100:.1f}%")
    print(f"   Contaminated sequences: {len(contaminated_indices)}/{len(sequences)}")
    
    return contamination_rate, contaminated_indices


def compute_physicochemical_properties(headers, sequences):
    """
    计算理化性质
    
    Returns:
        properties_df: 包含理化性质的 DataFrame
    """
    print(f"🔍 Computing physicochemical properties...")
    
    properties = []
    
    for h, s in zip(headers, sequences):
        props = {
            'header': h,
            'sequence': s,
            'length': len(s),
            'charge': compute_charge(s),
            'hydrophobicity': compute_hydrophobicity(s),
        }
        properties.append(props)
    
    df = pd.DataFrame(properties)
    
    print(f"   Properties computed for {len(df)} sequences")
    print(f"   Length: {df['length'].min()}-{df['length'].max()} (mean: {df['length'].mean():.1f})")
    print(f"   Charge: {df['charge'].min():.1f}-{df['charge'].max():.1f} (mean: {df['charge'].mean():.1f})")
    print(f"   Hydrophobicity: {df['hydrophobicity'].min():.2f}-{df['hydrophobicity'].max():.2f} (mean: {df['hydrophobicity'].mean():.2f})")
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fasta", type=str, required=True, help="输入 FASTA 文件")
    parser.add_argument("--mmseqs-db", type=str, default="data/processed/filter/filterdb", help="MMseqs2 数据库路径")
    parser.add_argument("--output-csv", type=str, help="输出评估结果 CSV 文件")
    parser.add_argument("--identity-threshold", type=float, default=0.70, help="新颖度 identity 阈值")
    parser.add_argument("--temp-dir", type=str, default="temp_eval", help="临时文件目录")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Evaluating Generated Sequences")
    print("="*60)
    
    # 读取序列
    print("\nReading sequences...")
    headers, sequences = read_fasta(args.input_fasta)
    print(f"Total sequences: {len(sequences)}")
    
    results = {}
    
    # 1. 新颖度评估
    print("\n" + "="*60)
    print("1. Novelty Assessment")
    print("="*60)
    novelty_rate, max_identities = compute_novelty_with_mmseqs(
        args.input_fasta,
        args.mmseqs_db,
        args.temp_dir,
        args.identity_threshold
    )
    results['novelty_rate'] = novelty_rate
    results['novel_count'] = sum(1 for v in max_identities.values() if v < args.identity_threshold * 100)
    results['total_count'] = len(max_identities)
    
    # 2. APOE 污染率
    print("\n" + "="*60)
    print("2. APOE Contamination Assessment")
    print("="*60)
    contamination_rate, contaminated_indices = compute_apoe_contamination(headers, sequences)
    results['apoe_contamination_rate'] = contamination_rate
    results['contaminated_count'] = len(contaminated_indices)
    
    # 3. 理化性质
    print("\n" + "="*60)
    print("3. Physicochemical Properties")
    print("="*60)
    properties_df = compute_physicochemical_properties(headers, sequences)
    
    # 添加新颖度和污染信息
    properties_df['max_identity'] = properties_df['header'].apply(
        lambda h: max_identities.get(h.split()[0] if ' ' in h else h, 0.0)
    )
    properties_df['is_novel'] = properties_df['max_identity'] < args.identity_threshold * 100
    properties_df['is_contaminated'] = properties_df.index.isin(contaminated_indices)
    
    # 保存结果
    if args.output_csv:
        properties_df.to_csv(args.output_csv, index=False)
        print(f"\n✅ Results saved to {args.output_csv}")
    
    # 打印总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total sequences: {results['total_count']}")
    print(f"Novel sequences (identity < {args.identity_threshold*100}%): {results['novel_count']} ({results['novelty_rate']*100:.1f}%)")
    print(f"APOE contaminated: {results['contaminated_count']} ({results['apoe_contamination_rate']*100:.1f}%)")
    print(f"\nPhysicochemical properties:")
    print(f"  Length: {properties_df['length'].min()}-{properties_df['length'].max()} (mean: {properties_df['length'].mean():.1f})")
    print(f"  Charge: {properties_df['charge'].min():.1f}-{properties_df['charge'].max():.1f} (mean: {properties_df['charge'].mean():.1f})")
    print(f"  Hydrophobicity: {properties_df['hydrophobicity'].min():.2f}-{properties_df['hydrophobicity'].max():.2f} (mean: {properties_df['hydrophobicity'].mean():.2f})")


if __name__ == "__main__":
    main()


