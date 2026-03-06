#!/usr/bin/env python3
"""
生成序列后过滤脚本

三重过滤：
1. MMseqs2/BLAST 同源性过滤（identity ≥ 75% 或 coverage ≥ 80% 剔除）
2. APOE motif 正则过滤
3. 可选：蛋白酶切位点过滤
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm


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


def write_fasta(path, headers, sequences):
    """写入 FASTA 文件"""
    with open(path, 'w') as f:
        for h, s in zip(headers, sequences):
            f.write(f">{h}\n{s}\n")


def filter_with_mmseqs(query_fasta, db_path, output_dir, identity_threshold=0.75, coverage_threshold=0.80):
    """
    使用 MMseqs2 进行同源性过滤
    
    Args:
        query_fasta: 查询序列 FASTA 文件
        db_path: MMseqs2 数据库路径
        output_dir: 输出目录
        identity_threshold: identity 阈值（≥此值则剔除）
        coverage_threshold: coverage 阈值（≥此值则剔除）
    
    Returns:
        filtered_headers: 过滤后的 headers
        filtered_sequences: 过滤后的 sequences
        hit_info: 命中信息字典 {header: (identity, coverage)}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🔍 Running MMseqs2 search...")
    print(f"   Query: {query_fasta}")
    print(f"   Database: {db_path}")
    
    # 创建查询数据库
    query_db = output_dir / "query_db"
    result_db = output_dir / "result_db"
    result_tsv = output_dir / "result.tsv"
    
    # 清理旧文件
    for f in [query_db, result_db, result_tsv]:
        if f.exists():
            subprocess.run(["rm", "-rf", str(f)], check=False)
    
    # 1. 创建查询数据库
    print("   Step 1: Creating query database...")
    subprocess.run([
        "mmseqs/bin/mmseqs", "createdb",
        str(query_fasta),
        str(query_db)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. 搜索
    print("   Step 2: Searching database...")
    subprocess.run([
        "mmseqs/bin/mmseqs", "search",
        str(query_db),
        str(db_path),
        str(result_db),
        str(output_dir / "tmp"),
        "--min-seq-id", str(identity_threshold),
        "--cov-mode", "1",  # coverage mode: query coverage
        "--c", str(coverage_threshold),
        "--threads", "16"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. 转换为 TSV
    print("   Step 3: Converting results...")
    subprocess.run([
        "mmseqs/bin/mmseqs", "convertalis",
        str(query_db),
        str(db_path),
        str(result_db),
        str(result_tsv),
        "--format-output", "query,target,pident,qcov"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 4. 读取结果
    print("   Step 4: Parsing results...")
    hit_headers = set()
    hit_info = {}
    
    if result_tsv.exists():
        with open(result_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    query = parts[0]
                    identity = float(parts[2])
                    coverage = float(parts[3])
                    
                    # 检查是否超过阈值
                    if identity >= identity_threshold * 100 or coverage >= coverage_threshold * 100:
                        hit_headers.add(query)
                        hit_info[query] = (identity, coverage)
    
    print(f"   Found {len(hit_headers)} sequences with high identity/coverage")
    
    # 5. 过滤序列
    headers, sequences = read_fasta(query_fasta)
    filtered_headers = []
    filtered_sequences = []
    
    for h, s in zip(headers, sequences):
        # 提取 header ID（MMseqs 使用 header 的第一部分作为 ID）
        header_id = h.split()[0] if ' ' in h else h
        
        if header_id not in hit_headers:
            filtered_headers.append(h)
            filtered_sequences.append(s)
    
    print(f"   Filtered: {len(filtered_headers)}/{len(headers)} sequences passed")
    
    return filtered_headers, filtered_sequences, hit_info


def filter_apoe_motif(headers, sequences):
    """
    使用正则表达式过滤 APOE motif
    
    Motif patterns:
    - LR.KLRK.LLR (LRXKLRKXLLR)
    - RLAVY + PLVEDM 附近 15 aa 内
    """
    print(f"🔍 Filtering APOE motifs...")
    
    # Pattern 1: LRXKLRKXLLR
    pattern1 = re.compile(r'LR.KLRK.LLR', re.IGNORECASE)
    
    # Pattern 2: RLAVY 和 PLVEDM 在 15 aa 内
    pattern2a = re.compile(r'RLAVY', re.IGNORECASE)
    pattern2b = re.compile(r'PLVEDM', re.IGNORECASE)
    
    filtered_headers = []
    filtered_sequences = []
    removed_count = 0
    
    for h, s in zip(headers, sequences):
        removed = False
        
        # 检查 Pattern 1
        if pattern1.search(s):
            removed = True
        
        # 检查 Pattern 2
        if not removed:
            match_a = pattern2a.search(s)
            match_b = pattern2b.search(s)
            if match_a and match_b:
                pos_a = match_a.start()
                pos_b = match_b.start()
                if abs(pos_a - pos_b) <= 15:
                    removed = True
        
        if not removed:
            filtered_headers.append(h)
            filtered_sequences.append(s)
        else:
            removed_count += 1
    
    print(f"   Removed {removed_count} sequences with APOE motifs")
    print(f"   Remaining: {len(filtered_headers)}/{len(headers)} sequences")
    
    return filtered_headers, filtered_sequences


def filter_protease_sites(headers, sequences, max_sites=3):
    """
    过滤蛋白酶切位点（KR/KK/RR）
    
    Args:
        headers: headers 列表
        sequences: sequences 列表
        max_sites: 最大允许的切位点数量（超过则剔除）
    """
    print(f"🔍 Filtering protease cleavage sites (KR/KK/RR)...")
    
    # 蛋白酶切位点模式
    patterns = [
        re.compile(r'KR', re.IGNORECASE),
        re.compile(r'KK', re.IGNORECASE),
        re.compile(r'RR', re.IGNORECASE),
    ]
    
    filtered_headers = []
    filtered_sequences = []
    removed_count = 0
    
    for h, s in zip(headers, sequences):
        total_sites = 0
        for pattern in patterns:
            total_sites += len(pattern.findall(s))
        
        if total_sites <= max_sites:
            filtered_headers.append(h)
            filtered_sequences.append(s)
        else:
            removed_count += 1
    
    print(f"   Removed {removed_count} sequences with >{max_sites} protease sites")
    print(f"   Remaining: {len(filtered_headers)}/{len(headers)} sequences")
    
    return filtered_headers, filtered_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fasta", type=str, required=True, help="输入 FASTA 文件")
    parser.add_argument("--output-fasta", type=str, required=True, help="输出 FASTA 文件")
    parser.add_argument("--mmseqs-db", type=str, default="data/processed/filter/filterdb", help="MMseqs2 数据库路径")
    parser.add_argument("--identity-threshold", type=float, default=0.75, help="Identity 阈值")
    parser.add_argument("--coverage-threshold", type=float, default=0.80, help="Coverage 阈值")
    parser.add_argument("--filter-apoe", action="store_true", help="启用 APOE motif 过滤")
    parser.add_argument("--filter-protease", action="store_true", help="启用蛋白酶切位点过滤")
    parser.add_argument("--max-protease-sites", type=int, default=3, help="最大蛋白酶切位点数")
    parser.add_argument("--output-stats", type=str, help="输出统计信息 CSV 文件路径")
    
    args = parser.parse_args()
    
    # 读取输入序列
    print("="*60)
    print("Reading input sequences...")
    print("="*60)
    headers, sequences = read_fasta(args.input_fasta)
    print(f"Total sequences: {len(sequences)}")
    
    original_count = len(sequences)
    stats = {
        "stage": ["Original"],
        "count": [original_count]
    }
    
    # 1. MMseqs2 过滤
    print("\n" + "="*60)
    print("Stage 1: MMseqs2 homology filtering")
    print("="*60)
    filtered_headers, filtered_sequences, hit_info = filter_with_mmseqs(
        args.input_fasta,
        args.mmseqs_db,
        Path(args.output_fasta).parent / "mmseqs_temp",
        args.identity_threshold,
        args.coverage_threshold
    )
    stats["stage"].append("After MMseqs2")
    stats["count"].append(len(filtered_sequences))
    
    # 2. APOE motif 过滤
    if args.filter_apoe:
        print("\n" + "="*60)
        print("Stage 2: APOE motif filtering")
        print("="*60)
        filtered_headers, filtered_sequences = filter_apoe_motif(filtered_headers, filtered_sequences)
        stats["stage"].append("After APOE motif")
        stats["count"].append(len(filtered_sequences))
    
    # 3. 蛋白酶切位点过滤
    if args.filter_protease:
        print("\n" + "="*60)
        print("Stage 3: Protease cleavage site filtering")
        print("="*60)
        filtered_headers, filtered_sequences = filter_protease_sites(
            filtered_headers, filtered_sequences, args.max_protease_sites
        )
        stats["stage"].append("After protease sites")
        stats["count"].append(len(filtered_sequences))
    
    # 写入输出
    print("\n" + "="*60)
    print("Writing filtered sequences...")
    print("="*60)
    write_fasta(args.output_fasta, filtered_headers, filtered_sequences)
    print(f"✅ Saved {len(filtered_sequences)} sequences to {args.output_fasta}")
    
    # 保存统计信息
    if args.output_stats:
        df_stats = pd.DataFrame(stats)
        df_stats.to_csv(args.output_stats, index=False)
        print(f"✅ Statistics saved to {args.output_stats}")
    
    # 打印总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Original: {original_count}")
    print(f"Filtered: {len(filtered_sequences)}")
    print(f"Removed: {original_count - len(filtered_sequences)} ({100*(original_count-len(filtered_sequences))/original_count:.1f}%)")
    print(f"Novelty rate: {100*len(filtered_sequences)/original_count:.1f}%")


if __name__ == "__main__":
    main()


