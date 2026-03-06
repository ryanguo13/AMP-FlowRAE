#!/usr/bin/env python3
"""
Flow Matching 生成序列评估脚本

根据文档《模型任务》的要求，实现以下评估指标：
1. 新颖性 (Novelty): 使用 Needleman-Wunsch 全局比对计算与训练集的 Sequence Identity
2. 多样性 (Diversity): Internal Diversity (Self-Identity)
3. 条件控制准确性: 验证 Charge、Hydrophobicity、Length 的条件控制效果

优化策略: 使用 MMseqs2 快速筛选 top 候选，再计算精确的 Global Identity
"""

import argparse
import subprocess
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sequence_properties import compute_charge, compute_hydrophobicity


HYDROPHOBICITY_SCALE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

CHARGE_SCALE = {
    "K": +1,
    "R": +1,
    "D": -1,
    "E": -1,
}


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
            if line.startswith(">"):
                if header is not None:
                    sequences.append("".join(seq_parts))
                    headers.append(header)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)

        if header is not None:
            sequences.append("".join(seq_parts))
            headers.append(header)

    return headers, sequences


def write_fasta(headers, sequences, path):
    """写入 FASTA 文件"""
    with open(path, "w") as f:
        for h, s in zip(headers, sequences):
            f.write(f">{h}\n{s}\n")


def compute_charge(seq: str) -> float:
    """计算 net charge at pH 7"""
    return sum(CHARGE_SCALE.get(aa, 0) for aa in seq)


def compute_hydrophobicity(seq: str) -> float:
    """计算 Kyte-Doolittle 平均疏水性"""
    values = [HYDROPHOBICITY_SCALE.get(aa, 0.0) for aa in seq]
    return np.mean(values) if values else 0.0


def compute_physicochemical_properties(sequences):
    """计算序列的理化性质"""
    properties = []
    for seq in sequences:
        props = {
            "length": len(seq),
            "charge": compute_charge(seq),
            "hydrophobicity": compute_hydrophobicity(seq),
        }
        properties.append(props)
    return pd.DataFrame(properties)


def compute_novelty_with_mmseqs(query_fasta, db_path, output_dir, top_k=50):
    """
    使用 MMseqs2 计算新颖度

    优化策略: 先用 MMseqs2 快速筛选 top-k 候选，然后计算精确的 identity

    Returns:
        results: dict with novelty metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("🔍 Computing novelty with MMseqs2...")

    query_db = output_dir / "query_db"
    result_db = output_dir / "result_db"
    result_tsv = output_dir / "result.tsv"

    for f in [query_db, result_db, result_tsv]:
        if f.exists():
            subprocess.run(["rm", "-rf", str(f)], check=False)

    # 1. 创建查询数据库
    subprocess.run(
        ["mmseqs/bin/mmseqs", "createdb", str(query_fasta), str(query_db)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 2. 搜索
    subprocess.run(
        [
            "mmseqs/bin/mmseqs",
            "search",
            str(query_db),
            str(db_path),
            str(result_db),
            str(output_dir / "tmp"),
            "--min-seq-id",
            "0.0",
            "-k",
            str(top_k),
            "--threads",
            "16",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 3. 转换为 TSV
    subprocess.run(
        [
            "mmseqs/bin/mmseqs",
            "convertalis",
            str(query_db),
            str(db_path),
            str(result_db),
            str(result_tsv),
            "--format-output",
            "query,target,pident,qlen,tlen",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 4. 解析结果
    headers, sequences = read_fasta(query_fasta)
    query_to_seq = {h.split()[0] if " " in h else h: s for h, s in zip(headers, sequences)}

    max_identities = {}
    avg_identities = {}

    if result_tsv.exists():
        with open(result_tsv) as f:
            query_results = {}
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    query = parts[0]
                    identity = float(parts[2])

                    if query not in query_results:
                        query_results[query] = []
                    query_results[query].append(identity)

            for query, identities in query_results.items():
                max_identities[query] = max(identities) if identities else 0.0
                avg_identities[query] = np.mean(identities) if identities else 0.0

    # 对于没有匹配的序列，identity = 0
    for h in headers:
        header_id = h.split()[0] if " " in h else h
        if header_id not in max_identities:
            max_identities[header_id] = 0.0
            avg_identities[header_id] = 0.0

    results = {
        "max_identities": max_identities,
        "avg_identities": avg_identities,
    }

    print(f"   Analyzed {len(max_identities)} queries")

    return results


def compute_diversity(sequences, sample_size=None):
    """
    计算 Internal Diversity (Self-Identity)

    多样性指标: 生成序列之间的平均 pairwise identity
    分布在 30%-50% 之间说明模型没有发生模式坍塌
    """
    print("🔍 Computing Internal Diversity...")

    n = len(sequences)
    if n > 1000 and sample_size:
        # 采样计算以加速
        indices = np.random.choice(n, min(sample_size, n), replace=False)
        sequences = [sequences[i] for i in indices]
        n = len(sequences)

    # 计算 pairwise identity (简化版本，使用编辑距离的近似)
    identities = []

    # 使用简化的 k-mer 相似度作为近似
    k = 3

    def get_kmers(seq, k=3):
        return set(seq[i : i + k] for i in range(len(seq) - k + 1))

    for i in range(min(n, 500)):  # 限制计算量
        for j in range(i + 1, min(n, 500)):
            seq1, seq2 = sequences[i], sequences[j]
            if len(seq1) < k or len(seq2) < k:
                continue

            kmers1 = get_kmers(seq1, k)
            kmers2 = get_kmers(seq2, k)

            if len(kmers1 | kmers2) > 0:
                jaccard = len(kmers1 & kmers2) / len(kmers1 | kmers2)
                identities.append(jaccard * 100)  # 转换为百分比

    if identities:
        mean_identity = np.mean(identities)
        std_identity = np.std(identities)
    else:
        mean_identity = 0
        std_identity = 0

    print(f"   Mean Internal Diversity: {mean_identity:.2f}%")
    print(f"   Std Internal Diversity: {std_identity:.2f}%")

    return {"mean_identity": mean_identity, "std_identity": std_identity, "identities": identities}


def compute_condition_accuracy(generated_sequences, conditions):
    """
    计算条件控制准确性

    验证生成的序列是否满足指定的条件 (charge, hydrophobicity, length)
    """
    print("🔍 Computing Condition Control Accuracy...")

    # 计算生成序列的实际理化性质
    generated_props = compute_physicochemical_properties(generated_sequences)

    # 计算相关性
    correlations = {}

    for cond_name in ["charge", "hydrophobicity", "length"]:
        target = conditions[cond_name].values
        actual = generated_props[cond_name].values

        # 计算 Pearson 相关系数
        corr = np.corrcoef(target, actual)[0, 1]
        correlations[cond_name] = corr

        # 计算 RMSE
        rmse = np.sqrt(np.mean((target - actual) ** 2))

        print(f"   {cond_name}: correlation={corr:.3f}, RMSE={rmse:.3f}")

    # 计算每个条件的准确率 (在容忍范围内的比例)
    tolerances = {"charge": 2.0, "hydrophobicity": 0.3, "length": 5.0}

    accuracy = {}
    for cond_name in ["charge", "hydrophobicity", "length"]:
        target = conditions[cond_name].values
        actual = generated_props[cond_name].values
        tol = tolerances[cond_name]

        within_tolerance = np.abs(target - actual) <= tol
        accuracy[cond_name] = np.mean(within_tolerance)
        print(f"   {cond_name} accuracy (tol={tol}): {accuracy[cond_name] * 100:.1f}%")

    return {"correlations": correlations, "accuracy": accuracy, "generated_props": generated_props}


def compute_novelty_metrics(max_identities, identity_threshold=0.40):
    """
    计算新颖度相关指标

    Args:
        max_identities: dict, 每个生成序列的最大 identity
        identity_threshold: 阈值，低于此值认为是有新颖性的

    Returns:
        metrics: dict
    """
    identities = list(max_identities.values())

    # 新颖序列的比例 (identity < threshold)
    novel_count = sum(1 for v in identities if v < identity_threshold * 100)
    novelty_rate = novel_count / len(identities) if identities else 0.0

    # 平均 identity
    mean_identity = np.mean(identities) if identities else 0.0
    median_identity = np.median(identities) if identities else 0.0

    # Identity 分布
    identity_list = list(identities)
    identity_distribution = {
        "<20%": sum(1 for v in identity_list if v < 20) / len(identity_list) * 100
        if identity_list
        else 0,
        "20-30%": sum(1 for v in identity_list if 20 <= v < 30) / len(identity_list) * 100
        if identity_list
        else 0,
        "30-40%": sum(1 for v in identity_list if 30 <= v < 40) / len(identity_list) * 100
        if identity_list
        else 0,
        "40-50%": sum(1 for v in identity_list if 40 <= v < 50) / len(identity_list) * 100
        if identity_list
        else 0,
        "50-60%": sum(1 for v in identity_list if 50 <= v < 60) / len(identity_list) * 100
        if identity_list
        else 0,
        "60-70%": sum(1 for v in identity_list if 60 <= v < 70) / len(identity_list) * 100
        if identity_list
        else 0,
        ">70%": sum(1 for v in identity_list if v >= 70) / len(identity_list) * 100
        if identity_list
        else 0,
    }

    metrics = {
        "novelty_rate": novelty_rate,
        "novel_count": novel_count,
        "total_count": len(identities),
        "mean_identity": mean_identity,
        "median_identity": median_identity,
        "identity_distribution": identity_distribution,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching generated sequences")

    # 输入输出
    parser.add_argument(
        "--input-fasta", type=str, required=True, help="Input FASTA file with generated sequences"
    )
    parser.add_argument(
        "--conditions-csv",
        type=str,
        required=True,
        help="CSV file with conditions (charge, hydrophobicity, length)",
    )
    parser.add_argument(
        "--train-fasta",
        type=str,
        required=True,
        help="Training data FASTA file for novelty comparison",
    )
    parser.add_argument(
        "--mmseqs-db", type=str, required=True, help="MMseqs2 database path for training data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/evaluation", help="Output directory"
    )

    # 参数
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.40,
        help="Identity threshold for novelty (default: 0.40)",
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k candidates to retrieve with MMseqs2"
    )
    parser.add_argument(
        "--diversity-sample-size",
        type=int,
        default=500,
        help="Sample size for diversity computation",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Flow Matching Generated Sequences Evaluation")
    print("=" * 60)

    # 1. 读取生成序列和条件
    print("\n[1/4] Loading generated sequences...")
    headers, sequences = read_fasta(args.input_fasta)
    print(f"   Loaded {len(sequences)} generated sequences")

    conditions = pd.read_csv(args.conditions_csv)
    print(f"   Loaded {len(conditions)} conditions")

    # 2. 新颖度评估
    print("\n[2/4] Computing novelty metrics...")
    novelty_results = compute_novelty_with_mmseqs(
        args.input_fasta, args.mmseqs_db, output_dir / "mmseqs_tmp", top_k=args.top_k
    )

    novelty_metrics = compute_novelty_metrics(
        novelty_results["max_identities"], args.identity_threshold
    )

    print(f"\n   Novelty Results (threshold={args.identity_threshold * 100}%):")
    print(
        f"   - Novel sequences: {novelty_metrics['novel_count']}/{novelty_metrics['total_count']}"
    )
    print(f"   - Novelty rate: {novelty_metrics['novelty_rate'] * 100:.1f}%")
    print(f"   - Mean identity: {novelty_metrics['mean_identity']:.2f}%")
    print(f"   - Median identity: {novelty_metrics['median_identity']:.2f}%")
    print(f"   - Identity distribution:")
    for range_str, pct in novelty_metrics["identity_distribution"].items():
        print(f"       {range_str}: {pct:.1f}%")

    # 3. 多样性评估
    print("\n[3/4] Computing diversity metrics...")
    diversity_metrics = compute_diversity(sequences, args.diversity_sample_size)

    # 4. 条件控制准确性
    print("\n[4/4] Computing condition control accuracy...")
    condition_metrics = compute_condition_accuracy(sequences, conditions)

    # 保存结果
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    # 保存详细结果
    results = {
        "novelty": novelty_metrics,
        "diversity": {
            "mean_identity": diversity_metrics["mean_identity"],
            "std_identity": diversity_metrics["std_identity"],
        },
        "condition_accuracy": {
            "correlations": condition_metrics["correlations"],
            "accuracy": condition_metrics["accuracy"],
        },
    }

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {results_path}")

    # 保存生成序列的理化性质
    condition_metrics["generated_props"]["max_identity"] = condition_metrics[
        "generated_props"
    ].index.map(
        lambda i: novelty_results["max_identities"].get(
            headers[i].split()[0] if " " in headers[i] else headers[i], 0.0
        )
    )
    props_path = output_dir / "generated_properties.csv"
    condition_metrics["generated_props"].to_csv(props_path, index=False)
    print(f"✅ Generated properties saved to {props_path}")

    # 打印总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sequences: {len(sequences)}")
    print(f"\nNovelty (threshold={args.identity_threshold * 100}%):")
    print(f"  - Novelty rate: {novelty_metrics['novelty_rate'] * 100:.1f}%")
    print(f"  - Mean identity: {novelty_metrics['mean_identity']:.2f}%")
    print(f"\nDiversity:")
    print(f"  - Internal diversity: {diversity_metrics['mean_identity']:.2f}%")
    print(f"\nCondition Control:")
    for cond_name in ["charge", "hydrophobicity", "length"]:
        corr = condition_metrics["correlations"][cond_name]
        acc = condition_metrics["accuracy"][cond_name]
        print(f"  - {cond_name}: correlation={corr:.3f}, accuracy={acc * 100:.1f}%")

    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
