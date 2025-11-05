#!/usr/bin/env python3
"""
将生成的 AMP 序列导出为 FASTA 格式

支持按活性评分、新颖性等筛选和排序
"""

import argparse
import pandas as pd
from pathlib import Path


def export_to_fasta(
    csv_path: str,
    output_path: str,
    min_activity: float = 0.0,
    min_novelty: float = 0.0,
    max_sequences: int = None,
    sort_by: str = "activity_score",
    include_metadata: bool = True
):
    """
    将 CSV 导出为 FASTA 格式
    
    Args:
        csv_path: 输入 CSV 文件路径
        output_path: 输出 FASTA 文件路径
        min_activity: 最小活性评分（筛选条件）
        min_novelty: 最小新颖性（NN距离）
        max_sequences: 最大输出序列数
        sort_by: 排序依据列名
        include_metadata: 是否在 header 中包含详细信息
    """
    
    # 读取数据
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total sequences: {len(df)}")
    
    # 筛选
    if min_activity > 0:
        df = df[df['activity_score'] >= min_activity]
        print(f"After activity filter (>={min_activity}): {len(df)}")
    
    if min_novelty > 0:
        df = df[df['nn_distance'] >= min_novelty]
        print(f"After novelty filter (>={min_novelty}): {len(df)}")
    
    # 排序
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
        print(f"Sorted by {sort_by} (descending)")
    
    # 限制数量
    if max_sequences and len(df) > max_sequences:
        df = df.head(max_sequences)
        print(f"Limited to top {max_sequences} sequences")
    
    # 导出 FASTA
    print(f"\nExporting to {output_path}...")
    with open(output_path, 'w') as f:
        for idx, row in df.iterrows():
            # 构建 header
            if include_metadata:
                # 详细版：包含所有关键信息
                header = (
                    f">AMP_{idx:04d} | "
                    f"score={row['activity_score']:.3f} | "
                    f"charge={row['charge']:.1f} | "
                    f"hydro={row['hydrophobicity']:.2f} | "
                    f"len={int(row['length'])} | "
                    f"novelty={row['nn_distance']:.3f} | "
                    f"source={row['nn_source']}"
                )
            else:
                # 简洁版：仅ID和评分
                header = f">AMP_{idx:04d}_score{row['activity_score']:.3f}"
            
            # 写入
            f.write(header + '\n')
            f.write(row['sequence'] + '\n')
    
    print(f"✅ Exported {len(df)} sequences to {output_path}")
    
    # 统计信息
    print(f"\n=== Export Statistics ===")
    print(f"Sequences exported: {len(df)}")
    print(f"Activity score: {df['activity_score'].mean():.3f} ± {df['activity_score'].std():.3f}")
    print(f"  Range: [{df['activity_score'].min():.3f}, {df['activity_score'].max():.3f}]")
    print(f"Average length: {df['length'].mean():.1f} ± {df['length'].std():.1f}")
    print(f"Average charge: {df['charge'].mean():.2f} ± {df['charge'].std():.2f}")
    print(f"Average novelty: {df['nn_distance'].mean():.4f} ± {df['nn_distance'].std():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Export generated AMP sequences to FASTA format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 导出所有序列
  python export_fasta.py --input outputs/analysis/generated_amps_analyzed.csv \\
                         --output generated_amps.fasta
  
  # 导出高活性序列（Top 50）
  python export_fasta.py --input outputs/analysis/generated_amps_analyzed.csv \\
                         --output top50_amps.fasta \\
                         --min-activity 0.8 \\
                         --max-sequences 50
  
  # 导出高新颖性序列
  python export_fasta.py --input outputs/analysis/generated_amps_analyzed.csv \\
                         --output novel_amps.fasta \\
                         --min-novelty 0.6 \\
                         --max-sequences 100
  
  # 简洁格式（仅ID和评分）
  python export_fasta.py --input outputs/analysis/generated_amps_analyzed.csv \\
                         --output simple_amps.fasta \\
                         --simple
        """
    )
    
    # 输入输出
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input CSV file (analyzed results)")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output FASTA file")
    
    # 筛选条件
    parser.add_argument("--min-activity", type=float, default=0.0,
                        help="Minimum activity score (default: 0.0)")
    parser.add_argument("--min-novelty", type=float, default=0.0,
                        help="Minimum novelty (NN distance) (default: 0.0)")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Maximum number of sequences to export (default: all)")
    
    # 排序
    parser.add_argument("--sort-by", type=str, default="activity_score",
                        choices=["activity_score", "charge", "hydrophobicity", 
                                "length", "nn_distance", "boman_index"],
                        help="Sort sequences by (default: activity_score)")
    
    # 格式
    parser.add_argument("--simple", action="store_true",
                        help="Use simple header format (ID and score only)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 导出
    export_to_fasta(
        csv_path=args.input,
        output_path=args.output,
        min_activity=args.min_activity,
        min_novelty=args.min_novelty,
        max_sequences=args.max_sequences,
        sort_by=args.sort_by,
        include_metadata=not args.simple
    )


if __name__ == "__main__":
    main()



