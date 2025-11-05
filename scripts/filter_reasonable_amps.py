#!/usr/bin/env python3
"""
过滤生成的 AMP 序列，只保留合理长度的序列

合理的 AMP 通常：
- 长度：10-100 aa (大多数在 15-50)
- 有足够的电荷和疏水性
"""

import argparse
import pandas as pd
from pathlib import Path


def filter_reasonable_amps(
    input_csv: str,
    output_csv: str,
    min_length: int = 10,
    max_length: int = 100,
    min_activity: float = 0.0
):
    """
    过滤合理的 AMP 序列
    
    Args:
        input_csv: 输入 CSV 文件
        output_csv: 输出 CSV 文件
        min_length: 最小长度
        max_length: 最大长度
        min_activity: 最小活性评分
    """
    
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"\n=== 原始数据统计 ===")
    print(f"总序列数: {len(df)}")
    print(f"长度范围: [{df['length'].min():.0f}, {df['length'].max():.0f}]")
    print(f"活性范围: [{df['activity_score'].min():.3f}, {df['activity_score'].max():.3f}]")
    
    # 长度分布
    print(f"\n长度分布：")
    print(f"  <= 4:  {(df['length'] <= 4).sum():4d} ({(df['length'] <= 4).sum()/len(df)*100:5.1f}%)")
    print(f"  5-9:   {((df['length'] >= 5) & (df['length'] <= 9)).sum():4d} ({((df['length'] >= 5) & (df['length'] <= 9)).sum()/len(df)*100:5.1f}%)")
    print(f"  10-14: {((df['length'] >= 10) & (df['length'] <= 14)).sum():4d} ({((df['length'] >= 10) & (df['length'] <= 14)).sum()/len(df)*100:5.1f}%)")
    print(f"  15-30: {((df['length'] >= 15) & (df['length'] <= 30)).sum():4d} ({((df['length'] >= 15) & (df['length'] <= 30)).sum()/len(df)*100:5.1f}%)")
    print(f"  31-50: {((df['length'] >= 31) & (df['length'] <= 50)).sum():4d} ({((df['length'] >= 31) & (df['length'] <= 50)).sum()/len(df)*100:5.1f}%)")
    print(f"  > 50:  {(df['length'] > 50).sum():4d} ({(df['length'] > 50).sum()/len(df)*100:5.1f}%)")
    
    # 过滤
    print(f"\n=== 应用过滤条件 ===")
    print(f"最小长度: {min_length}")
    print(f"最大长度: {max_length}")
    print(f"最小活性: {min_activity}")
    
    filtered_df = df[
        (df['length'] >= min_length) &
        (df['length'] <= max_length) &
        (df['activity_score'] >= min_activity)
    ].copy()
    
    print(f"\n=== 过滤后统计 ===")
    print(f"保留序列数: {len(filtered_df)} / {len(df)} ({len(filtered_df)/len(df)*100:.1f}%)")
    print(f"去除序列数: {len(df) - len(filtered_df)}")
    print(f"长度范围: [{filtered_df['length'].min():.0f}, {filtered_df['length'].max():.0f}]")
    print(f"平均长度: {filtered_df['length'].mean():.1f} ± {filtered_df['length'].std():.1f}")
    print(f"活性范围: [{filtered_df['activity_score'].min():.3f}, {filtered_df['activity_score'].max():.3f}]")
    print(f"平均活性: {filtered_df['activity_score'].mean():.3f} ± {filtered_df['activity_score'].std():.3f}")
    
    # 质量统计
    print(f"\n=== 质量统计 ===")
    print(f"高活性 (>0.8): {(filtered_df['activity_score'] > 0.8).sum()} ({(filtered_df['activity_score'] > 0.8).sum()/len(filtered_df)*100:.1f}%)")
    print(f"中活性 (0.6-0.8): {((filtered_df['activity_score'] > 0.6) & (filtered_df['activity_score'] <= 0.8)).sum()} ({((filtered_df['activity_score'] > 0.6) & (filtered_df['activity_score'] <= 0.8)).sum()/len(filtered_df)*100:.1f}%)")
    print(f"高新颖性 (>0.5): {(filtered_df['nn_distance'] > 0.5).sum()} ({(filtered_df['nn_distance'] > 0.5).sum()/len(filtered_df)*100:.1f}%)")
    
    # 保存
    filtered_df.to_csv(output_csv, index=False)
    print(f"\n✅ 保存到: {output_csv}")
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Filter reasonable AMP sequences by length and quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 过滤掉超短序列（< 10 aa）
  python filter_reasonable_amps.py \\
    --input outputs/analysis/generated_amps_analyzed.csv \\
    --output outputs/analysis/generated_amps_filtered.csv \\
    --min-length 10
  
  # 只保留典型长度的高质量 AMP
  python filter_reasonable_amps.py \\
    --input outputs/analysis/generated_amps_analyzed.csv \\
    --output outputs/analysis/generated_amps_filtered.csv \\
    --min-length 15 \\
    --max-length 50 \\
    --min-activity 0.7
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input CSV file")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output CSV file")
    parser.add_argument("--min-length", type=int, default=10,
                        help="Minimum peptide length (default: 10)")
    parser.add_argument("--max-length", type=int, default=100,
                        help="Maximum peptide length (default: 100)")
    parser.add_argument("--min-activity", type=float, default=0.0,
                        help="Minimum activity score (default: 0.0)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 过滤
    filter_reasonable_amps(
        input_csv=args.input,
        output_csv=args.output,
        min_length=args.min_length,
        max_length=args.max_length,
        min_activity=args.min_activity
    )


if __name__ == "__main__":
    main()



