#!/usr/bin/env python3
"""
Deduplicate sequences using CD-HIT or pure Python fallback.

CD-HIT is the gold standard, but we provide a simple fallback.
"""

import subprocess
import shutil
import pandas as pd
from pathlib import Path
from Bio import SeqIO
import sys

# Quality control parameters (must match 01_parse_fasta.py)
MIN_AMP_LENGTH = 10   # Minimum peptide length (aa)
MAX_AMP_LENGTH = 100  # Maximum peptide length (aa)


def run_cdhit(input_fasta: Path, output_fasta: Path, identity: float = 0.9):
    """Run CD-HIT for deduplication."""
    
    # Check if CD-HIT is installed
    if not shutil.which('cd-hit'):
        print("❌ CD-HIT not found. Install with: conda install -c bioconda cd-hit")
        print("   Or: brew install cd-hit (macOS)")
        print("\n   Falling back to simple deduplication...")
        return False
    
    cmd = [
        'cd-hit',
        '-i', str(input_fasta),
        '-o', str(output_fasta),
        '-c', str(identity),  # sequence identity threshold
        '-n', '5',            # word length
        '-M', '8000',         # memory limit (MB)
        '-T', '0',            # use all CPU cores
        '-d', '0',            # unlimited length of description
    ]
    
    print(f"🔧 Running CD-HIT (identity threshold: {identity})...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ CD-HIT completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ CD-HIT failed: {e}")
        print(e.stderr)
        return False


def simple_deduplicate(input_fasta: Path, output_fasta: Path):
    """Simple exact-match deduplication (fallback)."""
    
    print("🔧 Running simple deduplication (exact matches only)...")
    
    seen_sequences = set()
    unique_records = []
    duplicates = 0
    
    with open(input_fasta, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq = str(record.seq).upper()
            
            if seq not in seen_sequences:
                seen_sequences.add(seq)
                unique_records.append(record)
            else:
                duplicates += 1
    
    # Write deduplicated FASTA
    with open(output_fasta, 'w') as f:
        SeqIO.write(unique_records, f, 'fasta')
    
    print(f"  Removed {duplicates} exact duplicates")
    print(f"  Kept {len(unique_records)} unique sequences")


def fasta_to_csv(fasta_path: Path, csv_path: Path):
    """Convert FASTA to CSV with metadata."""
    
    records = []
    with open(fasta_path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            # Parse header: idx|source|original_id
            parts = record.id.split('|')
            idx = parts[0] if len(parts) > 0 else record.id
            source = parts[1] if len(parts) > 1 else 'unknown'
            original_id = parts[2] if len(parts) > 2 else record.id
            
            records.append({
                'idx': idx,
                'id': original_id,
                'sequence': str(record.seq),
                'source': source,
                'length': len(record.seq),
            })
    
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to {csv_path}")
    
    return df


def main():
    # Paths
    data_processed = Path('data/processed')
    input_fasta = data_processed / 'raw_sequences.fasta'
    output_fasta = data_processed / 'dedup_sequences.fasta'
    output_csv = data_processed / 'dedup_sequences.csv'
    
    if not input_fasta.exists():
        print(f"❌ Input file not found: {input_fasta}")
        print("   Run 01_parse_fasta.py first")
        sys.exit(1)
    
    # Try CD-HIT first, fallback to simple dedup
    success = run_cdhit(input_fasta, output_fasta, identity=0.9)
    
    if not success:
        simple_deduplicate(input_fasta, output_fasta)
    
    # Convert to CSV
    df = fasta_to_csv(output_fasta, output_csv)
    
    # Double-check: Filter by length (safety check)
    initial_count = len(df)
    df = df[(df['length'] >= MIN_AMP_LENGTH) & (df['length'] <= MAX_AMP_LENGTH)]
    filtered_count = initial_count - len(df)
    
    if filtered_count > 0:
        print(f"\n⚠️  Removed {filtered_count} sequences outside length range [{MIN_AMP_LENGTH}, {MAX_AMP_LENGTH}]")
        # Re-save filtered data
        df.to_csv(output_csv, index=False)
        with open(output_fasta, 'w') as f:
            for _, row in df.iterrows():
                f.write(f">{row['idx']}|{row['source']}|{row['id']}\n")
                f.write(f"{row['sequence']}\n")
    
    print(f"\n📊 Final dataset:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Length range: [{df['length'].min()}, {df['length'].max()}] aa")
    print(f"  Mean length: {df['length'].mean():.1f} ± {df['length'].std():.1f} aa")
    print(f"\n  ✅ Quality control: MIN={MIN_AMP_LENGTH}, MAX={MAX_AMP_LENGTH} aa")
    print(f"\n  By source:")
    print(df['source'].value_counts())
    
    print(f"\n✅ Deduplication complete!")


if __name__ == '__main__':
    main()

