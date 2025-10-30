#!/usr/bin/env python3
"""
Parse all FASTA files and unify into a single CSV format.

Simple and direct - no over-engineering.
"""

import pandas as pd
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
import sys

# Quality control parameters
MIN_AMP_LENGTH = 10   # Minimum peptide length (aa)
MAX_AMP_LENGTH = 100  # Maximum peptide length (aa)


def parse_fasta_file(fasta_path: Path, source: str) -> list[dict]:
    """Parse a single FASTA file and return list of records."""
    records = []
    skipped = 0
    
    with open(fasta_path, 'r', encoding='utf-8', errors='ignore') as f:
        for record in SeqIO.parse(f, 'fasta'):
            try:
                # Handle potential encoding issues
                seq = str(record.seq).upper()
            except (UnicodeDecodeError, AttributeError):
                skipped += 1
                continue
            
            # Basic quality control
            if len(seq) < MIN_AMP_LENGTH or len(seq) > MAX_AMP_LENGTH:
                skipped += 1
                continue
            if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):  # Invalid AA
                skipped += 1
                continue
            
            records.append({
                'id': record.id,
                'sequence': seq,
                'source': source,
                'length': len(seq),
                'description': record.description
            })
    
    if skipped > 0:
        print(f"    ⚠️  Skipped {skipped} sequences (encoding issues)")
    
    return records


def main():
    # Paths
    data_raw = Path('data/raw')
    data_processed = Path('data/processed')
    data_processed.mkdir(exist_ok=True, parents=True)
    
    # Define source files
    fasta_files = {
        'dramp_Antimicrobial_amps.fasta': 'DRAMP_antimicrobial',
        'dramp_general_amps.fasta': 'DRAMP_general',
        'naturalAMPs_APD2024a.fasta': 'APD',
        'non-amp.fasta': 'non-AMP'
    }
    
    all_records = []
    
    print("🧬 Parsing FASTA files...")
    for filename, source in fasta_files.items():
        fasta_path = data_raw / filename
        
        if not fasta_path.exists():
            print(f"⚠️  Skipping {filename} (not found)")
            continue
        
        print(f"  - {filename} ({source})...")
        records = parse_fasta_file(fasta_path, source)
        all_records.extend(records)
        print(f"    ✓ {len(records)} sequences")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Basic statistics
    print(f"\n📊 Summary:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Length range: [{df['length'].min()}, {df['length'].max()}] aa")
    print(f"  Mean length: {df['length'].mean():.1f} ± {df['length'].std():.1f} aa")
    print(f"\n  ✅ Quality control applied:")
    print(f"     MIN_LENGTH = {MIN_AMP_LENGTH} aa")
    print(f"     MAX_LENGTH = {MAX_AMP_LENGTH} aa")
    print(f"\n  By source:")
    print(df['source'].value_counts())
    
    # Save
    output_csv = data_processed / 'raw_sequences.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved to {output_csv}")
    
    # Also save FASTA for CD-HIT
    output_fasta = data_processed / 'raw_sequences.fasta'
    with open(output_fasta, 'w') as f:
        for _, row in df.iterrows():
            # Use index as ID to avoid duplicates
            f.write(f">{row.name}|{row['source']}|{row['id']}\n")
            f.write(f"{row['sequence']}\n")
    print(f"✅ Saved to {output_fasta}")


if __name__ == '__main__':
    main()

