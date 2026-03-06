#!/usr/bin/env python3
"""
Parse sequence data from FASTA, TXT, XLSX, and CSV in data/raw and unify to CSV/FASTA.

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


def _detect_sequence_column(df: pd.DataFrame) -> str | None:
    """Heuristically detect the sequence column name in a tabular file."""
    candidate_names = [
        'sequence', 'Sequence', 'SEQUENCE',
        'seq', 'Seq', 'SEQ',
        'peptide', 'Peptide', 'PEPTIDE',
        'AASequence', 'aa_sequence', 'amino_acid_sequence'
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for name in candidate_names:
        if name.lower() in lower_cols:
            return lower_cols[name.lower()]
    # fallback: first object dtype column with AAs-only values
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50).str.upper()
            if len(sample) == 0:
                continue
            if sample.apply(lambda s:  MIN_AMP_LENGTH <= len(s) <= MAX_AMP_LENGTH and all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in s)).mean() > 0.5:
                return col
    return None


def parse_txt_file(txt_path: Path, source: str) -> list[dict]:
    """Parse a TXT file with one sequence per line."""
    records = []
    skipped = 0
    record_idx = 0
    
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                seq = line.strip().upper()
                if not seq:  # Skip empty lines
                    continue
                
                # Apply same QC as FASTA
                if len(seq) < MIN_AMP_LENGTH or len(seq) > MAX_AMP_LENGTH:
                    skipped += 1
                    continue
                if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
                    skipped += 1
                    continue
                
                record_idx += 1
                records.append({
                    'id': f"{txt_path.stem}_{record_idx}",
                    'sequence': seq,
                    'source': source,
                    'length': len(seq),
                    'description': ''
                })
    except Exception as e:
        print(f"    ❌ Failed to read {txt_path.name}: {e}")
        return []
    
    if skipped > 0:
        print(f"    ⚠️  Skipped {skipped} sequences (QC failed)")
    
    return records


def parse_tabular_file(path: Path, source: str) -> list[dict]:
    """Parse a CSV/XLSX file and return list of records with QC applied."""
    try:
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        else:
            # Requires openpyxl
            df = pd.read_excel(path)
    except Exception as e:
        print(f"    ❌ Failed to read {path.name}: {e}")
        return []

    seq_col = _detect_sequence_column(df)
    if seq_col is None:
        print(f"    ⚠️  No sequence column detected in {path.name}; skipped")
        return []

    id_col = None
    for cand in ['id', 'ID', 'Id', 'name', 'Name', 'accession', 'Accession']:
        if cand in df.columns:
            id_col = cand
            break

    records = []
    skipped = 0
    for _, row in df.iterrows():
        seq_raw = str(row[seq_col]) if not pd.isna(row[seq_col]) else ''
        seq = seq_raw.upper().strip()
        if len(seq) < MIN_AMP_LENGTH or len(seq) > MAX_AMP_LENGTH:
            skipped += 1
            continue
        if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
            skipped += 1
            continue
        rec_id = str(row[id_col]) if id_col and not pd.isna(row.get(id_col)) else f"{path.stem}_{_}"
        records.append({
            'id': rec_id,
            'sequence': seq,
            'source': source,
            'length': len(seq),
            'description': ''
        })

    if skipped > 0:
        print(f"    ⚠️  Skipped {skipped} sequences (QC failed)")
    return records


def main():
    # Paths
    data_raw = Path('data/raw')
    data_processed = Path('data/processed')
    data_processed.mkdir(exist_ok=True, parents=True)
    
    # Collect inputs
    all_records = []

    # 1) Known FASTA sources (optional, keep for backwards compatibility)
    fasta_files = {
        'dramp_Antimicrobial_amps.fasta': 'DRAMP_antimicrobial',
        'dramp_general_amps.fasta': 'DRAMP_general',
        'naturalAMPs_APD2024a.fasta': 'APD',
        'non-amp.fasta': 'non-AMP'
    }
    print("🧬 Parsing FASTA files (predefined names)...")
    for filename, source in fasta_files.items():
        fasta_path = data_raw / filename
        if not fasta_path.exists():
            continue
        print(f"  - {filename} ({source})...")
        records = parse_fasta_file(fasta_path, source)
        all_records.extend(records)
        print(f"    ✓ {len(records)} sequences")

    # 2) Any additional FASTA files in data/raw
    print("🧬 Scanning for additional FASTA files...")
    for p in sorted(data_raw.glob('*.fasta')):
        if p.name in fasta_files:
            continue
        source = p.stem
        print(f"  - {p.name} ({source})...")
        records = parse_fasta_file(p, source)
        all_records.extend(records)
        print(f"    ✓ {len(records)} sequences")

    # 3) TXT files (one sequence per line)
    print("📝 Parsing TXT files (one sequence per line)...")
    for p in sorted(data_raw.glob('*.txt')):
        source = p.stem
        print(f"  - {p.name} ({source})...")
        records = parse_txt_file(p, source)
        all_records.extend(records)
        print(f"    ✓ {len(records)} sequences")

    # 4) Tabular files (XLSX/CSV)
    print("📄 Parsing XLSX/CSV files...")
    for p in sorted(list(data_raw.glob('*.xlsx')) + list(data_raw.glob('*.csv'))):
        source = p.stem
        print(f"  - {p.name} ({source})...")
        records = parse_tabular_file(p, source)
        all_records.extend(records)
        print(f"    ✓ {len(records)} sequences")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    if len(df) == 0:
        print("❌ No sequences found. Ensure data/raw contains FASTA/XLSX/CSV with a sequence column.")
        sys.exit(1)
    
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

