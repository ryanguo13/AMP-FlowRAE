# 📂 Data Processing Pipeline

## Directory Structure

```
data/
├── raw/                    # Original FASTA files
│   ├── dramp_Antimicrobial_amps.fasta
│   ├── dramp_general_amps.fasta
│   ├── naturalAMPs_APD2024a.fasta
│   └── non-amp.fasta
│
├── processed/              # Cleaned and deduplicated data
│   ├── raw_sequences.csv       # All sequences unified
│   ├── raw_sequences.fasta     # For CD-HIT input
│   ├── dedup_sequences.csv     # After deduplication
│   └── dedup_sequences.fasta   # After deduplication
│
├── embeddings/             # ESM-3 embeddings
│   ├── esm3_embeddings.npz     # Embedding matrix [N, 1280]
│   └── metadata.csv            # Sequence metadata + embedding indices
│
└── splits/                 # Train/val/test splits (created in Phase 2)
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## Processing Steps

### Step 1: Parse FASTA files

```fish
python scripts/01_parse_fasta.py
```

**Input:** `data/raw/*.fasta`  
**Output:**
- `data/processed/raw_sequences.csv`
- `data/processed/raw_sequences.fasta`

**What it does:**
- Reads all FASTA files
- Filters invalid sequences (length > 100, non-standard AAs)
- Unifies format to CSV with columns: `[id, sequence, source, length, description]`

---

### Step 2: Deduplicate sequences

```fish
python scripts/02_deduplicate.py
```

**Input:** `data/processed/raw_sequences.fasta`  
**Output:**
- `data/processed/dedup_sequences.csv`
- `data/processed/dedup_sequences.fasta`

**What it does:**
- Uses CD-HIT with 90% identity threshold (standard in AMP research)
- Falls back to simple exact-match dedup if CD-HIT not available
- Removes redundant sequences to prevent training bias

**Install CD-HIT (recommended):**
```fish
# macOS
brew install cd-hit

# Linux (conda)
conda install -c bioconda cd-hit
```

---

### Step 3: Extract ESM-3 embeddings

```fish
python scripts/03_extract_esm3.py
```

**Input:** `data/processed/dedup_sequences.csv`  
**Output:**
- `data/embeddings/esm3_embeddings.npz` (compressed numpy array)
- `data/embeddings/metadata.csv`

**What it does:**
- Loads ESM-3 model (esm3-sm-open-v1)
- Extracts sequence embeddings with batched inference
- Uses FP16 on GPU for efficiency
- Falls back to ESM-2 if ESM-3 fails

**Requirements:**
```fish
# Install ESM package
pip install fair-esm
# or for ESM-3
pip install git+https://github.com/evolutionaryscale/esm.git
```

---

## Data Statistics

After running all 3 steps, you should have:

| Dataset | Sequences | Avg Length | Sources |
|---------|-----------|------------|---------|
| Raw | ~15,000 | 25-30 aa | DRAMP, APD, non-AMP |
| Deduplicated | ~12,000 | 25-30 aa | After 90% CD-HIT |
| Embeddings | ~12,000 | 1280-dim | ESM-3 encoded |

---

## Troubleshooting

### CD-HIT not found
- Install via `brew install cd-hit` (macOS) or `conda install -c bioconda cd-hit`
- Or let the script fallback to simple deduplication (exact matches only)

### ESM model errors
- Make sure you have `fair-esm` or ESM-3 package installed
- On Mac (CPU): expect ~1-2 min for 1000 sequences
- On GPU: expect ~10-30 sec for 1000 sequences

### Out of memory
- Reduce `batch_size` in `03_extract_esm3.py`
- Default: 32 (GPU), 16 (CPU)
- Try: 8 or 4 for limited memory

---

## Next Steps

After completing Phase 1 (data preparation), proceed to:

```fish
# Phase 2: RAE 训练
python rae/train.py
```

- Create train/val/test splits in `data/splits/`
- Visualize embedding space in `analysis/`

---

**Last Updated:** 2025-10-27

