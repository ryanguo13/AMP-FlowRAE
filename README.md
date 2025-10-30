# AMP-FlowRAE: Controllable Antimicrobial Peptide Generation System

**A Flow Matching-based Antimicrobial Peptide (AMP) Design System**

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)]()

---

## 🎯 Project Overview

AMP-FlowRAE is an end-to-end deep generative model system for designing novel antimicrobial peptides with specified physicochemical properties.

**Core Features:**
- ✅ Controllable Generation: Generate AMPs based on charge, hydrophobicity, and length
- ✅ High Novelty: 75% of generated sequences are distant from training set
- ✅ Quality Assessment: Automatic activity prediction with visualization
- ✅ Practical Output: FASTA format, ready for experimental validation

---

## 📊 Project Results

### Performance Metrics

| Metric | Result | Status |
|--------|--------|--------|
| **RAE Reconstruction** | MSE = 0.086 | ✅ Excellent |
| **Flow Matching** | Val Loss = 0.225 | ✅ Excellent |
| **Charge Control** | r = 0.661 | ✅ Strong |
| **Unique Sequence Rate** | 89.2% | ✅ Exceeded |
| **High Activity Rate** | 61.0% (>0.8) | ✅ Good |
| **High Novelty Rate** | 75.2% (>0.5) | ✅ Exceeded |

### Generation Examples

**Top 5 Generated AMP Sequences:**

| Sequence | Activity Score | Charge | Length | Novelty |
|----------|---------------|--------|--------|---------|
| GLLGPLLKIAAKVGKNLL | 0.948 | +3 | 18 | 0.711 |
| FLGALWKVAKKVF | 0.948 | +3 | 13 | 0.565 |
| KKKKLVLAFLFFF | 0.944 | +4 | 13 | 0.648 |
| TNWKKIGKC... | 0.942 | +3 | 41 | 0.604 |
| FMGGLIKAA... | 0.941 | +4 | 24 | 0.539 |

---

## 🏗️ System Architecture

```
User-specified conditions (charge, hydrophobicity, length)
           ↓
[Flow Matching] Generate latent vectors (64-dim)
           ↓
[RAE Decoder] Decode to ESM-3 embeddings (1536-dim)
           ↓
[Nearest Neighbor] Match to real sequences
           ↓
[Property Calculator] Compute physicochemical properties
           ↓
[Activity Predictor] Predict activity scores
           ↓
FASTA Output → Experimental Validation
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/ryanguo13/AMP-FlowRAE
cd AMP-FlowRAE

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate AMP Sequences

```bash
# Step 1: Generate latent vectors
python flow_matching/sample.py \
  --num-samples 100 \
  --charge-range 3 8 \
  --length-range 15 30 \
  --output-dir outputs/my_generation

# Step 2: Decode to sequences
python decoder/decode.py \
  --latents outputs/my_generation/sampled_latents.npy \
  --rae-checkpoint outputs/rae/checkpoints/best.pt \
  --output outputs/my_sequences.csv

# Step 3: Quality analysis
python scripts/09_analyze_generated.py \
  --input outputs/my_sequences.csv \
  --output-dir outputs/my_analysis

# Step 4: Export FASTA
python scripts/export_fasta.py \
  --input outputs/my_analysis/generated_amps_analyzed.csv \
  --output outputs/my_amps.fasta \
  --min-activity 0.8 \
  --max-sequences 20
```

### 3. View Results

```bash
# View sequences
head outputs/my_amps.fasta

# View visualizations
open outputs/my_analysis/*.png
```

---

## 📁 Project Structure

```
AMP-FlowRAE/
├── data/                    # Data
│   ├── embeddings/          # ESM-3 embeddings
│   └── splits/              # Train/Val/Test splits
│
├── rae/                     # Phase 2: RAE Model
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── flow_matching/           # Phase 3: Flow Matching
│   ├── model.py
│   ├── train.py
│   ├── sample.py
│   └── evaluate_samples.py
│
├── decoder/                 # Phase 4: Sequence Decoding
│   ├── nearest_neighbor.py
│   └── decode.py
│
├── utils/                   # Utility Functions
│   └── sequence_properties.py
│
├── scripts/                 # Data Processing Scripts
│   ├── 09_analyze_generated.py
│   └── export_fasta.py
│
├── outputs/                 # Output Files
│   ├── rae/                 # RAE Model
│   ├── flow/                # Flow Model
│   ├── analysis/            # Analysis Results
│   └── *.fasta              # FASTA Files
│
└── docs/                    # Documentation
    ├── PHASE1_COMPLETE.md
    ├── PHASE2_COMPLETE.md
    ├── PHASE3_COMPLETE.md
    ├── PHASE4_COMPLETE.md
    └── STATUS.md
```

---

## 📚 Documentation

- [**Project Summary**](SUMMARY.md) - Complete technical summary
- [**Phase 1-4 Documents**](docs/) - Detailed reports for each phase
- [**Quick Start Guide**](QUICKSTART_EXPORT.md) - FASTA export tutorial
- [**Project Status**](docs/STATUS.md) - Current progress

---

## 🔬 Technical Highlights

### 1. Conditional Flow Matching

- **Innovation:** Uses Flow Matching instead of traditional VAE/GAN
- **Advantages:** Stable training, high generation quality
- **Conditional Control:** Supports charge, hydrophobicity, and length

### 2. Nearest Neighbor Decoding

- **Method:** k-NN search based on cosine distance
- **Advantages:** Simple, fast, guarantees valid sequences
- **Trade-off:** Novelty vs reliability

### 3. Heuristic Activity Prediction

- **Based on:** Weighted scoring of physicochemical properties
- **Advantages:** No training required, highly interpretable
- **Usage:** Fast screening of candidate sequences

---

## 📈 Application Scenarios

### 1. Drug Discovery

- Generate novel antimicrobial peptide candidates
- Optimize physicochemical properties of existing AMPs
- Explore new sequence space

### 2. Scientific Research

- Study sequence-function relationships
- Validate generative models in biology
- Provide datasets for other research

### 3. Educational Demonstration

- Deep generative model case study
- Bioinformatics applications
- End-to-end machine learning pipeline

---

## ⚠️ Important Notes

### Limitations

1. **Activity Predictions are Estimates**
   - Heuristic scoring based on physicochemical properties
   - Requires experimental validation

2. **Simple Decoding Strategy**
   - Relies on nearest neighbor search
   - Not true "generation"

3. **Imperfect Conditional Control**
   - Weak hydrophobicity control (r=0.150)
   - Can be improved by increasing model capacity

### Recommendations

- ✅ Use for **candidate screening** and **preliminary exploration**
- ✅ Combine with experimental validation
- ⚠️ Do not use directly for clinical applications

---

## 🎓 Citation

If you use this project, please cite:

```bibtex
@software{amp_flowrae_2024,
  title = {AMP-FlowRAE: Controllable Antimicrobial Peptide Generation},
  year = {2024},
  author = {Ryan Guo},
  url = {https://github.com/ryanguo13/AMP-FlowRAE}
}
```

---

## 📞 Contact

- **Project Homepage:** https://github.com/ryanguo13/AMP-FlowRAE
- **Issue Tracker:** [Issues](https://github.com/ryanguo13/AMP-FlowRAE/issues)
- **Email:** your.email@example.com

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- **ESM-3** - Meta AI's protein language model
- **Flow Matching** - Novel approach to generative modeling
- **DRAMP & APD** - Antimicrobial peptide databases

---

**Let's advance antimicrobial peptide drug discovery together! 🚀**

_Last updated: 2024-10-30_
