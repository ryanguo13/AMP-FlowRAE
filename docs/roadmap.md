

---

```markdown
# 🧬 AMP-FlowRAE Roadmap

> **Project:** AMP-FlowRAE (Antimicrobial Peptide Representation and Flow Generation Framework)  
> **Core Idea:** Combine pretrained protein LM (ESM-3), Representation Autoencoder (RAE), and Conditional Flow Matching for controllable and efficient AMP sequence generation.  
> **Compute:** 2 × NVIDIA V100-16GB (mixed precision training)  
> **Duration:** ~9 weeks

---

## 1. 🧭 Project Overview

AMP-FlowRAE aims to create a **controllable generative model for antimicrobial peptides (AMPs)** by combining:
1. **ESM-3 embeddings** (sequence + structure + function aware)
2. **RAE semantic latent space** (smooth & disentangled AMP space)
3. **Conditional Flow Matching (CFM)** (fast and controllable generation)

### Core Pipeline

```

FASTA → ESM-3 embedding → RAE encoder → z (latent)

z + conditions (activity, solubility, length) → Flow Matching → z*

z* → RAE decoder → embedding → AA sequence → ESM3/Classifier scoring

```

---

## 2. ⚙️ Technical Modules

| Module | Description | Output |
|---------|--------------|--------|
| **ESM-3 Embedding** | Extract high-dim embeddings from esm3-sm-open-v1 (frozen) | `data/embeddings/*.pt` |
| **RAE (Representation Autoencoder)** | Build smooth 64-D latent space specialized for AMP semantics | `checkpoints/rae_encoder.pt`, `rae_decoder.pt` |
| **Conditional Flow Matching** | Model p(z | c) with activity/solubility/length as conditions | `checkpoints/flow_model.pt` |
| **Decoder & Scoring** | Decode latent z* back to sequences and re-rank | Generated sequences + scores |
| **Filtering & Analysis** | Filter by novelty, charge, solubility, diversity metrics | `outputs/top_candidates.csv` |

---

## 3. 📅 Development Timeline

### **Phase 1 – Data & Embedding (Week 1-2)**
**Goal:** Clean data and extract ESM-3 embeddings.

| Task | Details | Deliverables |
|------|----------|--------------|
| Dataset aggregation | Merge DBAASP, APD, dbAMP, UniProt (short peptides < 50 aa). | `data/raw/*.csv` |
| Preprocessing | Dedup (CD-HIT 90 %), label AMP/Non-AMP, add MIC, CamSol. | `data/processed/amps.csv` |
| ESM-3 embedding extraction | Run esm3-sm-open-v1 in FP16, store pooled & token embeddings. | `data/embeddings/esm3_amp.pt` |
| Sanity checks | PCA / t-SNE visualization of embeddings. | `analysis/esm3_vis.ipynb` |

**Success Metric:** embeddings computed for ≥ 95 % of sequences; separability of AMP vs non-AMP visible in latent plots.

---

### **Phase 2 – RAE Latent Space (Week 3-4)**
**Goal:** Learn a low-dimensional, smooth latent representation of AMP features.

| Task | Details | Deliverables |
|------|----------|--------------|
| Model definition | Encoder 1280→64, Decoder 64→1280 (MLP+ResBlock). | `rae/model.py` |
| Training | MSE + λz‖z‖² + λorth ‖WWᵀ − I‖; FP16 batch 256 × 50 epochs. | `checkpoints/rae_*.pt` |
| Evaluation | Reconstruction MSE < 0.02; latent t-SNE shows attribute clustering. | `analysis/latent_vis.ipynb` |
| Latent export | Save z_train.pt, z_val.pt for flow training. | `data/latent/*.pt` |

**Success Metric:** smooth z-space with clear separability between high/low activity & solubility groups.

---

### **Phase 3 – Conditional Flow Matching (Week 5-7)**
**Goal:** Train a controllable flow model p(z | c).

| Task | Details | Deliverables |
|------|----------|--------------|
| Condition design | Normalize activity, solubility, length to [0, 1]. | — |
| Model | fθ(z,t,c) = MLP(64 + 3 + 1) → 64; Flow Matching Loss. | `flow_matching/model.py` |
| Training | batch 512, lr 1e-4, epochs 100, DDP(2 × V100), FP16. | `checkpoints/flow_model.pt` |
| Sampling script | Generate z* from given conditions. | `flow_matching/sample.py` |

**Success Metric:** correlation > 0.7 between target condition and generated z* attributes; stable few-step sampling.

---

### **Phase 4 – Decoding & Filtering (Week 8-9)**
**Goal:** Decode latent z* to sequence and filter top AMP candidates.

| Task | Details | Deliverables |
|------|----------|--------------|
| RAE decoder → embedding → AA sequence | Small Transformer decoder (2 layers, top-p sampling). | `decode/decoder.py` |
| Scoring | Use ESM-3 scorer + CamSol / Protein-Sol for activity/solubility. | `decode/scorer.py` |
| Filtering | Charge > +2, CamSol > 0.6, identity < 80 %. | `decode/filter.py` |
| Analysis | Novelty & diversity metrics, embedding distance plots. | `analysis/diversity_metrics.py` |

**Success Metric:** > 50 % of generated sequences predicted as AMP (score > 0.8) and CamSol > 0.7.

---

## 4. 📈 Evaluation & Metrics

| Dimension | Metric | Target Value | Tool |
|------------|---------|--------------|------|
| Reconstruction | RAE MSE | < 0.02 | PyTorch |
| Latent Smoothness | Jacobian regularity | Low variance | RAE analysis |
| Condition Correlation | Corr(z*, c) | > 0.7 | CFM analysis |
| Generation Speed | z sampling time | < 1 ms | flow benchmark |
| Novelty | Seq identity | < 80 % | CD-HIT |
| Activity | Classifier score | > 0.8 | ESM head |
| Solubility | CamSol score | > 0.7 | CamSol API |
| Diversity | Shannon entropy | ↑ high | custom script |

---

## 5. 🧰 Engineering Practices

| Area | Technique |
|-------|-----------|
| **Precision** | FP16 mixed precision training (Apex / PyTorch AMP) |
| **Distributed Training** | DDP + DeepSpeed Stage 2 for Flow Matching |
| **Checkpointing** | RAE and Flow save every 5 epochs + wandb logging |
| **Data Pipeline** | mmap embeddings / DataLoader prefetch |
| **Visualization** | TensorBoard / Weights & Biases |
| **Filtering Stage** | async pipeline for CamSol + ESM re-scoring |

---

## 6. 🌱 Future Extensions (v2+)

1. **Adversarial RAE:** add latent discriminator to increase semantic disentanglement.  
2. **Multi-task Conditioning:** add toxicity and target bacterium as additional conditions.  
3. **Active Learning Loop:** generated → predict → MD simulate → feedback retrain.  
4. **Structural Integration:** incorporate ESM-3 structure track for 3D guided design.  
5. **Deployment:** wrap in Streamlit app for interactive AMP design.

---

## 7. 🪄 Milestone Summary

| Week | Milestone | Key Output |
|------|------------|-------------|
| 1-2 | Dataset & ESM-3 embeddings ready | `esm3_amp.pt` |
| 3-4 | RAE trained, latent space verified | `rae_encoder.pt` |
| 5-7 | Conditional Flow trained | `flow_model.pt` |
| 8-9 | Sequences decoded + filtered | `top_candidates.csv` |

---

## 8. 📚 References

- **ESM-3:** Meta AI (2024), “ESM3: Foundation model for protein sequences and structures.”  
- **RAE:** Kumar et al., ICLR 2021, “Regularized Autoencoders for Representation Learning.”  
- **Flow Matching:** Lipman et al., NeurIPS 2023, “Flow Matching for Generative Modeling.”  
- **ProtFlow:** Lu et al., 2024 BioRxiv, “Protein Design with Flow Matching.”  
- **CamSol / Protein-Sol:** Agostini et al., Bioinformatics 2014.

---

**End of Roadmap**
```
