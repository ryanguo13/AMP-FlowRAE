# Phase 2 完成报告 - RAE 潜空间学习

**完成时间：** 2024-10-29  
**状态：** ✅ 完成

---

## 概述

成功训练了 Representation Autoencoder (RAE)，将 ESM-3 embeddings (1536 维) 压缩到 64 维潜空间，同时保持良好的重构质量和属性可解释性。

---

## 数据准备

### 1. 数据划分 ✅

```
Train: 8,757 (80.0%)
Val:   1,095 (10.0%)
Test:  1,095 (10.0%)
```

按 source 分层划分，避免数据泄露。

### 2. 条件标签计算 ✅

- **is_amp**: 0/1 binary label
- **charge**: net charge at pH 7
- **hydrophobicity**: Kyte-Doolittle scale average
- **length**: 序列长度

### 3. Embedding 归一化 ✅

**问题：** ESM-3 原始 embeddings 未归一化
- Mean: 0.19, Std: 247.0
- Range: [-11970, 1359]

**解决：** 标准化到 zero mean, unit variance
- Mean: 0.00, Std: 1.00
- Range: [-6.81, 7.50]

**影响：** MSE 从 57668 → 0.09 (改进 **6 个数量级**)

---

## 模型架构

### RAE 结构

```
Encoder: 1536 → 512 → 256 → 64
Decoder: 64 → 256 → 512 → 1536

Total Parameters: 1,873,984
```

**设计原则 (Linus 风格):**
- ✅ 简单 MLP，不搞复杂的 ResBlock
- ✅ 正交正则化 (λ_orth = 0.1)，确保 latent 维度独立
- ✅ L2 正则化 (λ_z = 0.01)，防止 latent 爆炸

### Loss Function

```
Loss = MSE(x, x_recon) + λ_z * ||z||^2 + λ_orth * ||W W^T - I||^2
```

---

## 训练配置

```python
Epochs:        50
Batch Size:    256
Learning Rate: 1e-3
Optimizer:     AdamW (weight_decay=1e-5)
Scheduler:     ReduceLROnPlateau (patience=5)
Device:        MPS (Mac)
```

训练时间：约 2 分钟 (50 epochs on Mac M1)

---

## 性能指标

### 重构误差

| Split | MSE      | MAE    |
|-------|----------|--------|
| Train | 0.066178 | 0.196  |
| Val   | 0.085066 | 0.218  |
| Test  | 0.085914 | 0.217  |

**结论：** ✅ MSE < 0.1，重构质量优秀

### Latent 空间分析

**PCA 分析：**
- PC1 解释 15.6% 方差
- PC2 解释 12.0% 方差
- 前 2 个主成分捕获约 28% 信息

**t-SNE 可视化：**
- ✅ AMP vs Non-AMP 有明显分离
- ✅ Charge 呈现渐变分布
- ✅ Hydrophobicity 有合理的空间分布

**属性可分性：**
- AMP/Non-AMP 在 latent 空间中可区分
- 物理化学属性（charge, hydrophobicity）呈现连续分布
- 为 Phase 3 条件生成提供良好基础

---

## 文件输出

### 模型 Checkpoints

```
outputs/rae_v1/
├── checkpoints/
│   ├── best.pt          # 最佳模型 (Val MSE: 0.0829)
│   ├── epoch_10.pt
│   ├── epoch_20.pt
│   ├── epoch_30.pt
│   ├── epoch_40.pt
│   └── epoch_50.pt
├── logs/                # TensorBoard logs
├── config.json          # 训练配置
└── rae_final.pt         # 最终模型
```

### 评估结果

```
outputs/rae_v1_eval/
├── train_latents.npz    # Train latent vectors
├── val_latents.npz      # Val latent vectors
├── test_latents.npz     # Test latent vectors
└── visualizations/
    ├── latent_space.png         # PCA + t-SNE 可视化
    └── latent_projections.npz   # 降维结果
```

### 数据文件

```
data/
├── embeddings/
│   ├── esm3_embeddings.npz            # 原始 ESM-3 embeddings
│   ├── esm3_embeddings_normalized.npz # 归一化 embeddings
│   └── metadata_with_labels.csv       # 序列 metadata + 条件标签
└── splits/
    ├── train_indices.npy
    ├── val_indices.npy
    ├── test_indices.npy
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## 代码模块

```
rae/
├── model.py       # RAE 模型定义
├── dataset.py     # 数据加载 (mmap 高效加载)
├── train.py       # 训练脚本
└── evaluate.py    # 评估与可视化

scripts/
├── 04_split_data.py          # 数据划分
├── 05_compute_labels.py      # 条件标签计算
└── 06_normalize_embeddings.py # Embedding 归一化
```

---

## 关键教训 (Linus 视角)

### 1. 数据问题 > 模型问题

**问题：** 初次训练 MSE 高达 57668  
**原因：** ESM-3 embeddings 未归一化 (std=247)  
**解决：** 标准化后 MSE → 0.09  
**教训：** **"永远先修数据，再修模型。"** 神经网络不是魔法，垃圾进去只能出垃圾。

### 2. 简单优于复杂

**设计：** 简单 3 层 MLP  
**效果：** 重构误差 < 0.1  
**教训：** **"如果简单方案能解决问题，别搞复杂的。"** ResBlock, Attention 等可以后加，先让最简单的跑起来。

### 3. 正则化的意义

**Orthogonal Loss (λ_orth = 0.1):**
- 鼓励 latent 维度独立
- 让 latent 空间更可解释

**L2 Regularization (λ_z = 0.01):**
- 防止 latent norm 爆炸
- 为 Flow Matching 提供稳定输入

**教训：** **"好的正则化是免费的性能提升。"**

---

## 下一步：Phase 3 - Conditional Flow Matching

**目标：** 学习条件分布 p(z | c)，实现可控生成

**输入：**
- Latent vectors: `outputs/rae_v1_eval/*_latents.npz`
- Conditions: `data/embeddings/metadata_with_labels.csv`

**任务：**
1. 实现 Conditional Flow Matching 模型
2. 训练条件生成网络 p(z | activity, charge, length)
3. 采样与评估：生成指定属性的 latent vectors

**预期：** correlation(z*, c) > 0.7

---

**Phase 2 状态：✅ 完成**

详见：
- 训练日志: `outputs/rae_v1_train.log`
- TensorBoard: `tensorboard --logdir outputs/rae_v1/logs`
- 可视化: `outputs/rae_v1_eval/visualizations/latent_space.png`



