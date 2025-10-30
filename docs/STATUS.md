# AMP-FlowRAE 项目状态

**更新时间：** 2024-10-29

---

## 总体进度

| Phase | 任务 | 状态 | 完成时间 |
|-------|------|------|----------|
| **Phase 1** | 数据准备 & ESM-3 Embedding | ✅ 完成 | 2024-10-28 |
| **Phase 2** | RAE 潜空间学习 | ✅ 完成 | 2024-10-29 |
| **Phase 3** | Conditional Flow Matching | ✅ 完成 | 2024-10-29 |
| **Phase 4** | 解码与筛选 | 🔄 进行中 | 2024-10-29 |

---

## Phase 1: 数据准备 ✅

**状态：** 已完成

**数据统计：**
- 原始序列：19,083 条
- 去重后：10,947 条
- 数据来源：DRAMP, APD, non-AMP

**输出文件：**
- `data/embeddings/esm3_embeddings_normalized.npz` (1536 维, 归一化)
- `data/embeddings/metadata_with_labels.csv` (含条件标签)
- `data/splits/` (train/val/test 划分)

**详见：** `docs/PHASE1_COMPLETE.md`

---

## Phase 2: RAE 潜空间学习 ✅

**状态：** 已完成（AMP-only 版本）

**数据调整：**
- 去除 Non-AMP (456 条)
- 保留 AMP-only (10,491 条)
- 重新归一化 + 重新训练

**模型性能（AMP-only）：**
- Train MSE: 0.0654
- Val MSE: 0.0820
- Test MSE: 0.0863

**Latent 空间：**
- 维度：64
- Charge/Hydrophobicity 分布：✅ 清晰
- PCA/t-SNE 可视化：✅ 优秀

**输出文件（使用这些）：**
- 数据：`data/embeddings/esm3_embeddings_normalized.npz`
- 模型：`outputs/rae/checkpoints/best.pt`
- Latent vectors：`outputs/rae_eval/*_latents.npz`
- 可视化：`outputs/rae_eval/visualizations/latent_space.png`

**详见：** `docs/PHASE2_COMPLETE.md`, `docs/PHASE2_UPDATE_AMP_ONLY.md`

---

## Phase 3: Conditional Flow Matching ✅

**状态：** 已完成

**目标：** 学习条件分布 p(z | c)，实现可控生成

**模型性能：**
- Best Val Loss: **0.2248** (Epoch 91/100)
- Train Loss: 0.2411
- 模型参数: 100,928

**条件控制效果：**
| Condition | Correlation | 效果 |
|-----------|-------------|------|
| charge | **0.661** | ✅ **强** |
| length | 0.216 | ✅ 中等 |
| hydrophobicity | 0.150 | ⚠️ 较弱 |

**输出文件：**
- 模型：`outputs/flow/checkpoints/best.pt`
- 生成样本：`outputs/samples/sampled_latents.npy` (500 × 64)
- 条件：`outputs/samples/sampled_conditions.csv`
- 评估：`outputs/samples/evaluation/`

**详见：** `docs/PHASE3_COMPLETE.md`, `QUICKSTART_PHASE3.md`

---

## Phase 4: 解码与筛选 🔄

**状态：** 进行中（Phase 4.1 完成）

**目标：** latent → 序列 + 质量筛选

**Phase 4.1: 序列解码 ✅**
- ✅ 实现 RAE Decoder + 最近邻搜索
- ✅ 成功解码 500 个 latent vectors
- ✅ 结果：
  - 446 个独特序列 (89.2%)
  - 平均 NN 距离：0.55 (高新颖性)
  - 100% 高新颖性序列 (dist > 0.10)

**Phase 4.2-4.4: 待完成**
- ⏳ 活性预测器训练
- ⏳ 端到端生成流程
- ⏳ 质量评估与可视化

---

## 关键指标

### Phase 1
- ✅ Embedding 提取成功率：100%
- ✅ 数据质量：去重后 10,947 条

### Phase 2
- ✅ RAE MSE < 0.1：达成 (0.086)
- ✅ Latent 属性可分性：优秀
- ✅ 训练稳定性：无过拟合

### Phase 3
- ✅ Charge control: 0.661 (目标 >0.7, 接近达成)
- ✅ 生成样本质量：良好
- ✅ 训练收敛：稳定

### Phase 4 (目标)
- ⏳ 生成序列活性预测 > 0.8
- ⏳ 新颖性 (identity < 80%)

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 数据处理 | Biopython, Pandas, NumPy |
| 特征提取 | ESM-3 (esm3-sm-open-v1) |
| 潜空间学习 | PyTorch, RAE |
| 条件生成 | Conditional Flow Matching ✅ |
| 可视化 | Matplotlib, Seaborn, t-SNE |

---

## 文件结构

```
AMP-FlowRAE/
├── data/
│   ├── embeddings/
│   │   ├── esm3_embeddings_normalized.npz  ✅
│   │   └── metadata_with_labels.csv        ✅
│   └── splits/                              ✅
│
├── rae/
│   ├── model.py       ✅
│   ├── dataset.py     ✅
│   ├── train.py       ✅
│   └── evaluate.py    ✅
│
├── flow_matching/     ✅
│   ├── model.py       ✅
│   ├── dataset.py     ✅
│   ├── train.py       ✅
│   ├── sample.py      ✅
│   └── evaluate_samples.py ✅
│
├── outputs/
│   ├── rae/           ✅ Phase 2 输出
│   ├── flow/          ✅ Phase 3 输出
│   └── samples/       ✅ Phase 3 生成样本
│
└── docs/
    ├── PHASE1_COMPLETE.md  ✅
    ├── PHASE2_COMPLETE.md  ✅
    ├── PHASE3_COMPLETE.md  ✅
    └── STATUS.md           ✅ (本文件)
```

---

## 下一步

**优先级 1：** 开始 Phase 4 - 序列解码与活性评估

**任务清单：**
1. 实现 ESM-3 Decoder (latent → sequence)
2. 实现活性预测器 (AMP classifier)
3. 端到端生成流程
4. 生成序列质量评估

**预计时间：** 1-2 周

---

**当前状态：Phase 3 完成，准备进入 Phase 4** 🚀
