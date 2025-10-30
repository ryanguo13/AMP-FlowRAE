# Phase 4 计划：序列解码与活性评估

**目标：** 从 latent vectors 生成真实的 AMP 序列并评估质量

---

## 🎯 核心任务

### 1. 序列解码 (Latent → Sequence)

**挑战：** ESM-3 是单向编码器，没有官方 decoder

**方案选择：**

#### 方案 A: 基于相似度搜索（最简单，推荐）✅
```
生成的 latent → RAE Decoder → ESM-3 embedding 
→ 在训练集中找最近邻 → 返回对应序列
```

**优点：**
- ✅ 实现简单，可靠
- ✅ 保证生成的序列是有效的 AMP
- ✅ 可以作为 baseline

**缺点：**
- ⚠️ 可能缺乏新颖性（总是返回训练集中的序列）
- ⚠️ 但可以通过"近邻插值"增加多样性

#### 方案 B: 训练独立的 Decoder（复杂）
```
ESM-3 embedding → Transformer Decoder → 序列
```

**优点：**
- ✅ 可生成全新序列
- ✅ 更大的探索空间

**缺点：**
- ⚠️ 需要大量数据训练
- ⚠️ 生成序列可能无效
- ⚠️ 实现复杂度高

#### 方案 C: 混合方案（平衡）
```
1. 先用方案A找k个近邻
2. 对近邻序列进行"突变"（随机替换、插入、删除）
3. 用ESM-3重新编码，保留与目标最接近的
```

**决策：先实现方案A，效果好则继续，效果不佳再考虑B/C**

---

### 2. 活性预测

**目标：** 预测生成序列的抗菌活性

**方案：**

#### 2.1 训练 AMP 分类器
```
输入: ESM-3 embedding (1536维)
输出: P(is_AMP) 概率

架构: 简单 MLP
- Input: 1536
- Hidden: [512, 256]
- Output: 1 (sigmoid)
```

**训练数据：**
- 正样本：DRAMP + APD (10,491 条 AMP)
- 负样本：需要收集 Non-AMP（蛋白质数据库）

#### 2.2 理化性质评估
- Charge（电荷）
- Hydrophobicity（疏水性）
- Length（长度）
- Boman Index
- Aliphatic Index

**这些可以直接从序列计算，无需训练模型**

---

### 3. 端到端生成流程

```python
# 指定条件
conditions = {
    "charge": +8,
    "hydrophobicity": -0.5,
    "length": 25
}

# Step 1: Flow Matching 生成 latent
z = flow_model.sample(conditions)

# Step 2: RAE 解码到 embedding
embedding = rae.decode(z)

# Step 3: 相似度搜索 → 序列
sequence = nearest_neighbor_search(embedding, train_set)

# Step 4: 活性预测
activity_score = amp_classifier(embedding)

# Step 5: 理化性质验证
props = compute_properties(sequence)
```

---

## 📋 实现清单

### Phase 4.1: 序列解码 (1-2天)
- [x] ~~Phase 3 完成~~
- [ ] 实现 `decoder/nearest_neighbor.py`
  - 加载训练集 embeddings + sequences
  - 构建高效索引（FAISS/Annoy）
  - k-NN 搜索
- [ ] 实现 `decoder/decode.py`
  - RAE decode: latent → embedding
  - NN search: embedding → sequence
- [ ] 测试：对 Phase 3 生成的 500 个 latent 解码

### Phase 4.2: 活性预测器 (2-3天)
- [ ] 收集 Non-AMP 数据
  - UniProt 随机蛋白
  - 或者使用其他负样本数据集
- [ ] 训练 AMP 分类器 `classifier/train.py`
  - 简单 MLP
  - 评估指标：Accuracy, Precision, Recall, AUC
- [ ] 实现推理脚本 `classifier/predict.py`

### Phase 4.3: 端到端评估 (1-2天)
- [ ] 实现 `pipeline/generate.py`
  - 条件 → latent → embedding → 序列 → 评分
- [ ] 质量评估
  - 新颖性（与训练集的相似度）
  - 活性预测得分
  - 理化性质分布
  - 去重

### Phase 4.4: 可视化与报告 (1天)
- [ ] 生成序列的理化性质分布
- [ ] 与训练集对比
- [ ] 成功案例展示
- [ ] 完成 `docs/PHASE4_COMPLETE.md`

---

## 🔬 评估指标

### 解码质量
- **重建准确性：** 训练集序列 → embedding → decode → 是否返回原序列？
- **最近邻距离：** 生成的 embedding 到训练集的平均距离

### 活性预测
- **分类器性能：** AUC > 0.90
- **生成序列活性：** P(is_AMP) > 0.80

### 新颖性
- **序列相似度：** 与训练集最相似序列的 identity
- **目标：** 60-80% identity（既保留功能，又有新颖性）

### 条件控制
- **属性匹配度：** 生成序列的 charge/hydro/length 与指定条件的差异
- **目标：** charge error < ±2, length error < ±5

---

## 🚀 快速开始（预计）

```bash
# Step 1: 解码生成的 latent vectors
python decoder/decode.py \
  --latents outputs/samples/sampled_latents.npy \
  --rae-checkpoint outputs/rae/checkpoints/best.pt \
  --output outputs/decoded_sequences.csv

# Step 2: 活性预测（需先训练分类器）
python classifier/train.py \
  --positive-data data/embeddings/esm3_embeddings_normalized.npz \
  --negative-data data/non_amp_embeddings.npz \
  --output outputs/classifier/

# Step 3: 端到端生成
python pipeline/generate.py \
  --num-samples 100 \
  --charge-range 5 10 \
  --length-range 20 30 \
  --output outputs/generated_amps.csv
```

---

## 📊 预期输出

```
outputs/
├── decoded_sequences.csv      # 解码的序列 + 属性
├── classifier/
│   └── best.pt                # 活性分类器
└── generated_amps.csv         # 最终生成的 AMP + 评分
```

---

## ⚠️ 潜在问题

### 1. 新颖性不足
如果最近邻搜索总是返回训练集中的序列：
- **方案：** 增加"突变"步骤（随机替换氨基酸）
- **方案：** 返回 top-k 近邻的加权平均 embedding

### 2. 活性预测器训练数据不足
如果 Non-AMP 数据难以获取：
- **方案：** 使用单类分类（One-Class SVM）
- **方案：** 使用无监督异常检测

### 3. 生成序列与条件不匹配
如果解码后的序列属性与指定条件差异大：
- **原因：** Flow Matching 条件控制不够强（Phase 3 hydro 已较弱）
- **方案：** 后处理筛选（只保留满足条件的序列）
- **方案：** 改进 Flow Matching 训练（增加条件约束）

---

## 🎯 Phase 4 成功标准

- ✅ 能从 latent 解码回有效的 AMP 序列
- ✅ 活性分类器 AUC > 0.90
- ✅ 生成序列活性预测 > 0.80
- ✅ 至少 30% 的生成序列具有新颖性（identity < 80%）
- ✅ 条件控制有效（charge error < ±2）

---

**预计完成时间：** 1 周  
**当前状态：** 计划中，准备开始实现


