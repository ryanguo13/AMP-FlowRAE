# 🧬 AMP-FlowRAE

**Antimicrobial Peptide Representation and Flow Generation Framework**

## 核心目标

基于 **ESM-3 表征 + RAE 潜空间 + Flow Matching** 生成可控的抗菌肽序列。

**技术路线：**
1. ESM-3 提取序列 embedding（冻结）
2. 训练 RAE 学习连续潜空间
3. 训练 Conditional Flow Matching 实现可控生成
4. 解码 + 筛选高质量候选序列

**环境：** Mac (开发) → CUDA (训练)  
**工具：** Python + uv + fish shell

---

## 项目结构

```
AMP-FlowRAE/
├── data/
│   ├── raw/                # FASTA 源文件
│   ├── processed/          # 去重后序列
│   ├── embeddings/         # ESM3 embeddings
│   └── splits/             # train/val/test
│
├── scripts/                # 数据处理脚本
│   ├── 01_parse_fasta.py
│   ├── 02_deduplicate.py
│   └── 03_extract_esm3.py
│
├── rae/                    # Representation Autoencoder
│   ├── model.py
│   └── train.py
│
├── flow_matching/          # Conditional Flow Matching
│   ├── model.py
│   ├── train.py
│   └── sample.py
│
├── decode/                 # 解码与筛选
│   ├── decoder.py
│   └── filter.py
│
└── analysis/               # 分析与可视化
```

---

## 开发阶段

### Phase 1: 数据准备 ✅

**状态：已完成**

- ✅ 解析 FASTA → CSV (19,083 条序列)
- ✅ 去重 (10,947 条)
- ✅ ESM-3 embedding 提取 (1280 维)
- ✅ 数据验证

**数据来源：** DRAMP, APD, non-AMP  
**输出：** `data/embeddings/esm3_embeddings.npz`

---

### Phase 2: RAE 潜空间

**目标：** 在 ESM-3 embedding 上学习低维连续潜空间

**核心任务：**
- 实现 Encoder/Decoder 网络
- 训练 RAE (embedding → latent → embedding)
- 评估重构质量
- 可视化 latent 空间属性分布

**输入：** ESM-3 embeddings (1280 维)  
**输出：** latent vectors (低维，待定)

---

### Phase 3: Conditional Flow Matching

**目标：** 学习条件分布 p(z|c)，实现可控生成

**核心任务：**
- 实现 Flow Matching 模型
- 训练条件生成网络
- 采样与评估

**条件：** 活性、溶解性等属性  
**输出：** 可控的 latent vectors

---

### Phase 4: 解码与筛选

**目标：** latent → 序列 + 质量筛选

**核心任务：**
- latent → embedding → 序列
- 评分（活性、溶解性、理化属性）
- 去重与新颖性过滤

**输出：** 高质量候选序列

---

## 评估指标

| 维度       | 指标                     |
| ---------- | ------------------------ |
| 表征质量   | RAE 重构误差             |
| 潜空间结构 | 属性分布可分性 (PCA/UMAP)|
| 可控性     | 生成序列 vs 目标属性相关性|
| 新颖性     | 与训练集序列相似度       |
| 活性       | 分类器评分               |
| 可溶性     | 预测评分                 |

---

## 技术栈

| 模块          | 技术选型                      |
| ------------- | ----------------------------- |
| 特征提取      | ESM-3 (冻结)                  |
| 潜空间学习    | Representation Autoencoder    |
| 条件生成      | Conditional Flow Matching     |
| 序列解码      | Transformer decoder / 采样    |
| 质量评估      | 分类器 + 理化属性计算         |

---

## 当前状态

**Phase 1:** ✅ 完成  
**Phase 2-4:** 待开发

详见 `STATUS.md` 和 `docs/roadmap.md`
