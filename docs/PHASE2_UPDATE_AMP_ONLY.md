# Phase 2 更新 - AMP-only 模型

**更新时间：** 2024-10-29  
**原因：** 去除 Non-AMP 数据，专注 AMP 生成

---

## 数据调整

### 决策理由

**问题：** 原始数据集极度不平衡
- AMP: 10,491 (95.8%)
- Non-AMP: 456 (4.2%)
- t-SNE 显示 AMP/Non-AMP 无明显分离

**分析：**
- 项目目标：**生成 AMP**，不是 AMP 分类
- Non-AMP 只占 4.2%，对整体影响极小
- is_amp 标签不可分（ESM-3 不编码活性）

**决策：去除 Non-AMP，专注 AMP 生成**

### 数据处理

**过滤：**
```
原始:   10,947 条 (10,491 AMP + 456 Non-AMP)
过滤后: 10,491 条 (100% AMP)
```

**关键操作：**
1. ✅ 裁剪 embeddings（未重新跑 ESM-3）
2. ✅ 重新归一化（基于 AMP-only）
3. ✅ 更新 train/val/test splits

**归一化差异：**
```
原来 (All):  Mean=0.19, Std=246.96
现在 (AMP):  Mean=0.20, Std=247.67
差异: < 0.3% ✅ 可忽略
```

---

## 模型性能对比

### RAE 重构误差

| Split | 原始模型 (All) | AMP-only | 改进 |
|-------|---------------|----------|------|
| Train | 0.0674        | 0.0654   | ↓ 3.0% |
| Val   | 0.0829        | 0.0820   | ↓ 1.1% |
| Test  | 0.0911        | 0.0863   | ↓ 5.3% |

**结论：** AMP-only 模型略优，符合预期。

### Latent 空间分析

**PCA/t-SNE 可视化：**
- ✅ Charge 分布清晰
- ✅ Hydrophobicity 连续分布
- ✅ 无 Non-AMP 干扰，更纯净

---

## 文件结构

### 新增文件

```
data/
├── embeddings/
│   ├── esm3_embeddings_amp_only.npz            # 原始 (未归一化)
│   ├── esm3_embeddings_amp_only_normalized.npz # 归一化 ✅ 使用这个
│   └── metadata_amp_only.csv
└── splits_amp_only/
    ├── train_indices.npy  (8,392 条)
    ├── val_indices.npy    (1,049 条)
    └── test_indices.npy   (1,050 条)

outputs/
├── rae_amp_only/
│   ├── checkpoints/best.pt      # AMP-only 最佳模型
│   └── logs/
└── rae_amp_only_eval/
    ├── train_latents.npz
    ├── val_latents.npz
    ├── test_latents.npz
    └── visualizations/latent_space.png
```

### 保留文件（参考）

```
outputs/
├── rae_v1/           # 原始模型 (含 Non-AMP) - 仅供对比
└── rae_v1_eval/      # 原始评估 - 仅供对比
```

---

## Phase 3 准备

### 使用 AMP-only 数据

**训练 Conditional Flow Matching：**

```bash
python flow_matching/train.py \
  --latents-path outputs/rae_amp_only_eval/train_latents.npz \
  --metadata-path data/embeddings/metadata_amp_only.csv \
  --splits-dir data/splits_amp_only \
  --output-dir outputs/flow_amp_only
```

**条件设置（去掉 is_amp）：**
```python
# 原计划
p(z | is_amp, charge, hydrophobicity, length)

# 修正计划
p(z | charge, hydrophobicity, length)
```

**理由：**
- ✅ 全部是 AMP，不需要 is_amp 条件
- ✅ Charge/Hydrophobicity 在 latent 空间可分
- ✅ 生成时控制物理化学属性更实用

---

## Linus 总结

> **"数据不平衡不总是问题。关键是它是否阻止你解决真正的问题。"**

**核心洞察：**

1. **数据反映真实情况**
   - DRAMP/APD 本来就是 AMP 数据库
   - Non-AMP 少是正常的，不是数据收集失误

2. **目标决定策略**
   - 目标：生成 AMP → 不需要 Non-AMP
   - 目标：分类 → 需要平衡数据

3. **is_amp 是糟糕的标签**
   - ESM-3 不编码活性
   - 二分类不可分（ROC-AUC = 0.58）
   - 连续属性（charge, hydrophobicity）才是可用的

4. **实用主义**
   - 裁剪 < 1 分钟
   - 重新跑 ESM-3 = 数小时
   - 差异 < 0.3%，不重跑是对的

---

**Phase 2 状态：✅ 完成（AMP-only 版本）**

**下一步：Phase 3 - Conditional Flow Matching**



