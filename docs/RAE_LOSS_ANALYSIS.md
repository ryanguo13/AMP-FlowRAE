# RAE Loss 分析与改进方案

**日期**: 2024-12-19  
**问题**: RAE 训练 loss 较大，需要客观评估和改进

---

## 📊 当前结果评估

### 训练结果
```
Train MSE: ~194.6
Val MSE:   ~211.3  
Test MSE:  ~212.8
MAE:       ~10.6-11.0
```

### 数据统计（实际使用的文件）
```
文件: data/embeddings/esm3_embeddings.npz
Shape: (43597, 1536)
Mean:  0.06
Std:   229.32
Range: [-12282, 1367]
Per-dim variance: 52589
```

---

## 🔍 问题诊断

### 1. **核心问题：数据未归一化** ⚠️

**发现**：
- 训练使用的是**未归一化**的 embeddings
- 标准差 = 229.32，方差 = 52589
- 这导致 MSE 数值看起来很大（~200），但实际相对误差可能还可以

**相对误差分析**：
- 如果 embeddings 方差是 52589，MSE ~200 意味着：
  - 相对误差 = sqrt(200/52589) ≈ **0.06** (6%)
  - 相对于标准差：200/52589 ≈ **0.38%**

**对比历史结果**：
- 文档显示归一化后：MSE ~0.08 (目标 < 0.02)
- 当前未归一化：MSE ~200
- **差距：2500倍** ❌

### 2. **训练收敛性分析**

**观察**：
- Epoch 486-500: Val MSE 稳定在 ~212.88
- 没有明显下降趋势
- 可能已经收敛到局部最优

**问题**：
- 训练 loss (194.6) < Val loss (212.9)，存在轻微过拟合
- 但差距不大（~9%），说明模型容量可能不足或数据质量有问题

### 3. **模型容量评估**

**架构**：
```
Encoder: 1536 → 512 → 256 → 64
Decoder: 64 → 256 → 512 → 1536
Parameters: ~1.87M
```

**压缩比**：
- 1536 → 64 = **24倍压缩**
- 对于未归一化的数据，这个压缩比可能过于激进

---

## ✅ 客观评价

### 当前状态：**需要改进** ⚠️

**优点**：
1. ✅ 训练稳定，没有崩溃
2. ✅ Train/Val gap 不大，过拟合不严重
3. ✅ 模型收敛到稳定状态

**问题**：
1. ❌ **数据未归一化**（最关键）
2. ❌ MSE 绝对值大（虽然相对误差可能可接受）
3. ❌ 与历史归一化结果差距巨大（2500倍）
4. ⚠️ 模型可能容量不足（24倍压缩太激进）

---

## 🎯 改进方案

### 优先级 1：数据归一化（必须）🔥

**问题**：使用未归一化的 embeddings 训练

**解决方案**：

#### 方案 A：使用已归一化的文件（推荐）

```bash
# 检查是否有归一化文件
ls -lh data/embeddings/*normalized*.npz

# 如果有，修改训练脚本使用归一化文件
python rae/train.py \
    --npz-path data/embeddings/esm3_embeddings_normalized.npz \
    --output-dir outputs/rae_normalized
```

#### 方案 B：重新归一化数据

```bash
# 运行归一化脚本
python scripts/06_normalize_embeddings.py

# 然后使用归一化文件训练
python rae/train.py \
    --npz-path data/embeddings/esm3_embeddings_normalized.npz \
    --output-dir outputs/rae_normalized
```

**预期改进**：
- MSE 从 ~200 → **< 0.1** (改进 2000倍)
- 训练更稳定
- 收敛更快

---

### 优先级 2：增加模型容量（可选）💡

**问题**：24倍压缩可能过于激进

**解决方案**：

#### 方案 A：增加 latent 维度

```bash
python rae/train.py \
    --latent-dim 128 \  # 从 64 增加到 128
    --hidden-dims 512 256 128 \  # 调整中间层
    --output-dir outputs/rae_latent128
```

**预期**：
- 更好的重构质量
- 但 latent 空间可能不够紧凑

#### 方案 B：增加隐藏层维度

```bash
python rae/train.py \
    --hidden-dims 1024 512 256 \  # 从 [512, 256] 增加
    --output-dir outputs/rae_larger
```

**预期**：
- 更好的表达能力
- 但参数更多，可能过拟合

---

### 优先级 3：调整训练策略（可选）🔧

#### 方案 A：降低学习率

```bash
python rae/train.py \
    --lr 5e-4 \  # 从 1e-3 降低
    --output-dir outputs/rae_lr5e4
```

#### 方案 B：增加正则化

```bash
python rae/train.py \
    --lambda-z 0.05 \  # 从 0.01 增加
    --lambda-orth 0.2 \  # 从 0.1 增加
    --output-dir outputs/rae_reg
```

#### 方案 C：更长的训练

```bash
python rae/train.py \
    --epochs 200 \  # 从 500 可能不够
    --output-dir outputs/rae_long
```

---

### 优先级 4：数据质量检查（诊断）🔍

**检查数据分布**：

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/embeddings/esm3_embeddings.npz')
emb = data['embeddings']

# 检查每个维度的分布
print(f"Per-dimension stats:")
print(f"  Mean range: [{emb.mean(axis=0).min():.2f}, {emb.mean(axis=0).max():.2f}]")
print(f"  Std range:  [{emb.std(axis=0).min():.2f}, {emb.std(axis=0).max():.2f}]")

# 检查异常值
outliers = np.abs(emb) > 1000
print(f"  Outliers (>1000): {outliers.sum()} ({outliers.sum()/emb.size*100:.2f}%)")
```

**可能的问题**：
- 某些维度方差过大
- 存在异常值
- 数据分布不均匀

---

## 📋 推荐行动计划

### 立即执行（必须）

1. ✅ **检查是否有归一化文件**
   ```bash
   ls data/embeddings/*normalized*.npz
   ```

2. ✅ **如果有，重新训练使用归一化数据**
   ```bash
   python rae/train.py \
       --npz-path data/embeddings/esm3_embeddings_normalized.npz \
       --epochs 100 \
       --output-dir outputs/rae_fixed
   ```

3. ✅ **如果没有，运行归一化脚本**
   ```bash
   python scripts/06_normalize_embeddings.py
   python rae/train.py \
       --npz-path data/embeddings/esm3_embeddings_normalized.npz \
       --epochs 100 \
       --output-dir outputs/rae_fixed
   ```

### 后续优化（可选）

4. 如果归一化后 MSE 仍然 > 0.1，考虑：
   - 增加 latent 维度到 128
   - 增加隐藏层容量
   - 调整学习率和正则化

5. 监控训练过程：
   ```bash
   tensorboard --logdir outputs/rae_fixed/logs
   ```

---

## 🎯 预期结果

### 归一化后预期指标

```
Train MSE: < 0.1  (目标: < 0.02)
Val MSE:   < 0.12
Test MSE:  < 0.12
MAE:       < 0.3
```

### 成功标准

- ✅ MSE < 0.1（相对于归一化数据）
- ✅ Train/Val gap < 20%
- ✅ 训练曲线平滑下降
- ✅ Latent 空间可视化合理

---

## 📝 总结

**当前问题**：
1. **数据未归一化**（最关键，导致 MSE 数值大）
2. 模型可能容量不足（24倍压缩）
3. 训练可能未充分收敛

**改进优先级**：
1. 🔥 **数据归一化**（必须，预期改进 2000倍）
2. 💡 增加模型容量（可选，如果归一化后仍不够）
3. 🔧 调整训练策略（可选，微调）

**预期时间**：
- 归一化：5分钟
- 重新训练：10-30分钟（取决于硬件）
- 总时间：< 1小时

---

**Last Updated**: 2024-12-19

