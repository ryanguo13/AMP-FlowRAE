# RAE Loss 改进计划 - 执行指南

**问题诊断完成** ✅  
**归一化文件已存在** ✅  
**准备执行改进**

---

## 🔍 问题总结

### 当前状态
- **Train MSE**: ~194.6
- **Val MSE**: ~212.9
- **Test MSE**: ~212.8
- **数据**: 使用**未归一化**文件（std=229.32）

### 根本原因
**使用了未归一化的 embeddings 文件**，导致：
- MSE 数值看起来很大（~200）
- 虽然相对误差可能可接受（~6%），但不符合训练目标（< 0.02）
- 与历史归一化结果差距 **2500倍**

---

## ✅ 解决方案

### 已确认资源
- ✅ 归一化文件存在：`data/embeddings/esm3_embeddings_normalized.npz`
- ✅ 统计正确：Mean=0, Std=1, Variance=1

### 立即执行步骤

#### 1. 使用归一化文件重新训练

```bash
cd /home/ryankwok/Documents/AMP-FlowRAE

# 激活环境
source .venv/bin/activate.fish  # 或 source .venv/bin/activate

# 使用归一化文件训练
python rae/train.py \
    --npz-path data/embeddings/esm3_embeddings_normalized.npz \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --output-dir outputs/rae_normalized
```

#### 2. 监控训练过程

```bash
# 在另一个终端
tensorboard --logdir outputs/rae_normalized/logs
```

#### 3. 预期结果

使用归一化数据后，预期：
- **Train MSE**: < 0.1（改进 **2000倍**）
- **Val MSE**: < 0.12
- **Test MSE**: < 0.12
- **MAE**: < 0.3

---

## 📊 对比分析

### 数据对比

| 指标 | 未归一化 | 归一化 | 改进 |
|------|---------|--------|------|
| Mean | 0.06 | 0.00 | ✅ |
| Std | 229.32 | 1.00 | ✅ |
| Variance | 52589 | 1.00 | ✅ |
| Range | [-12282, 1367] | [-7.7, 6.9] | ✅ |

### 预期 MSE 对比

| Split | 当前（未归一化） | 预期（归一化） | 改进倍数 |
|-------|----------------|--------------|---------|
| Train | ~194.6 | < 0.1 | **~2000x** |
| Val | ~212.9 | < 0.12 | **~1800x** |
| Test | ~212.8 | < 0.12 | **~1800x** |

---

## 🎯 如果归一化后仍不够好

如果使用归一化数据后 MSE 仍然 > 0.1，考虑以下优化：

### 方案 1：增加 Latent 维度

```bash
python rae/train.py \
    --npz-path data/embeddings/esm3_embeddings_normalized.npz \
    --latent-dim 128 \  # 从 64 增加到 128
    --hidden-dims 512 256 128 \
    --output-dir outputs/rae_latent128
```

### 方案 2：增加模型容量

```bash
python rae/train.py \
    --npz-path data/embeddings/esm3_embeddings_normalized.npz \
    --hidden-dims 1024 512 256 \  # 从 [512, 256] 增加
    --output-dir outputs/rae_larger
```

### 方案 3：调整学习率

```bash
python rae/train.py \
    --npz-path data/embeddings/esm3_embeddings_normalized.npz \
    --lr 5e-4 \  # 从 1e-3 降低
    --output-dir outputs/rae_lr5e4
```

### 方案 4：更长的训练

```bash
python rae/train.py \
    --npz-path data/embeddings/esm3_embeddings_normalized.npz \
    --epochs 200 \  # 从 100 增加
    --output-dir outputs/rae_long
```

---

## 📋 执行检查清单

- [ ] 1. 确认归一化文件存在
- [ ] 2. 使用归一化文件重新训练
- [ ] 3. 监控训练过程（TensorBoard）
- [ ] 4. 检查训练结果是否符合预期（MSE < 0.1）
- [ ] 5. 如果不够好，尝试方案 1-4
- [ ] 6. 评估最终模型性能

---

## ⏱️ 时间估算

- **重新训练**：10-30 分钟（取决于硬件）
- **评估**：5 分钟
- **总时间**：< 1 小时

---

## 📝 注意事项

1. **备份当前模型**（如果需要）：
   ```bash
   cp -r outputs/rae outputs/rae_backup
   ```

2. **检查数据一致性**：
   - 确保 splits 文件与归一化文件匹配
   - 检查数据量是否一致

3. **对比实验**：
   - 可以同时运行多个实验对比
   - 使用不同的 `--output-dir` 区分

---

**Last Updated**: 2024-12-19

