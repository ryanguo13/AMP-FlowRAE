# 数据质量说明

## ⚠️ 发现的问题：超短序列

### 问题描述

在生成的 AMP 序列中，发现了一些不合理的超短序列（长度 ≤ 4 aa），例如：
- `PF` (2 aa)
- `GL` (2 aa)
- `KKR` (3 aa)
- `ERRP` (4 aa)

这些序列显然不可能是功能性的抗菌肽。

### 根本原因

**问题源自训练数据集本身：**

```
训练集统计（data/embeddings/metadata.csv）:
- 总序列数: 10,491
- 长度 ≤ 4:  94 个 (0.9%)
- 长度 ≤ 10: 1,050 个 (10.0%)
- 最短序列: 2 aa
- 最长序列: 100 aa
- 平均长度: 27.6 aa
```

这些超短序列可能来自：
1. 数据库中的片段或降解产物
2. 数据收集/处理过程中的错误
3. 某些特殊的肽段标注

### 影响

由于使用 **最近邻解码 (Nearest Neighbor Decoding)**，生成的潜空间向量会被映射到训练集中最相似的序列，包括这些超短序列。

**生成结果统计：**
```
原始生成（500 条）:
- 长度 ≤ 4:  11 条 (2.2%)
- 长度 ≤ 10: 35 条 (7.0%)
- 长度 ≥ 15: 421 条 (84.2%)
```

---

## ✅ 解决方案

### 1. 过滤脚本

创建了 `scripts/filter_reasonable_amps.py` 来过滤合理的 AMP 序列：

```bash
python scripts/filter_reasonable_amps.py \
  --input outputs/analysis/generated_amps_analyzed.csv \
  --output outputs/analysis/generated_amps_filtered.csv \
  --min-length 10 \
  --max-length 100
```

**过滤后结果：**
```
保留序列数: 468 / 500 (93.6%)
去除序列数: 32

质量提升：
- 长度范围: [10, 99] aa
- 平均长度: 39.5 ± 24.1 aa
- 平均活性: 0.811 ± 0.133
- 高活性 (>0.8): 302 (64.5%)
- 高新颖性 (>0.5): 351 (75.0%)
```

### 2. 新的 FASTA 文件

生成了过滤后的 FASTA 文件：

```
outputs/
├── generated_amps_filtered.fasta    # 全部 468 条合理序列
├── top50_filtered.fasta             # Top 50 高活性
└── top20_filtered.fasta             # Top 20 最优候选
```

**推荐使用这些过滤后的文件进行后续分析和实验验证。**

---

## 🔧 长期解决方案

### 选项 1: 重新处理训练数据（推荐）

在 Phase 1 数据准备阶段，添加长度过滤：

```python
# 在 scripts/01_parse_fasta.py 或 02_deduplicate.py 中添加
MIN_AMP_LENGTH = 10
MAX_AMP_LENGTH = 100

df_filtered = df[
    (df['length'] >= MIN_AMP_LENGTH) & 
    (df['length'] <= MAX_AMP_LENGTH)
]
```

**优点：**
- 从根源解决问题
- 提高模型质量
- 减少训练时间和计算资源

**实施：**
```bash
# 1. 修改数据处理脚本
# 2. 重新运行 Phase 1
python scripts/01_parse_fasta.py
python scripts/02_deduplicate.py
python scripts/03_extract_esm3.py

# 3. 重新训练 RAE (Phase 2)
python rae/train.py --epochs 50

# 4. 重新训练 Flow Matching (Phase 3)
python flow_matching/train.py --epochs 100
```

### 选项 2: 在解码时过滤（当前方案）

在 `decoder/decode.py` 中添加长度过滤：

```python
# 在 decode.py 的 nearest neighbor search 后
MIN_LENGTH = 10
results_filtered = [
    r for r in results 
    if len(r['sequence']) >= MIN_LENGTH
]
```

**优点：**
- 无需重新训练
- 快速实施

**缺点：**
- 治标不治本
- 仍然会生成映射到短序列的潜向量

---

## 📊 合理的 AMP 长度范围

根据文献和 AMP 数据库统计：

| 长度范围 | 说明 | 占比（典型AMP数据库）|
|---------|------|---------------------|
| 2-9 aa | 不合理 | < 1% |
| **10-14 aa** | 短肽 | ~15% |
| **15-30 aa** | 典型 AMP | ~50% |
| **31-50 aa** | 长肽 | ~25% |
| **51-100 aa** | 蛋白质片段 | ~9% |
| > 100 aa | 小蛋白 | < 1% |

**建议的过滤标准：**
- **最小长度**: 10 aa（保守） 或 15 aa（严格）
- **最大长度**: 50 aa（严格） 或 100 aa（宽松）

---

## 🎯 使用建议

### 用于实验验证

**推荐使用：**
```bash
# Top 20 候选（已过滤，15-50 aa，高活性）
outputs/top20_filtered.fasta
```

**筛选标准：**
- 长度: 15-50 aa
- 活性评分: > 0.85
- 新颖性: > 0.5

### 用于进一步分析

```bash
# 全部合理序列（已过滤，10-100 aa）
outputs/generated_amps_filtered.fasta
```

---

## 📝 总结

1. ✅ **问题已识别**: 训练集包含 94 个超短序列（≤ 4 aa）
2. ✅ **短期方案**: 使用过滤脚本去除不合理序列
3. ✅ **质量提升**: 过滤后保留 468/500 (93.6%) 高质量序列
4. 💡 **长期优化**: 建议在 Phase 1 重新处理数据，从源头过滤

**当前可直接使用过滤后的 FASTA 文件进行实验！**

---

_Last updated: 2024-10-29_


