# AMP-FlowRAE: 可控抗菌肽生成系统

**基于 Flow Matching 的抗菌肽（AMP）设计系统**

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)]()

---

## 🎯 项目概述

AMP-FlowRAE 是一个端到端的深度生成模型系统，用于设计具有特定理化性质的新型抗菌肽。

**核心功能：**
- ✅ 可控生成：根据电荷、疏水性、长度生成 AMP
- ✅ 高新颖性：75% 生成序列远离训练集
- ✅ 质量评估：自动预测活性并提供可视化
- ✅ 实用输出：FASTA 格式，可直接用于实验

---

## 📊 项目成果

### 性能指标

| 指标 | 结果 | 状态 |
|------|------|------|
| **RAE 重建误差** | MSE = 0.086 | ✅ 优秀 |
| **Flow Matching** | Val Loss = 0.225 | ✅ 优秀 |
| **Charge 控制** | r = 0.661 | ✅ 强 |
| **独特序列率** | 89.2% | ✅ 超额 |
| **高活性率** | 61.0% (>0.8) | ✅ 良好 |
| **高新颖性率** | 75.2% (>0.5) | ✅ 超额 |

### 生成示例

**Top 5 生成的 AMP 序列：**

| 序列 | 活性评分 | 电荷 | 长度 | 新颖性 |
|------|---------|------|------|--------|
| GLLGPLLKIAAKVGKNLL | 0.948 | +3 | 18 | 0.711 |
| FLGALWKVAKKVF | 0.948 | +3 | 13 | 0.565 |
| KKKKLVLAFLFFF | 0.944 | +4 | 13 | 0.648 |
| TNWKKIGKC... | 0.942 | +3 | 41 | 0.604 |
| FMGGLIKAA... | 0.941 | +4 | 24 | 0.539 |

---

## 🏗️ 系统架构

```
用户指定条件 (charge, hydrophobicity, length)
           ↓
[Flow Matching] 生成潜空间向量 (64-dim)
           ↓
[RAE Decoder] 解码到 ESM-3 embeddings (1536-dim)
           ↓
[Nearest Neighbor] 匹配到真实序列
           ↓
[Property Calculator] 计算理化性质
           ↓
[Activity Predictor] 预测活性评分
           ↓
FASTA 输出 → 实验验证
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-repo/AMP-FlowRAE
cd AMP-FlowRAE

# 安装依赖
pip install -r requirements.txt
```

### 2. 生成 AMP 序列

```bash
# Step 1: 生成潜空间向量
python flow_matching/sample.py \
  --num-samples 100 \
  --charge-range 3 8 \
  --length-range 15 30 \
  --output-dir outputs/my_generation

# Step 2: 解码到序列
python decoder/decode.py \
  --latents outputs/my_generation/sampled_latents.npy \
  --rae-checkpoint outputs/rae/checkpoints/best.pt \
  --output outputs/my_sequences.csv

# Step 3: 质量分析
python scripts/09_analyze_generated.py \
  --input outputs/my_sequences.csv \
  --output-dir outputs/my_analysis

# Step 4: 导出 FASTA
python scripts/export_fasta.py \
  --input outputs/my_analysis/generated_amps_analyzed.csv \
  --output outputs/my_amps.fasta \
  --min-activity 0.8 \
  --max-sequences 20
```

### 3. 查看结果

```bash
# 查看序列
head outputs/my_amps.fasta

# 查看可视化
open outputs/my_analysis/*.png
```

---

## 📁 项目结构

```
AMP-FlowRAE/
├── data/                    # 数据
│   ├── embeddings/          # ESM-3 embeddings
│   └── splits/              # 训练/验证/测试划分
│
├── rae/                     # Phase 2: RAE 模型
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
├── decoder/                 # Phase 4: 序列解码
│   ├── nearest_neighbor.py
│   └── decode.py
│
├── utils/                   # 工具函数
│   └── sequence_properties.py
│
├── scripts/                 # 数据处理脚本
│   ├── 09_analyze_generated.py
│   └── export_fasta.py
│
├── outputs/                 # 输出文件
│   ├── rae/                 # RAE 模型
│   ├── flow/                # Flow 模型
│   ├── analysis/            # 分析结果
│   └── *.fasta              # FASTA 文件
│
└── docs/                    # 文档
    ├── PHASE1_COMPLETE.md
    ├── PHASE2_COMPLETE.md
    ├── PHASE3_COMPLETE.md
    ├── PHASE4_COMPLETE.md
    └── STATUS.md
```

---

## 📚 详细文档

- [**项目总结**](SUMMARY.md) - 完整的技术总结
- [**Phase 1-4 文档**](docs/) - 各阶段详细报告
- [**快速开始指南**](QUICKSTART_EXPORT.md) - FASTA 导出教程
- [**项目状态**](docs/STATUS.md) - 当前进度

---

## 🔬 技术亮点

### 1. Conditional Flow Matching

- **创新点：** 使用 Flow Matching 而非传统 VAE/GAN
- **优势：** 训练稳定，生成质量高
- **条件控制：** 支持电荷、疏水性、长度

### 2. 最近邻解码

- **方法：** 基于 cosine 距离的 k-NN 搜索
- **优势：** 简单、快速、保证序列有效性
- **权衡：** 新颖性 vs 可靠性

### 3. 启发式活性预测

- **基于：** 理化性质加权评分
- **优势：** 无需训练，可解释性强
- **用途：** 快速筛选候选序列

---

## 📈 应用场景

### 1. 药物发现

- 生成新型抗菌肽候选
- 优化现有 AMP 的理化性质
- 探索新的序列空间

### 2. 科学研究

- 研究序列-功能关系
- 验证生成模型在生物学中的应用
- 提供数据集用于其他研究

### 3. 教学演示

- 深度生成模型案例
- 生物信息学应用
- 端到端机器学习流程

---

## ⚠️ 重要说明

### 局限性

1. **活性预测为估计值**
   - 基于理化性质的启发式评分
   - 需要实验验证

2. **解码方案简单**
   - 依赖最近邻搜索
   - 不是真正的"生成"

3. **条件控制不完美**
   - Hydrophobicity 控制较弱 (r=0.150)
   - 可通过增加模型容量改进

### 建议

- ✅ 用于**候选筛选**和**初步探索**
- ✅ 结合实验验证
- ⚠️ 不要直接用于临床

---

## 🎓 引用

如果您使用了本项目，请引用：

```bibtex
@software{amp_flowrae_2024,
  title = {AMP-FlowRAE: Controllable Antimicrobial Peptide Generation},
  year = {2024},
  author = {Your Name},
  url = {https://github.com/your-repo/AMP-FlowRAE}
}
```

---

## 📞 联系方式

- **项目主页：** https://github.com/your-repo/AMP-FlowRAE
- **问题反馈：** [Issues](https://github.com/your-repo/AMP-FlowRAE/issues)
- **邮箱：** your.email@example.com

---

## 📜 许可证

MIT License

---

## 🙏 致谢

- **ESM-3** - Meta AI 的蛋白质语言模型
- **Flow Matching** - 生成建模新方法
- **DRAMP & APD** - AMP 数据库

---

**让我们一起推动抗菌肽药物发现！🚀**

_Last updated: 2024-10-29_
