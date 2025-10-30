# ⚡ Quick Start - Phase 1

## 一键运行（Fish Shell）

```fish
# 1. 环境配置（首次）
cd /Users/guojunhua/Documents/AMP-FlowRAE
uv venv
source .venv/bin/activate.fish
uv pip install biopython pandas numpy torch esm httpx

# 2. 运行 Phase 1
fish scripts/run_phase1.fish

# 或分步运行：
python scripts/01_parse_fasta.py      # 解析 FASTA
python scripts/02_deduplicate.py      # 去冗余  
python scripts/03_extract_esm3.py     # 提取 embeddings (MPS 加速)

# 3. 验证结果
python scripts/validate_data.py
```

---

## 当前配置

### 数据
- **原始**: 19,083 sequences (4 sources)
- **去冗余**: 10,947 sequences
- **长度**: 2-100 aa (平均 ~30 aa)

### 模型
- **ESM-3**: esm3-sm-open-v1
- **维度**: 1536 (不是 1280)
- **设备**: MPS (Mac GPU) ⚡
- **Batch size**: 16

### 时间估算
- Parse FASTA: ~5 秒
- Deduplicate: ~10 秒
- **ESM-3 embeddings**: ~**5-10 分钟** (MPS)

---

## 检查进度

```fish
# 查看运行进程
ps aux | grep extract_esm3 | grep -v grep

# 查看输出文件
ls -lh data/embeddings/

# 完成后应该有:
# esm3_embeddings.npz (60-80 MB)
# metadata.csv
```

---

## 设备支持

脚本自动检测并使用最优设备：

| 优先级 | 设备 | Batch | 速度 |
|--------|------|-------|------|
| 1 | **CUDA** | 32 | 最快 (2-5 min) |
| 2 | **MPS** | 16 | 快 (5-10 min) ⚡ |
| 3 | CPU | 8 | 慢 (30-60 min) |

当前：**MPS** (Mac M1/M2/M3 GPU)

---

## 输出

```
data/
├── processed/
│   ├── raw_sequences.csv       (19,083)
│   ├── dedup_sequences.csv     (10,947) ✅
│   └── dedup_sequences.fasta   ✅
│
└── embeddings/
    ├── esm3_embeddings.npz     (10947, 1536) ✅
    └── metadata.csv            ✅
```

---

## 下一步 (Phase 2)

```fish
# 创建数据划分
python rae/prepare_splits.py  # TODO

# 训练 RAE (注意维度是 1536)
python rae/train.py  # TODO
```

**重要**: projectplan.md 里写的是 1280 维，但 **ESM3-sm 实际是 1536 维**

RAE 架构需要调整为：
```python
Encoder: [1536 → 512 → 128 → 64]
Decoder: [64 → 128 → 512 → 1536]
```

---

## 故障排除

### MPS 内存不足
```python
# 编辑 scripts/03_extract_esm3.py
batch_size = 8  # 降低到 8
```

### 切换到 CPU
```python
# 编辑 scripts/03_extract_esm3.py
device = 'cpu'
batch_size = 8
```

### 后续转 CUDA
不需要改代码，脚本会自动检测并使用 CUDA。

---

**Last Updated**: 2025-10-27  
**Status**: ⚡ Phase 1 running on MPS

