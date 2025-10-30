# ✅ Phase 1 完成指南

## 当前进度

### 已完成 ✅
- [x] 项目结构搭建
- [x] uv 环境配置 (.venv/)
- [x] 数据解析 (01_parse_fasta.py)
  - 原始数据: **19,083** 序列
  - 来源: DRAMP, APD, non-AMP
- [x] 序列去冗余 (02_deduplicate.py)
  - 去重后: **10,947** 序列
  - 方法: 简单精确匹配 (CD-HIT 未安装)
- [x] 依赖安装
  - biopython, pandas, numpy, torch, fair-esm

### 进行中 ⏳
- [ ] ESM-2 embedding 提取 (03_extract_esm3.py)
  - 状态: 后台运行中
  - 模型: ESM-2 (650M, 1280 dims)
  - 预计时间: 10-30 分钟 (CPU)

---

## 检查 Embedding 提取进度

### 方法 1: 查看进程
```fish
ps aux | grep python | grep 03_extract
```

### 方法 2: 查看输出文件
```fish
ls -lh data/embeddings/
# 应该看到 esm2_embeddings.npz
```

### 方法 3: 手动运行（如果后台任务失败）
```fish
# 激活环境
source .venv/bin/activate.fish

# 运行提取脚本
python scripts/03_extract_esm3.py

# 预计输出:
# 🔧 Loading ESM-2 model...
# Extracting embeddings: 100%|████████| 1368/1368 [XX:XX<00:00]
# ✅ Embeddings saved: esm2_embeddings.npz
```

---

## 完成后验证

```fish
# 运行验证脚本
python scripts/validate_data.py

# 预期输出:
# ✓ Loaded 10947 sequences
# ✓ Embedding shape: (10947, 1280)
# ✓ No NaN values
# ✓ No Inf values
# 🎉 All checks passed!
```

---

## 输出文件清单

```
data/
├── processed/
│   ├── raw_sequences.csv          ✅ 19,083 条
│   ├── dedup_sequences.csv        ✅ 10,947 条
│   └── dedup_sequences.fasta      ✅
│
└── embeddings/
    ├── esm2_embeddings.npz        ⏳ (10947, 1280)
    └── metadata.csv               ⏳
```

---

## 常见问题

### Q: Embedding 提取太慢
**A:** Mac CPU 处理 ~11k 序列预计 15-30 分钟。可以：
1. 减小 batch_size（编辑 03_extract_esm3.py，改为 4 或 2）
2. 用 GPU 机器运行（快 10-50 倍）
3. 先用小数据集测试

### Q: 模型下载失败
**A:** ESM-2 模型 (~2.5GB) 会自动下载到 `~/.cache/torch/hub/`
如果网络问题，可以：
```fish
# 使用 HuggingFace 镜像
set -x HF_ENDPOINT https://hf-mirror.com
```

### Q: 内存不足
**A:** ESM-2 (650M) 在 CPU 上需要约 4-6 GB 内存
- 关闭其他程序
- 减小 batch_size
- 或使用更小的模型 (esm2_t12_35M)

---

## 下一步 (Phase 2)

Embedding 提取完成后：

```fish
# 1. 验证数据
python scripts/validate_data.py

# 2. 创建 train/val/test 划分
python rae/prepare_splits.py

# 3. 训练 RAE
python rae/train.py
```

详见 [`docs/roadmap.md`](docs/roadmap.md) Phase 2

---

**Last Updated:** 2025-10-27  
**Current Status:** ⏳ Waiting for embedding extraction

