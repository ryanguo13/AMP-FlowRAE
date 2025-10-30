# ✅ Phase 1 - ESM-3 提取进行中

## 当前状态

**模型**: ESM-3 (esm3-sm-open-v1)  
**维度**: 1536  
**设备**: CPU (Mac)  
**数据**: 10,947 sequences  
**状态**: ⏳ 后台运行中

---

## 检查进度

### 查看进程
```fish
ps aux | grep extract_esm3 | grep -v grep
```

### 查看输出
```fish
# 检查是否生成了文件
ls -lh data/embeddings/

# 应该看到:
# esm3_embeddings.npz  (完成后 ~60-80 MB)
# metadata.csv
```

### 查看运行日志（如果有）
```fish
# 如果用 nohup 运行
tail -f nohup.out

# 或重新前台运行查看进度
.venv/bin/python scripts/03_extract_esm3.py
```

---

## 预期时间

- **CPU (Mac)**: 30-60 分钟
- **GPU**: 3-10 分钟

ESM-3 比 ESM-2 慢一些，因为是多模态模型。

---

## 如果需要重启

```fish
# 1. 杀掉旧进程
pkill -f extract_esm3

# 2. 重新运行
cd /Users/guojunhua/Documents/AMP-FlowRAE
source .venv/bin/activate.fish
python scripts/03_extract_esm3.py
```

---

## 完成后验证

```fish
python scripts/validate_data.py

# 预期:
# ✓ Embedding shape: (10947, 1536)  # 注意是 1536 不是 1280
# ✓ No NaN values
# 🎉 All checks passed!
```

---

## 注意

projectplan.md 里写的是 1280 维，但 **ESM3-sm 实际输出是 1536 维**。

后续 RAE 架构需要调整为：
```python
Encoder: [1536 → 512 → 128 → 64]  # 不是 [1280 → ...]
Decoder: [64 → 128 → 512 → 1536]
```

---

**Last Updated**: 现在  
**Status**: ⏳ ESM-3 extraction running

