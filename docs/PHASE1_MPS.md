# 🚀 Phase 1 - 使用 MPS 加速

## 设备配置

**优先级**: CUDA (NVIDIA GPU) > **MPS (Mac GPU)** > CPU

### 当前环境
- **设备**: Mac with Apple Silicon (M1/M2/M3)
- **加速**: MPS (Metal Performance Shaders)
- **Batch size**: 16 (MPS 优化)
- **预期速度**: 比 CPU 快 **5-10 倍**

---

## MPS vs CPU vs CUDA

| 设备 | Batch Size | 预计时间 (11k sequences) |
|------|------------|--------------------------|
| CPU  | 8          | 30-60 分钟 |
| **MPS** | **16** | **5-10 分钟** ⚡ |
| CUDA | 32         | 2-5 分钟 |

---

## 当前运行状态

```fish
# 查看进程
ps aux | grep extract_esm3 | grep -v grep

# 查看 GPU 使用情况 (Mac Activity Monitor)
# 或命令行:
sudo powermetrics --samplers gpu_power -i 1000 -n 1
```

---

## 设备切换

### 方法 1: 自动检测（推荐）
脚本已自动检测并使用 MPS

### 方法 2: 强制使用特定设备
编辑 `scripts/03_extract_esm3.py`，修改设备检测部分：

```python
# 强制使用 CPU (测试用)
device = 'cpu'
batch_size = 8

# 强制使用 MPS
device = 'mps'
batch_size = 16

# 强制使用 CUDA (转到 GPU 服务器后)
device = 'cuda'
batch_size = 32
```

---

## 性能优化建议

### Mac MPS 优化
```fish
# 1. 关闭其他占用 GPU 的应用
# 2. 连接电源（性能模式）
# 3. 确保散热良好

# 查看系统负载
top -o cpu
```

### 后续转 CUDA 时
```fish
# 检查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 脚本会自动使用 CUDA (优先级最高)
```

---

## Batch Size 调整

如果遇到内存问题：

```python
# 编辑 scripts/03_extract_esm3.py，line ~113-118

# 降低 MPS batch size
elif torch.backends.mps.is_available():
    device = 'mps'
    batch_size = 8  # 从 16 降到 8
```

---

## 输出

**ESM-3 (1536 维) embeddings**

```
data/embeddings/
├── esm3_embeddings.npz    (10947, 1536) ~60-80 MB
└── metadata.csv           序列元数据
```

---

## 验证

完成后运行：

```fish
python scripts/validate_data.py

# 预期输出:
# 🖥️  Device: mps
# ✓ Embedding shape: (10947, 1536)
# 🎉 All checks passed!
```

---

## 后续 Phase 2 (RAE 训练)

**记得更新架构**：

```python
# rae/model.py
Encoder: [1536 → 512 → 128 → 64]  # ESM-3 输出 1536 dims
Decoder: [64 → 128 → 512 → 1536]
```

RAE 训练也会自动使用 MPS 加速。

---

**Last Updated**: 现在  
**Status**: ⚡ MPS 加速运行中  
**预计完成**: 5-10 分钟

