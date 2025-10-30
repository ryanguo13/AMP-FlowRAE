# ESM3 本地加载配置指南

## 问题说明

你在 `models/` 目录只有 `esm3_sm_open_v1.pth` 权重文件，但 ESM3 的 `from_pretrained()` 需要 **完整的模型目录**：

```
models/esm3_sm_open_v1/
├── config.json              ❌ 缺失
├── pytorch_model.bin        ❌ 缺失  
├── tokenizer_config.json    ❌ 缺失
└── special_tokens_map.json  ❌ 缺失
```

## 解决方案（3 种）

### 方案 A：自动下载（推荐 - 最简单）

修改后的 `03_extract_esm3.py` 会自动处理：

```fish
# 直接运行，脚本会自动下载完整模型
python scripts/03_extract_esm3.py
```

**优点**：
- ✅ 零配置，自动下载
- ✅ 模型会缓存到 `~/.cache/esm/`，下次直接用
- ✅ 保证模型完整性

**缺点**：
- 需要联网（首次下载 ~1-2 GB）

---

### 方案 B：手动下载完整模型

如果你想完全离线使用：

```fish
# 1. 运行下载脚本（需要联网一次）
python scripts/download_esm3_model.py

# 2. 之后就可以完全离线使用
python scripts/03_extract_esm3.py
```

这会下载完整模型到 `models/esm3_sm_open_v1/`。

---

### 方案 C：从 HuggingFace 手动下载

如果自动下载失败：

```fish
# 1. 安装 huggingface-cli
uv pip install huggingface-hub

# 2. 下载完整模型
huggingface-cli download \
  EvolutionaryScale/esm3-sm-open-v1 \
  --local-dir models/esm3_sm_open_v1

# 3. 验证文件
ls -lh models/esm3_sm_open_v1/
# 应该看到: config.json, model.safetensors, tokenizer/ 等
```

---

## 验证模型加载

```fish
# 快速测试
python -c "
from esm.models.esm3 import ESM3
model = ESM3.from_pretrained('esm3-sm-open-v1')
print('✅ ESM3 model loaded successfully')
print(f'   Hidden size: {model.config.hidden_size}')
"
```

预期输出：
```
✅ ESM3 model loaded successfully
   Hidden size: 1536
```

---

## 常见问题

### Q1: 下载太慢怎么办？

**A:** 使用 HuggingFace 镜像：

```fish
# 中国用户可用
set -x HF_ENDPOINT https://hf-mirror.com
python scripts/download_esm3_model.py
```

### Q2: 磁盘空间不够

**A:** ESM3-sm 约需要 **2 GB** 空间：
- 模型权重: ~1.5 GB
- 配置文件: ~1 MB
- Tokenizer: ~500 KB

如果空间紧张，可以删除 `models/esm3_sm_open_v1.pth`（不完整的文件）。

### Q3: 能用 CPU 吗？

**A:** 可以，但 **很慢**：
- CPU (Mac): ~1-2 分钟/1000 序列
- MPS (Mac GPU): ~20-40 秒/1000 序列  
- CUDA (GPU): ~5-10 秒/1000 序列

建议先用小数据集测试（100 条），完整运行用 GPU。

---

## 下一步

模型加载成功后：

```fish
# 1. 提取 embeddings
python scripts/03_extract_esm3.py

# 2. 验证数据
python scripts/validate_data.py

# 3. 查看结果
ls -lh data/embeddings/
# 应该看到:
#   esm3_embeddings.npz  (~60-80 MB, shape: 10947 × 1536)
#   metadata.csv
```

---

## Linus 视角总结

**问题本质**：数据结构错了 - 你有零件，但不是完整产品。

**解决方案**：
1. **让系统自动处理**（方案 A） - 最简单，没有特殊情况
2. 不要试图手动组装模型文件 - 这是过度设计

**原则**：
- 用工具提供的标准方式（`from_pretrained`）
- 不要绕过工具自己造轮子
- 自动下载是 feature，不是 bug

---

**Last Updated**: 2025-10-27

