# 🐟 Fish Shell 快速命令

## 环境设置

```fish
# 进入项目目录
cd /Users/guojunhua/Documents/AMP-FlowRAE

# 激活 uv 环境
source .venv/bin/activate.fish

# 检查 Python
which python  # 应该指向 .venv/bin/python
```

---

## Phase 1 运行命令

### 一键运行
```fish
fish scripts/run_phase1.fish
```

### 分步运行
```fish
# Step 1: 解析 FASTA
python scripts/01_parse_fasta.py

# Step 2: 去冗余
python scripts/02_deduplicate.py

# Step 3: 提取 embeddings
python scripts/03_extract_esm3.py
```

---

## 检查进度

### 查看后台进程
```fish
ps aux | grep python | grep extract
```

### 查看输出文件
```fish
ls -lh data/processed/
ls -lh data/embeddings/
```

### 验证数据质量
```fish
python scripts/validate_data.py
```

---

## 环境变量 (Fish 语法)

```fish
# 设置 HuggingFace 镜像
set -x HF_ENDPOINT https://hf-mirror.com

# 设置 CUDA 设备
set -x CUDA_VISIBLE_DEVICES 0

# 查看所有环境变量
env | grep -E "(HF|CUDA|PYTHON)"
```

---

## 常用 Fish 命令

### 查看历史
```fish
history | grep python
```

### 后台运行
```fish
python scripts/03_extract_esm3.py &
# 或使用 nohup
nohup python scripts/03_extract_esm3.py > logs/extract.log 2>&1 &
```

### 杀死进程
```fish
# 查找进程 ID
ps aux | grep python

# 杀死进程
kill -9 <PID>
```

---

## uv 常用命令 (Fish)

```fish
# 创建虚拟环境
uv venv

# 激活环境
source .venv/bin/activate.fish

# 安装包
uv pip install <package>

# 安装项目依赖
uv pip install -e .

# 列出已安装包
uv pip list

# 生成 requirements
uv pip freeze > requirements.txt
```

---

## 数据处理快捷命令

### 查看 CSV 前几行
```fish
head data/processed/dedup_sequences.csv
```

### 统计序列数
```fish
wc -l data/processed/dedup_sequences.csv
```

### 查看 FASTA 数量
```fish
grep -c "^>" data/raw/dramp_Antimicrobial_amps.fasta
```

### 检查文件大小
```fish
du -sh data/embeddings/
```

---

## Debugging

### 查看完整错误
```fish
python scripts/03_extract_esm3.py 2>&1 | tee logs/debug.log
```

### 查看 Python 模块路径
```fish
python -c "import sys; print('\n'.join(sys.path))"
```

### 测试导入
```fish
python -c "import torch; import esm; print('OK')"
```

---

**Fish Shell 特性备注：**
- 使用 `set -x` 设置环境变量（不是 `export`）
- 使用 `source` 激活环境（不是 `.`）
- 条件判断: `if test -f file.txt; ...; end`
- 循环: `for i in (seq 1 10); ...; end`

---

**Last Updated:** 2025-10-27

