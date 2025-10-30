# 🧬 AMP-FlowRAE Pipeline 脚本

简洁的 Fish 脚本集合，用于运行完整的 AMP 生成 Pipeline (Phase 1-4)。

---

## 📋 脚本概览

我为你创建了 **4 个版本** 的 pipeline 脚本，各有特点：

| 脚本 | 特点 | 使用场景 | 推荐 |
|------|------|----------|------|
| **pipeline_simple.fish** | 最简洁，通过注释控制 Phase | 日常使用，快速屏蔽某些 Phase | ⭐⭐⭐⭐⭐ |
| **run_pipeline.fish** | 灵活，通过命令参数控制 | 命令行快速选择运行哪些 Phase | ⭐⭐⭐⭐ |
| **quick_test.fish** | 快速测试版（少量 epochs） | 验证代码改动，快速迭代 | ⭐⭐⭐ |
| **pipeline.fish** | 完整固定版 | 一键运行全部，不需要调整 | ⭐⭐ |

---

## 🚀 快速开始

### 1️⃣ 环境准备（仅首次）

```fish
# 配置环境
fish scripts/setup_env.fish

# 激活环境
source .venv/bin/activate.fish
```

### 2️⃣ 选择并运行

#### 推荐：使用 pipeline_simple.fish

```fish
# 直接运行全部
./pipeline_simple.fish

# 或编辑文件，注释掉不需要的 Phase，然后运行
vim pipeline_simple.fish  # 或 code pipeline_simple.fish
./pipeline_simple.fish
```

**如何注释掉某个 Phase：**

打开 `pipeline_simple.fish`，找到要跳过的 Phase，整块注释掉：

```fish
# ============================================================================
# Phase 1: 数据准备
# 如果数据已准备好，注释掉这整个块
# ============================================================================
# echo "━━━━ Phase 1: 数据准备 ━━━━"
# python scripts/01_parse_fasta.py
# python scripts/02_deduplicate.py
# python scripts/03_extract_esm3.py
# python scripts/04_split_data.py
# python scripts/05_compute_labels.py
# python scripts/06_normalize_embeddings.py
# python scripts/validate_data.py
# echo "✅ Phase 1 完成"
# echo ""
```

保存后运行即可跳过 Phase 1。

---

#### 或者：使用 run_pipeline.fish（命令行参数）

```fish
# 运行全部
./run_pipeline.fish

# 只运行 Phase 1
./run_pipeline.fish 1

# 只运行 Phase 2 和 3
./run_pipeline.fish 2 3

# 只运行 Phase 4（重新分析已有结果）
./run_pipeline.fish 4
```

---

#### 快速测试：使用 quick_test.fish

```fish
# 快速测试全流程（少量 epochs）
./quick_test.fish

# 快速测试部分流程
./quick_test.fish 2 3 4
```

测试参数：
- RAE: 10 epochs (vs. 正式 50)
- Flow: 20 epochs (vs. 正式 100)
- Samples: 100 个 (vs. 正式 500)

---

## 📊 Pipeline 阶段说明

| Phase | 名称 | 主要任务 | 预计时间 | 输出 |
|-------|------|----------|----------|------|
| **1** | 数据准备 | 解析 FASTA, 去重, 提取 ESM-3 embeddings | 20-40 分钟 | `data/embeddings/` |
| **2** | RAE 训练 | 学习潜空间 (1536→64 维) | 2-5 分钟 | `outputs/rae/` |
| **3** | Flow Matching | 训练条件生成 + 采样 latent | 5-10 分钟 | `outputs/flow/`, `outputs/samples/` |
| **4** | 解码分析 | 解码序列 + 质量评估 | 1-2 分钟 | `outputs/decoded_sequences.csv`, `outputs/analysis/` |

**总时间：** 约 30-60 分钟（完整运行）

---

## 🎯 典型使用场景

### 场景 1: 首次完整运行

```fish
./pipeline_simple.fish
```

### 场景 2: 数据已准备，只重新训练模型

```fish
# 方式 1: 编辑 pipeline_simple.fish，注释掉 Phase 1

# 方式 2: 使用参数
./run_pipeline.fish 2 3 4
```

### 场景 3: 调整 RAE 参数，重新训练

1. 编辑 `pipeline_simple.fish` 或 `run_pipeline.fish`
2. 修改 Phase 2 的参数（epochs, batch-size, lr）
3. 注释掉 Phase 1
4. 运行脚本

### 场景 4: 生成更多/不同的序列

1. 编辑 Phase 3 的采样参数（num-samples, charge-range 等）
2. 注释掉 Phase 1, 2
3. 运行脚本

### 场景 5: 快速验证代码改动

```fish
# 修改代码后
./quick_test.fish 2    # 快速测试是否正常

# 如果通过，正式训练
./run_pipeline.fish 2 3 4
```

---

## 🔧 参数调整

所有脚本都可以直接编辑来调整参数。

### Phase 2: RAE 训练参数

```fish
python rae/train.py \
    --epochs 50 \           # 训练轮数 (10-100)
    --batch-size 256 \      # 批大小 (128-512)
    --lr 1e-3 \             # 学习率 (1e-4 到 1e-2)
    --latent-dim 64 \       # 潜空间维度 (32-128)
    --output-dir outputs/rae_v2  # 不同实验用不同输出目录
```

### Phase 3: Flow Matching 参数

```fish
python flow_matching/train.py \
    --epochs 100 \          # 训练轮数 (20-200)
    --batch-size 512 \      # 批大小 (256-1024)
    --lr 1e-4 \             # 学习率 (1e-5 到 1e-3)
    --output-dir outputs/flow_v2

python flow_matching/sample.py \
    --num-samples 1000 \    # 生成数量 (100-5000)
    --ode-steps 100 \       # ODE 步数 (50-200)
    --mode grid \           # grid 或 random
    --charge-range 3 10 \   # 可选：指定电荷范围
    --length-range 15 35    # 可选：指定长度范围
```

---

## 📈 监控训练

### TensorBoard（推荐）

```fish
# 查看所有训练曲线
tensorboard --logdir outputs/

# 或分别查看
tensorboard --logdir outputs/rae/logs
tensorboard --logdir outputs/flow/logs
```

### 查看日志

```fish
tail -f outputs/rae_train.log
tail -f outputs/flow_train.log
```

---

## 📊 查看结果

### 生成序列统计

```fish
head -20 outputs/analysis/generated_amps_analyzed.csv
```

### 可视化

```fish
open outputs/analysis/*.png
```

### 筛选最佳候选

```python
import pandas as pd
df = pd.read_csv('outputs/analysis/generated_amps_analyzed.csv')

# 高活性 + 高新颖性
top = df[
    (df['activity_score'] > 0.85) & 
    (df['nn_distance'] > 0.5)
].sort_values('activity_score', ascending=False)

print(top[['sequence', 'activity_score', 'charge', 'length']].head(20))
```

---

## 💡 最佳实践

1. **首次运行**：用 `quick_test.fish` 验证环境
2. **正式训练**：用 `pipeline_simple.fish` 或 `run_pipeline.fish`
3. **调试代码**：用 `quick_test.fish` 快速迭代
4. **不同实验**：修改 `--output-dir` 避免覆盖
5. **监控训练**：开启 TensorBoard
6. **保存成果**：重要实验打 git tag

---

## 🛠️ 高级用法

### 批量实验

创建 `experiments.fish`:

```fish
#!/usr/bin/env fish

# 实验 1: latent_dim=32
python rae/train.py --latent-dim 32 --output-dir outputs/exp_dim32

# 实验 2: latent_dim=64
python rae/train.py --latent-dim 64 --output-dir outputs/exp_dim64

# 实验 3: latent_dim=128
python rae/train.py --latent-dim 128 --output-dir outputs/exp_dim128
```

### 并行运行

```fish
# 后台运行
python rae/train.py --output-dir outputs/exp1 &
python rae/train.py --epochs 100 --output-dir outputs/exp2 &

# 等待完成
wait
```

---

## ⚠️ 常见问题

### Q: Pipeline 中途失败？

```fish
# 从失败的 Phase 继续
./run_pipeline.fish 3 4
```

### Q: 想改参数重新训练？

1. 编辑脚本修改参数
2. 注释掉已完成的 Phase
3. 运行脚本

### Q: 如何并行测试多组参数？

```fish
# 方法 1: 后台运行，不同输出目录
python rae/train.py --output-dir outputs/exp1 &
python rae/train.py --epochs 100 --output-dir outputs/exp2 &

# 方法 2: 复制脚本，分别修改
cp pipeline_simple.fish exp1.fish
cp pipeline_simple.fish exp2.fish
# 分别编辑参数和输出目录
```

---

## 📚 详细文档

- **快速参考**: `PIPELINE_QUICKREF.md` ⭐ 推荐
- **完整指南**: `docs/PIPELINE_USAGE.md`
- **Phase 细节**: `docs/PHASE{1,2,3,4}_COMPLETE.md`
- **项目总览**: `projectplan.md`

---

## 🎓 脚本对比总结

### pipeline_simple.fish ⭐⭐⭐⭐⭐

**优点：**
- ✅ 最简洁直观
- ✅ 通过注释控制，可见性强
- ✅ 方便快速屏蔽某个 Phase
- ✅ 参数调整方便

**缺点：**
- ⚠️ 每次需要编辑文件

**推荐场景：** 日常使用，反复调整参数

---

### run_pipeline.fish ⭐⭐⭐⭐

**优点：**
- ✅ 命令行参数控制，灵活
- ✅ 无需编辑文件
- ✅ 适合快速切换运行不同 Phase

**缺点：**
- ⚠️ 调整训练参数仍需编辑文件

**推荐场景：** 命令行快速测试，CI/CD

---

### quick_test.fish ⭐⭐⭐

**优点：**
- ✅ 快速验证（少量 epochs）
- ✅ 适合调试代码改动
- ✅ 输出到独立目录，不影响正式训练

**缺点：**
- ⚠️ 训练不充分，仅供测试

**推荐场景：** 代码调试，快速迭代

---

### pipeline.fish ⭐⭐

**优点：**
- ✅ 一键运行，零配置
- ✅ 适合首次运行

**缺点：**
- ⚠️ 缺乏灵活性
- ⚠️ 被其他脚本完全覆盖

**推荐场景：** 首次完整运行

---

## 🎁 总结

**我的推荐：**

1. **日常使用** → `pipeline_simple.fish` （通过注释灵活控制）
2. **快速测试** → `run_pipeline.fish 2 3 4`（命令行参数）
3. **代码调试** → `quick_test.fish`（快速验证）

**典型工作流：**

```fish
# Day 1: 数据准备（一次性）
./run_pipeline.fish 1

# Day 2-N: 反复调整模型参数
# 编辑 pipeline_simple.fish，注释掉 Phase 1
# 修改 Phase 2/3 的参数
./pipeline_simple.fish

# 最终：选出最佳模型，生成大量序列
# 编辑 Phase 3 采样数量为 5000
./run_pipeline.fish 3 4
```

---

**创建时间:** 2025-10-29  
**维护者:** Linus 风格的简洁实用主义者 😎

祝你训练顺利！🚀


