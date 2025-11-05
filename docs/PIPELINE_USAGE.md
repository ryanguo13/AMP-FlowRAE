# Pipeline 使用指南

AMP-FlowRAE 提供了三种 pipeline 脚本，适用于不同场景。

---

## 📋 脚本概览

| 脚本 | 用途 | 特点 |
|------|------|------|
| `pipeline.fish` | **标准完整流程** | 一次性运行所有步骤，参数固定 |
| `run_pipeline.fish` | **灵活运行** | 支持选择性运行 Phase，推荐使用 |
| `quick_test.fish` | **快速测试** | 少量 epochs，快速验证代码改动 |

---

## 🚀 使用方法

### 1. 环境准备

```fish
# 首次运行，配置环境
fish scripts/setup_env.fish

# 激活环境（每次新终端需要）
source .venv/bin/activate.fish
```

### 2. 选择运行模式

#### 方式 A: 完整运行（推荐新手）

```fish
# 赋予执行权限
chmod +x pipeline.fish

# 一键运行全部
./pipeline.fish
```

**适用场景：**
- 第一次运行整个项目
- 从零开始完整训练
- 不需要调试

---

#### 方式 B: 灵活运行（推荐日常使用）

```fish
chmod +x run_pipeline.fish

# 运行所有 Phase
./run_pipeline.fish

# 只运行 Phase 1 (数据准备)
./run_pipeline.fish 1

# 只运行 Phase 2 和 3 (跳过数据准备)
./run_pipeline.fish 2 3

# 只运行 Phase 4 (重新分析已有结果)
./run_pipeline.fish 4
```

**适用场景：**
- 数据已准备好，只想重新训练模型
- 模型已训练，只想重新生成序列
- 调整某个 Phase 的参数后重新运行

**Phase 说明：**
- **Phase 1:** 数据准备（解析 FASTA, 去重, 提取 embeddings）
- **Phase 2:** RAE 训练（50 epochs, batch 256）
- **Phase 3:** Flow Matching 训练 + 采样（100 epochs, 500 samples）
- **Phase 4:** 序列解码 + 质量分析

---

#### 方式 C: 快速测试（推荐调试）

```fish
chmod +x quick_test.fish

# 快速测试全流程
./quick_test.fish

# 快速测试 Phase 2-4（跳过数据准备）
./quick_test.fish 2 3 4
```

**测试参数：**
- RAE: 10 epochs (vs. 正式 50)
- Flow: 20 epochs (vs. 正式 100)
- Samples: 100 个 (vs. 正式 500)
- ODE steps: 50 (vs. 正式 100)

**适用场景：**
- 修改了模型代码，快速验证是否正常运行
- 调整超参数前的初步测试
- 快速迭代实验

**输出目录：**
- `outputs/rae_test/`
- `outputs/flow_test/`
- `outputs/analysis_test/`

---

## 📊 查看结果

### TensorBoard 可视化

```fish
# 查看训练曲线
tensorboard --logdir outputs/

# 只看 RAE 训练
tensorboard --logdir outputs/rae/logs

# 只看 Flow 训练
tensorboard --logdir outputs/flow/logs
```

### 分析结果

```fish
# 查看生成序列统计
head -20 outputs/analysis/generated_amps_analyzed.csv

# 查看可视化图表
open outputs/analysis/*.png
```

---

## 🎯 常见使用场景

### 场景 1: 完整运行一次

```fish
# 首次运行
./run_pipeline.fish
```

**预计时间：**
- Phase 1: 20-40 分钟（ESM-3 提取最慢）
- Phase 2: 2-5 分钟（RAE 训练）
- Phase 3: 5-10 分钟（Flow 训练 + 采样）
- Phase 4: 1-2 分钟（解码 + 分析）

**总计：** 约 30-60 分钟

---

### 场景 2: 调整 RAE 参数重新训练

修改 `run_pipeline.fish` 中 Phase 2 的参数：

```fish
# 例如改为 100 epochs, batch 512
run_step "训练 RAE" \
    python rae/train.py \
        --epochs 100 \
        --batch-size 512 \
        --lr 5e-4 \
        --output-dir outputs/rae_v2
```

然后运行：

```fish
# 只运行 Phase 2-4
./run_pipeline.fish 2 3 4
```

---

### 场景 3: 生成更多序列

修改 Phase 3 的采样数量：

```fish
# 在 run_pipeline.fish 中修改
--num-samples 2000  # 改为 2000
```

然后运行：

```fish
# 只重新采样和解码
./run_pipeline.fish 3 4
```

---

### 场景 4: 快速验证代码改动

修改了 `rae/model.py` 后：

```fish
# 快速测试是否正常运行
./quick_test.fish 2

# 如果测试通过，正式训练
./run_pipeline.fish 2 3 4
```

---

## 🛠️ 自定义参数

### 修改训练参数

直接编辑 `run_pipeline.fish` 中对应的命令：

```fish
# 例如：修改 Flow Matching 参数
run_step "训练 Flow Matching" \
    python flow_matching/train.py \
        --epochs 200 \           # 增加到 200 epochs
        --batch-size 1024 \      # 更大的 batch
        --lr 5e-5 \              # 更小的学习率
        --output-dir outputs/flow_large
```

### 修改采样条件

```fish
# 在 Phase 3 采样步骤中
run_step "采样 latent vectors" \
    python flow_matching/sample.py \
        --checkpoint outputs/flow/checkpoints/best.pt \
        --num-samples 1000 \
        --charge-range 3 10 \        # 指定电荷范围
        --length-range 15 35 \       # 指定长度范围
        --hydro-range -0.5 0.5 \     # 指定疏水性范围
        --output-dir outputs/samples_custom
```

---

## ⚠️ 常见问题

### Q1: Pipeline 中途失败怎么办？

```fish
# 从失败的 Phase 继续
./run_pipeline.fish 3 4  # 例如从 Phase 3 继续
```

### Q2: 想重新生成序列但不重新训练？

```fish
# 只运行 Phase 3-4
./run_pipeline.fish 3 4
```

### Q3: 数据已准备好，想测试不同模型参数？

```fish
# 编辑 run_pipeline.fish 修改参数
# 然后只运行模型训练部分
./run_pipeline.fish 2 3 4
```

### Q4: 如何并行测试多组参数？

```fish
# 方法 1: 修改输出目录
python rae/train.py --epochs 50 --output-dir outputs/rae_exp1 &
python rae/train.py --epochs 100 --output-dir outputs/rae_exp2 &

# 方法 2: 创建多个测试脚本
cp run_pipeline.fish experiment1.fish
cp run_pipeline.fish experiment2.fish
# 分别修改参数和输出目录
```

---

## 📈 监控训练进度

### 实时查看 Loss

```fish
# 方法 1: TensorBoard（推荐）
tensorboard --logdir outputs/

# 方法 2: 查看日志文件
tail -f outputs/rae_train.log
tail -f outputs/flow_train.log

# 方法 3: 直接看终端输出
# Pipeline 会实时显示进度条和 loss
```

---

## 🔧 高级用法

### 手动运行单个脚本

```fish
# 数据准备
python scripts/01_parse_fasta.py
python scripts/02_deduplicate.py
python scripts/03_extract_esm3.py

# RAE 训练
python rae/train.py --epochs 50 --batch-size 256

# Flow Matching
python flow_matching/train.py --epochs 100

# 采样
python flow_matching/sample.py \
    --checkpoint outputs/flow/checkpoints/best.pt \
    --num-samples 500

# 解码
python decoder/decode.py \
    --latents outputs/samples/sampled_latents.npy \
    --output outputs/decoded.csv

# 分析
python scripts/09_analyze_generated.py \
    --input outputs/decoded.csv
```

### 批量实验

创建实验配置文件 `experiments.fish`:

```fish
#!/usr/bin/env fish

# 实验 1: 小模型
python rae/train.py --latent-dim 32 --output-dir outputs/exp1_dim32 &
set exp1_pid $last_pid

# 实验 2: 中等模型
python rae/train.py --latent-dim 64 --output-dir outputs/exp2_dim64 &
set exp2_pid $last_pid

# 实验 3: 大模型
python rae/train.py --latent-dim 128 --output-dir outputs/exp3_dim128 &
set exp3_pid $last_pid

# 等待所有实验完成
wait $exp1_pid $exp2_pid $exp3_pid

echo "所有实验完成！"
```

---

## 💡 最佳实践

1. **首次运行**: 使用 `quick_test.fish` 快速验证环境
2. **正式训练**: 使用 `run_pipeline.fish` 灵活控制
3. **调试代码**: 使用 `quick_test.fish` 快速迭代
4. **监控训练**: 开启 TensorBoard 实时查看
5. **保存结果**: 不同实验使用不同的 `--output-dir`
6. **版本管理**: 重要实验打 git tag

---

## 📚 进一步阅读

- **数据准备细节**: `docs/PHASE1_COMPLETE.md`
- **RAE 训练细节**: `docs/PHASE2_COMPLETE.md`
- **Flow Matching 细节**: `docs/PHASE3_COMPLETE.md`
- **解码与分析**: `docs/PHASE4_COMPLETE.md`
- **项目总览**: `projectplan.md`

---

**更新时间:** 2025-10-29  
**维护者:** AMP-FlowRAE Team



