# Phase 3 计划 - Conditional Flow Matching

**状态：** 准备开始  
**预计时间：** 1-2 周

---

## 目标

学习条件分布 `p(z | charge, hydrophobicity, length)`，实现可控生成。

**输入：** 指定的物理化学属性  
**输出：** 对应的 latent vector z*  
**解码：** z* → embedding → 序列

---

## 技术路线

### 1. Flow Matching 简介

**核心思想：** 学习从 noise 到 data 的概率路径

```
t=0: z₀ ~ N(0, I)        (噪声)
t=1: z₁ ~ p_data         (真实 latent)

学习: v_θ(z_t, t, c) 使得 z_t 沿着路径从 z₀ 流向 z₁
```

**优势：**
- 比 Diffusion 更快（1-5 steps vs 50-1000 steps）
- 训练稳定
- 条件控制简单

### 2. 条件设置

**去掉 is_amp，使用连续属性：**

| 条件           | 类型  | 范围          | 归一化方式     |
|----------------|-------|---------------|----------------|
| charge         | 连续  | [-20, 30]     | MinMax [0, 1]  |
| hydrophobicity | 连续  | [-4.5, 4.2]   | MinMax [0, 1]  |
| length         | 连续  | [2, 100]      | MinMax [0, 1]  |

**为什么去掉 is_amp？**
- 全是 AMP，不需要这个条件
- ESM-3 不编码活性，is_amp 不可分
- 连续属性更有实际意义

### 3. 模型架构

**Flow Model:**
```python
class ConditionalFlowModel(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=3):
        # Input: (z_t, t, c)
        # z_t: (B, 64) latent at time t
        # t:   (B, 1)  time step
        # c:   (B, 3)  conditions [charge, hydro, length]
        
        # Output: v_t (B, 64) velocity field
        
        self.net = MLP([64 + 1 + 3, 256, 256, 64])
    
    def forward(self, z_t, t, c):
        x = torch.cat([z_t, t, c], dim=-1)
        v_t = self.net(x)
        return v_t
```

**简洁设计原则 (Linus)：**
- 先用简单 MLP，能用再优化
- 不搞 Transformer、Attention 等复杂结构
- 3 层 MLP 足够

### 4. 训练流程

**Flow Matching Loss:**
```python
# 1. Sample data latent
z_1 = sample_from_data()  # (B, 64)

# 2. Sample noise
z_0 = torch.randn_like(z_1)  # (B, 64)

# 3. Sample time
t = torch.rand(B, 1)  # (B, 1) in [0, 1]

# 4. Interpolate
z_t = t * z_1 + (1 - t) * z_0

# 5. Compute target velocity
v_target = z_1 - z_0

# 6. Predict velocity
v_pred = model(z_t, t, c)

# 7. Loss
loss = MSE(v_pred, v_target)
```

**超参数：**
- Epochs: 100
- Batch size: 512
- Learning rate: 1e-4
- Optimizer: AdamW

### 5. 采样流程

**ODE 求解（生成）：**
```python
# 1. Start from noise
z_0 = torch.randn(B, 64)

# 2. Given conditions
c = [charge, hydrophobicity, length]

# 3. Solve ODE from t=0 to t=1
for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:  # 5 steps
    v_t = model(z_t, t, c)
    z_t = z_t + dt * v_t

# 4. z_1 is the generated latent
return z_1
```

---

## 数据准备

### 输入文件

```
outputs/rae_amp_only_eval/
├── train_latents.npz    # z vectors (8,392 × 64)
├── val_latents.npz      # z vectors (1,049 × 64)
└── test_latents.npz     # z vectors (1,050 × 64)

data/embeddings/
└── metadata_amp_only.csv  # conditions
```

### Dataloader

```python
class LatentConditionDataset:
    def __init__(self, latents_npz, metadata_csv, indices):
        self.latents = np.load(latents_npz)["latents"]
        self.metadata = pd.read_csv(metadata_csv)
        self.indices = indices
        
        # Normalize conditions to [0, 1]
        self.charge_norm = MinMaxScaler()
        self.hydro_norm = MinMaxScaler()
        self.length_norm = MinMaxScaler()
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        z = self.latents[real_idx]
        
        # Get conditions
        row = self.metadata.iloc[real_idx]
        charge = self.charge_norm(row["charge"])
        hydro = self.hydro_norm(row["hydrophobicity"])
        length = self.length_norm(row["length"])
        
        c = [charge, hydro, length]
        return z, c
```

---

## 评估指标

| 指标                    | 目标值  | 含义                          |
|-------------------------|---------|-------------------------------|
| Condition Correlation   | > 0.7   | 生成的 z* 与目标条件的相关性  |
| Reconstruction Quality  | < 0.1   | z* 经 RAE 解码后的质量        |
| Diversity               | High    | 生成样本的多样性              |
| Validity                | > 95%   | 生成序列的有效性              |

---

## 实现步骤

### Step 1: 数据加载 (今天)
```python
flow_matching/dataset.py
- LatentConditionDataset
- condition normalization
- dataloader
```

### Step 2: 模型实现 (今天)
```python
flow_matching/model.py
- ConditionalFlowModel
- forward pass
- loss computation
```

### Step 3: 训练脚本 (明天)
```python
flow_matching/train.py
- training loop
- validation
- checkpointing
```

### Step 4: 采样脚本 (明天)
```python
flow_matching/sample.py
- ODE solver
- conditional sampling
- save generated latents
```

### Step 5: 评估 (后天)
```python
flow_matching/evaluate.py
- condition correlation
- diversity metrics
- visualization
```

---

## 预期结果

**生成示例：**

```python
# 指定条件
charge = 8.0              # 高正电荷
hydrophobicity = -1.5     # 中等疏水性
length = 25               # 25 个氨基酸

# 生成
z_generated = flow_model.sample(charge, hydrophobicity, length)

# 解码
embedding = rae_decoder(z_generated)
sequence = decode_to_sequence(embedding)

# 验证
assert compute_charge(sequence) ≈ 8.0
assert compute_hydrophobicity(sequence) ≈ -1.5
assert len(sequence) ≈ 25
```

---

## 参考文献

1. **Flow Matching for Generative Modeling**  
   Lipman et al., ICLR 2023

2. **Conditional Flow Matching**  
   Tong et al., ICML 2023

3. **ProtFlow: Flow Models for Protein Design**  
   Lu et al., 2024

---

**准备就绪，开始 Phase 3！**


