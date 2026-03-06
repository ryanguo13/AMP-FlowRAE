## 任务与待办

- 梳理负样本与黑名单策略，明确训练和过滤两阶段的改动顺序（含 general 黑名单方案）。
- 盘点现有数据：`data/raw/AMP_from_5databases.xlsx` 已含三套 AMP 基准库和若干补充库，确认文件清单与格式。
- 补齐负样本来源（人源短肽、APOE 滑窗、家族模板库），定义抽取/滑窗/去重规范。
- 设计训练时的三重防御参数（阈值、权重、oversample 比例），并标注代码落点。
- 设计生成后过滤规则与工具链（MMseqs2/BLAST 阈值、motif 正则、可选酶切位点过滤）。
- 确认评估指标与验证脚本需求（新颖度、污染率、理化性质）。

## 数据准备清单（下载或确认存在）

- 已有：`data/raw/AMP_from_5databases.xlsx`（三套 AMP 基准库 + 其他补充库），需列目录确认哪些文件可直接用作正样本/过滤库。
- UniProtKB/Swiss-Prot reviewed 全集（已下载：`data/raw/uniprot_sprot.fasta`）：用于提取 30–80 aa 人源/哺乳动物短肽负样本。
- 人类 APOE 全长序列：生成 50–70 aa 滑动窗口负样本（训练时高权重重复）。
- 负样本家族模板：
  - Cathelicidin（keyword:"Cathelicidin"）
  - Histatin & Histone H2A/H4 25–60 aa 片段（人源）
  - Hepcidin（keyword:"Hepcidin"）
  - α/β-defensin 滑动窗口
  - Buforin / Thrombocidin 手工集合
- AMP 正样本基准库：APD3、DRAMP、DBAASP（如需更新至最新版本则重新下载）。
- 生成后过滤参考库：合并 UniProtKB + DRAMP + APD3 + DBAASP（供 MMseqs2/BLAST），注意 UniProt 全集只作负样本/过滤，不作正样本。
- 可选：蛋白酶切位点规则表（KR/KK/RR 等）用于后过滤。

## General 方案（黑名单＋白名单＋三重防御）

- 目标：生成 10k 序列时，≥97% 序列对 UniProtKB + DRAMP + APD3 + DBAASP 的最大 identity < 70%，同时避免 APOE/高危家族污染。
- 黑名单（一次性建全局毒瘤模板库并长期复用）：
  - APOE 家族：人类＋哺乳动物 APOE 全长滑窗 50–70 aa → `negative_templates/apoe.fasta`
  - Cathelicidin 家族：UniProt keyword:"Cathelicidin" 全物种 → `negative_templates/cathelidin.fasta`
  - Histatin & Histone H2A/H4：人源 25–60 aa 片段 → `negative_templates/histone_h2a_h4.fasta`
  - Hepcidin 家族：keyword:"Hepcidin" → `negative_templates/hepcidin.fasta`
  - α/β-defensin cryptic：全家族滑窗 → `negative_templates/defensin_cryptic.fasta`
  - Buforin / Thrombocidin：手工收录 ~100 条 → `negative_templates/buforin_thrombo.fasta`
  - 合并：`negative_templates/master_toxin_4k.fasta`，训练时 DataLoader 里 oversample 20–30×。
- 白名单（正样本）：APD3/DRAMP/DBAASP 主体 + 自有 AMP_from_5databases.xlsx 里确认的正样本字段。
- 训练防御：沿用三重防御（NLL 罚项、latent classifier、MinMax push-away），在损失里对 master_toxin_4k 的 latent 也可加 push-away。
- 生成后过滤：先 MMseqs2/BLAST（identity ≥75% 或 coverage ≥80% 直接淘汰），再 APOE motif 正则，最后可选酶切位点过滤。

## 下载与准备操作占位（仅列来源与命令示意，先确认再执行）

- UniProtKB/Swiss-Prot reviewed（已完成）：`data/raw/uniprot_sprot.fasta`；来源：https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
- APOE 全长序列（人类）（已完成）：UniProt P02649 FASTA，`data/raw/apoe_full.fasta`。
- 家族黑名单模板（均已下载，除 Buforin/Thrombocidin 手工集）：
  - Cathelicidin：`data/raw/cathelicidin.fasta`
  - Histatin & Histone H2A/H4：`data/raw/histone_h2a_h4.fasta`
  - Hepcidin：`data/raw/hepcidin.fasta`
  - α/β-defensin：`data/raw/defensin_raw.fasta`
  - Buforin/Thrombocidin：`data/raw/buforin_thrombo.fasta`
  - 已生成：滑窗/合并/去重输出
    - `negative_templates/apoe_windows_50_70.fasta`
    - `negative_templates/defensin_cryptic.fasta`
    - `negative_templates/master_raw.fasta`
    - `negative_templates/master_toxin_4k.fasta`（当前 ~730 万条，暂不收缩）
- AMP 正样本（已完成）：
  - APD3: `data/raw/naturalAMPs_APD2024a.fasta` ✅
  - DRAMP: `data/raw/dramp_Antimicrobial_amps.fasta`, `data/raw/dramp_general_amps.fasta` ✅
  - DBAASP: `data/raw/peptides-fasta.fasta` ✅
- 过滤用合并库构建（已完成）：
  - 合并文件：`data/processed/filter/merged.fasta`（602,288 条有效序列）✅
  - MMseqs 数据库：`data/processed/filter/filterdb`（已创建并索引）✅
  - 包含：UniProtKB + DRAMP + APD3 + DBAASP

## 训练阶段设计（不写代码，先定规范）

- 负样本构建与 oversample：
  - UniProt 短肽（30–80 aa，哺乳动物优先）抽取，下采样到 8–10 万条。
  - APOE 90–160 位滑窗 50–70 aa，重复 20×权重加入训练。
  - 已知高同源模板（LL-37、Magainin、Cecropin、Indolicidin、Buforin 等）各复制 5–10×。
  - 训练集比例参考：正样本 3–4 万，负样本 8–10 万。
- 三重防御：
  - NLL 罚项：重建序列与 UniProt 最大 identity > 85% 时，增加 10×MSE（落点 `rae/loss.py`）。
  - Latent classifier 正则：用 `models/esm3-sm-open-v1` 判定“人源蛋白像”，在 z 上加 `||classifier(z)||²`（落点 `rae/classifier_head.py`）。
  - MinMax / push-away：对 APOE latent 施加 `λ * max(0, 0.9 - cos(z, z_apoe))`（RAE loss 内）。
  - DataLoader 采样建议：正样本 1×，UniProt 短肽 2–3×，master_toxin_4k 3–5×，APOE 滑窗子集额外 10×。

## 生成后过滤（规范与阈值）

- 同源过滤：MMseqs2/BLAST 对合并库，identity ≥ 75% 或 coverage ≥ 80% 直接淘汰。
- APOE motif 正则：如 `LR.KLRK.LLR` 或 `RLAVY` 与 `PLVEDM` 相隔 ≤15 aa。
- 可选酶切位点过滤：KR/KK/RR 频繁的序列下调或剔除。

## 过滤库构建（MMseqs2/BLAST）

- 合并源：`data/raw/uniprot_sprot.fasta` + DRAMP + APD3 + DBAASP（路径待明确，来自 AMP_from_5databases.xlsx 同步版本）。
- 建库（示例，先存占位命令，不执行）：
  - MMseqs2：`mmseqs createdb merged.fasta filterdb && mmseqs createindex filterdb tmp --threads 16`
  - BLAST：`makeblastdb -in merged.fasta -dbtype prot -out filterdb`
- 同源判定：identity ≥75% 或 coverage ≥80% 即淘汰；评估时用 identity<70% 统计新颖度。

### 合并与建库命令占位（后续可直接执行）

- 合并 FASTA：
  - `cat data/raw/uniprot_sprot.fasta data/raw/DRAMP.fasta data/raw/APD3.fasta data/raw/DBAASP.fasta > data/processed/filter/merged.fasta`
- MMseqs2：
  - `mmseqs createdb data/processed/filter/merged.fasta data/processed/filter/filterdb`
  - `mmseqs createindex data/processed/filter/filterdb data/processed/filter/tmp --threads 16`
- BLAST：
  - `makeblastdb -in data/processed/filter/merged.fasta -dbtype prot -out data/processed/filter/filterdb`

## 生成后过滤脚本要点

- 管线顺序：MMseqs2/BLAST 同源过滤 → APOE motif 正则 → （可选）酶切位点过滤。
- 输出分层：保留命中过滤阈值的列表以便统计污染率；最终候选再做理化性质检查。

### 生成后过滤逻辑占位（伪流程）

- 输入：候选 FASTA
- 步骤：
  1. MMseqs2 search: `mmseqs easy-search candidates.fasta filterdb hits.m8 tmp --min-seq-id 0.75 --cov-mode 1 -c 0.8`
  2. 过滤：命中 identity≥75% 或 coverage≥80% 的序列标记淘汰；未命中进入下一步
  3. 正则：APOE motif (`LR.KLRK.LLR` 或 `RLAVY.*PLVEDM` 距离≤15) 淘汰
  4. 可选：KR/KK/RR 高频酶切位点淘汰或降权
  5. 输出：保留集 + 淘汰原因列表；统计污染率、新颖度

## 评估与验证需求

- 新颖度：对合并库最大 identity < 70% 的占比。
- APOE 污染率：命中 APOE 滑窗或 motif 的比例。
- 理化性质：基本理化特征分布是否在期望范围（长度、电荷、疏水性等）。

## 代码实现状态

### ✅ 已完成

1. **负样本数据准备脚本** (`scripts/10_prepare_negative_samples.py`)
   - UniProt 短肽提取（30-80 aa，人源/哺乳动物）
   - APOE 滑窗生成（50-70 aa，重复 20×）
   - ESM-3 embeddings 提取

2. **负样本数据集** (`rae/dataset_with_negatives.py`)
   - 支持正负样本混合
   - 支持负样本 oversample（可配置权重）

3. **三重防御 Loss** (`rae/loss_with_defense.py`)
   - NLL penalty：与 UniProt 高同源时惩罚
   - Classifier-guided regularization：分类器引导的 latent 正则化
   - APOE push-away：最大化与 APOE latent 的距离

4. **生成后过滤脚本** (`scripts/11_filter_generated_sequences.py`)
   - MMseqs2 同源性过滤（identity ≥ 75% 或 coverage ≥ 80%）
   - APOE motif 正则过滤
   - 蛋白酶切位点过滤（可选）

5. **评估脚本** (`scripts/12_evaluate_novelty.py`)
   - 新颖度评估（MMseqs2）
   - APOE 污染率统计
   - 理化性质分析

### 📝 使用说明

1. **准备负样本数据**：
   ```bash
   python scripts/10_prepare_negative_samples.py \
       --uniprot-fasta data/raw/uniprot_sprot.fasta \
       --apoe-fasta data/raw/apoe_full.fasta \
       --output-dir data/negative_samples \
       --device cuda
   ```

2. **训练 RAE（带负样本和防御）**：
   - 修改 `rae/train.py` 使用 `dataset_with_negatives.py` 和 `loss_with_defense.py`
   - 配置负样本权重和防御系数

3. **生成后过滤**：
   ```bash
   python scripts/11_filter_generated_sequences.py \
       --input-fasta outputs/generated_sequences.fasta \
       --output-fasta outputs/filtered_sequences.fasta \
       --mmseqs-db data/processed/filter/filterdb \
       --filter-apoe \
       --filter-protease
   ```

4. **评估新颖度**：
   ```bash
   python scripts/12_evaluate_novelty.py \
       --input-fasta outputs/filtered_sequences.fasta \
       --mmseqs-db data/processed/filter/filterdb \
       --output-csv outputs/evaluation_results.csv
   ```


