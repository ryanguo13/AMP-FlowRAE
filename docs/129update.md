以下为从数据集下载完成后的操作与问题小结（只读）：
1) 过滤库与数据准备
合并 UniProt + DRAMP + APD3 + DBAASP，清洗无效序列，构建 MMseqs2 库 data/processed/filter/filterdb（索引已建）。
负样本模板已整理并滑窗、去重，生成 negative_templates/master_toxin_4k.fasta（未收缩）。
2) 负样本准备脚本与输出
脚本：scripts/10_prepare_negative_samples.py
生成 FASTA：
UniProt 短肽（30–80 aa，人/哺乳动物关键词）：915 条
APOE 滑窗（90–160区间，50–70 aa，重复）：720 条
生成 embeddings（最新一次成功，非零）：
data/negative_samples/uniprot_short_peptides_embeddings.npz
data/negative_samples/apoe_windows_embeddings.npz
对应 metadata CSV 已生成。
3) 模型加载问题与修复
本地有 HFD 结构权重（data/weights/esm3_sm_open_v1.pth），但 ESM3.from_pretrained 仍尝试联网，因 SSL 证书错误多次失败。
最终：自动下载流程（仍失败校验证书）后再加载缓存/本地权重，模型成功加载；过程中若失败会再触发下载，但已可继续。
原因：ESM3 需要 HuggingFace 注册信息，即使权重在本地；证书校验失败时自动下载会报错，但最终完成。
调整：推理用 torch.amp.autocast(device_type='cuda', dtype=torch.float16) 替换原 bfloat16，避免 “unsupported ScalarType BFloat16” 全部失败的问题。
4) Embeddings 全零问题
早期因 bfloat16 不支持导致全部序列失败（填零）。修复为 float16 后重新提取，现已非零。统计示例：
UniProt embeddings: 915×1536，非零计数 1,405,440，均值约 -0.08，Std ≈ 243.8
APOE embeddings: 720×1536，非零计数 1,105,920，均值约 -0.15，Std ≈ 247.2
5) 现存状态
负样本 embeddings 就绪；正样本数据未动。
训练脚本尚未集成负样本 DataLoader 和三重防御（已有实现文件：rae/dataset_with_negatives.py, rae/loss_with_defense.py），但 rae/train.py 还未改。
生成后过滤脚本 scripts/11_filter_generated_sequences.py、评估脚本 scripts/12_evaluate_novelty.py 已就绪（未运行）。
建议的下一步
在 rae/train.py 接入 dataset_with_negatives.py 和 loss_with_defense.py，配置负样本权重与防御系数。
训练后用 scripts/11_filter_generated_sequences.py 过滤，再用 scripts/12_evaluate_novelty.py 评估新颖度/污染率/理化性质。