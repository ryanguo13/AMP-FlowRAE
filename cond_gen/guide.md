条件生成(Conditional Flow/Conditional RAE)

思路:在 flow-matching 的采样器里把属性向量c(例如:[solubility_score,activity_scoretoxicity_score])作为条件输入，让模型学到 p(z|c)。采样时按目标c采样得到 z，再用 RAE decoder解码。
实现要点(repo 改动):
修改 f1ow_matching/mode1.py:让网络输入变为(t,x,c)或把c通过条件投影并与x拼接(或用FiLM/cross-attention)。
修改 flow_matching/sample.py:支持按用户给定条件采样(你README的--charge-range
等可扩展为 --target-solubility0.8)。训辫顳朋练时:在训练 batch 中附带对应属性向量(从 property predictor 或 ground-truth 标签)俎朙稞鳉腧潼点:直接控制，采样质量高;兼顾多属性。
鹾经机点:需要足够多带标签的样本来学习 p(zlc)。
