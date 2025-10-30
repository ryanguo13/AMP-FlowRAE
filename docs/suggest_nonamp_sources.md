# Non-AMP 数据源建议

如果需要平衡数据集，可以从以下来源收集更多 Non-AMP 短肽：

## 1. UniProt 短肽

### Signal Peptides (信号肽)
- **来源**: UniProt (Keyword: Signal)
- **特点**: 长度 15-30 aa，无抗菌活性
- **数量**: 数万条
- **API**: https://rest.uniprot.org/uniprotkb/search

```bash
# 示例查询
(keyword:signal) AND (length:[15 TO 50]) AND (reviewed:true)
```

### Hormone Peptides (激素肽)
- **来源**: UniProt (Keyword: Hormone)
- **特点**: 生物活性肽，非抗菌
- **示例**: 胰岛素、胰高血糖素

### Neuropeptides (神经肽)
- **来源**: NeuroPep database
- **特点**: 神经调节功能，非抗菌

---

## 2. 专门的 Negative 数据集

### AmPEP Negative Set
- **论文**: Bhadra et al., 2018
- **来源**: https://github.com/tlawrence3/amPEP
- **数量**: ~10,000 条
- **特点**: 专门用于 AMP 分类的阴性对照

### dbAMP v2.0 Non-AMP
- **来源**: http://csb.cse.yzu.edu.tw/dbAMP/
- **特点**: 人工标注的非抗菌肽

---

## 3. 随机短肽（不推荐）

从蛋白质数据库随机提取短片段：
- **风险**: 可能包含未知的抗菌活性
- **优点**: 数量无限

---

## 4. 合成负样本（谨慎使用）

### Shuffled AMPs
- 打乱 AMP 序列
- **风险**: 可能仍有活性

### Low-charge peptides
- 净电荷 < 0 的短肽
- **假设**: 低电荷 → 低抗菌活性（不总是对的）

---

## 推荐策略

**Option 1: 快速方案**
1. 下载 AmPEP negative set (~10k)
2. 过滤到长度 < 100 aa
3. 去重后合并到现有数据

**Option 2: 严谨方案**
1. 从 UniProt 收集 Signal peptides
2. 从 dbAMP 获取标注的 Non-AMP
3. 人工验证无重叠
4. 目标：收集 ~5000 条 Non-AMP

**Option 3: 最简方案**
- 不收集 Non-AMP
- 专注连续属性生成（推荐！）

---

## 数据收集脚本模板

```python
#!/usr/bin/env python3
"""
从 UniProt 获取 Non-AMP 短肽
"""

import requests
import time

def fetch_uniprot_peptides(query, max_results=5000):
    """从 UniProt API 获取短肽"""
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    
    params = {
        "query": query,
        "format": "fasta",
        "size": 500,  # per page
    }
    
    sequences = []
    
    for offset in range(0, max_results, 500):
        params["offset"] = offset
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            sequences.append(response.text)
            time.sleep(1)  # 避免 rate limit
        else:
            break
    
    return "".join(sequences)

# 示例：获取信号肽
query = "(keyword:signal) AND (length:[15 TO 50]) AND (reviewed:true)"
fasta = fetch_uniprot_peptides(query, max_results=5000)

with open("signal_peptides.fasta", "w") as f:
    f.write(fasta)
```

---

**但再次强调：如果目标是生成 AMP，不需要收集更多 Non-AMP。**

