#!/usr/bin/env python3
"""
计算肽序列的理化性质

无需训练模型，直接从序列计算
- Charge（电荷）
- Hydrophobicity（疏水性）
- Boman Index（生物活性预测）
- Aliphatic Index（热稳定性）
"""

from typing import Dict
import numpy as np


# 氨基酸理化性质表
AA_PROPERTIES = {
    # 电荷 (pH 7.0)
    'charge': {
        'K': 1, 'R': 1, 'H': 0.5,  # 正电荷
        'D': -1, 'E': -1,           # 负电荷
    },
    
    # 疏水性 (Kyte-Doolittle scale)
    'hydrophobicity': {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
        'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
        'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
    },
    
    # Boman Index (蛋白质相互作用潜力)
    'boman': {
        'L': 4.92, 'I': 4.92, 'V': 4.04, 'F': 2.98, 'M': 2.35,
        'W': 2.33, 'A': 1.81, 'C': 1.28, 'G': 0.94, 'Y': 0.14,
        'T': -2.57, 'S': -3.40, 'H': -4.66, 'Q': -5.54, 'K': -5.55,
        'N': -6.64, 'E': -6.81, 'D': -8.72, 'R': -14.92, 'P': -1.50
    },
    
    # Aliphatic Index (脂肪族残基含量)
    'aliphatic': {
        'A': 1.0, 'V': 2.9, 'I': 2.9, 'L': 3.9
    }
}


def compute_charge(sequence: str) -> float:
    """
    计算序列的净电荷 (pH 7.0)
    
    正电荷: K, R, H (半个)
    负电荷: D, E
    """
    charge = 0.0
    for aa in sequence.upper():
        charge += AA_PROPERTIES['charge'].get(aa, 0)
    return charge


def compute_hydrophobicity(sequence: str) -> float:
    """
    计算序列的平均疏水性 (Kyte-Doolittle scale)
    
    正值 = 疏水
    负值 = 亲水
    """
    if not sequence:
        return 0.0
    
    total = 0.0
    count = 0
    for aa in sequence.upper():
        if aa in AA_PROPERTIES['hydrophobicity']:
            total += AA_PROPERTIES['hydrophobicity'][aa]
            count += 1
    
    return total / count if count > 0 else 0.0


def compute_boman_index(sequence: str) -> float:
    """
    计算 Boman Index (蛋白质相互作用潜力)
    
    高值 (>2.5) = 高生物活性潜力
    """
    if not sequence:
        return 0.0
    
    total = 0.0
    count = 0
    for aa in sequence.upper():
        if aa in AA_PROPERTIES['boman']:
            total += AA_PROPERTIES['boman'][aa]
            count += 1
    
    return total / count if count > 0 else 0.0


def compute_aliphatic_index(sequence: str) -> float:
    """
    计算 Aliphatic Index (热稳定性指标)
    
    公式: AI = X_A + 2.9*X_V + 3.9*(X_I + X_L)
    其中 X 是该氨基酸的摩尔百分比
    
    高值 = 高热稳定性
    """
    if not sequence:
        return 0.0
    
    seq_upper = sequence.upper()
    length = len(seq_upper)
    
    counts = {
        'A': seq_upper.count('A'),
        'V': seq_upper.count('V'),
        'I': seq_upper.count('I'),
        'L': seq_upper.count('L')
    }
    
    ai = (counts['A'] + 2.9 * counts['V'] + 3.9 * (counts['I'] + counts['L']))
    return (ai / length) * 100 if length > 0 else 0.0


def compute_all_properties(sequence: str) -> Dict[str, float]:
    """
    计算序列的所有理化性质
    
    Returns:
        dict with keys: charge, hydrophobicity, boman_index, aliphatic_index, length
    """
    return {
        'charge': compute_charge(sequence),
        'hydrophobicity': compute_hydrophobicity(sequence),
        'boman_index': compute_boman_index(sequence),
        'aliphatic_index': compute_aliphatic_index(sequence),
        'length': len(sequence)
    }


def predict_amp_activity(sequence: str) -> Dict[str, float]:
    """
    基于理化性质的简单活性预测（启发式）
    
    AMP 通常具有：
    - 正电荷 (charge > 2)
    - 适度疏水性 (-1 < hydro < 1)
    - 高 Boman Index (> 0)
    - 中等长度 (10-50 氨基酸)
    
    Returns:
        dict with score and individual components
    """
    props = compute_all_properties(sequence)
    
    # 各项评分 (0-1)
    scores = {}
    
    # 1. 电荷评分 (理想范围 +2 到 +10)
    charge = props['charge']
    if charge < 0:
        scores['charge_score'] = 0.0
    elif charge < 2:
        scores['charge_score'] = charge / 2
    elif charge <= 10:
        scores['charge_score'] = 1.0
    else:
        scores['charge_score'] = max(0, 1.0 - (charge - 10) / 10)
    
    # 2. 疏水性评分 (理想范围 -1 到 1)
    hydro = props['hydrophobicity']
    if -1 <= hydro <= 1:
        scores['hydro_score'] = 1.0
    elif hydro < -1:
        scores['hydro_score'] = max(0, 1.0 + (hydro + 1) / 2)
    else:
        scores['hydro_score'] = max(0, 1.0 - (hydro - 1) / 2)
    
    # 3. Boman Index 评分 (理想 > 0)
    boman = props['boman_index']
    scores['boman_score'] = min(1.0, max(0, (boman + 10) / 15))
    
    # 4. 长度评分 (理想 10-50)
    length = props['length']
    if 10 <= length <= 50:
        scores['length_score'] = 1.0
    elif length < 10:
        scores['length_score'] = length / 10
    else:
        scores['length_score'] = max(0, 1.0 - (length - 50) / 50)
    
    # 综合评分 (加权平均)
    weights = {
        'charge_score': 0.3,
        'hydro_score': 0.3,
        'boman_score': 0.2,
        'length_score': 0.2
    }
    
    overall_score = sum(scores[k] * weights[k] for k in weights)
    
    return {
        'activity_score': overall_score,
        **scores,
        **props
    }


if __name__ == "__main__":
    # 测试
    test_sequences = [
        "KLLKLLKKLLKLLK",  # 典型阳离子AMP
        "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin-2
        "RRWWRF",  # 短AMP
    ]
    
    for seq in test_sequences:
        print(f"\nSequence: {seq}")
        result = predict_amp_activity(seq)
        print(f"Activity Score: {result['activity_score']:.3f}")
        print(f"  Charge: {result['charge']:.1f} (score: {result['charge_score']:.3f})")
        print(f"  Hydro: {result['hydrophobicity']:.2f} (score: {result['hydro_score']:.3f})")
        print(f"  Boman: {result['boman_index']:.2f} (score: {result['boman_score']:.3f})")
        print(f"  Length: {result['length']} (score: {result['length_score']:.3f})")


