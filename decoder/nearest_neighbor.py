#!/usr/bin/env python3
"""
最近邻搜索 - 从 embedding 找到最相似的训练序列

简单、直接、有效 - Linus 风格
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
import torch


class NearestNeighborDecoder:
    """
    基于最近邻的序列解码器
    
    原理：
    1. 加载训练集的 embeddings 和对应序列
    2. 对于新的 embedding，找 k 个最近邻
    3. 返回最近邻的序列（可选：返回多个候选）
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        n_neighbors: int = 5,
        metric: str = "cosine"
    ):
        """
        Args:
            embeddings: (N, 1536) ESM-3 embeddings of training set
            metadata: DataFrame with 'sequence' column
            n_neighbors: number of neighbors to return
            metric: distance metric ('cosine', 'euclidean', 'manhattan')
        """
        self.embeddings = embeddings
        self.metadata = metadata
        self.n_neighbors = n_neighbors
        self.metric = metric
        
        # 构建索引
        print(f"Building k-NN index ({metric})...")
        self.knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm='auto',  # 自动选择最优算法
            n_jobs=-1  # 使用所有CPU核心
        )
        self.knn.fit(embeddings)
        print(f"✅ Index built with {len(embeddings)} embeddings")
    
    def decode(
        self,
        query_embeddings: np.ndarray,
        return_distances: bool = True,
        return_all_neighbors: bool = False
    ) -> List[dict]:
        """
        解码：从 query embeddings 找最近邻序列
        
        Args:
            query_embeddings: (M, 1536) 查询 embeddings
            return_distances: 是否返回距离
            return_all_neighbors: 是否返回所有 k 个近邻（否则只返回最近的）
        
        Returns:
            List of dicts, 每个包含:
                - sequence: 序列
                - distance: 距离（如果 return_distances=True）
                - neighbors: 所有近邻（如果 return_all_neighbors=True）
        """
        # 搜索最近邻
        distances, indices = self.knn.kneighbors(query_embeddings)
        
        results = []
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            result = {}
            
            if return_all_neighbors:
                # 返回所有 k 个近邻
                neighbors = []
                for dist, idx in zip(dists, idxs):
                    row = self.metadata.iloc[idx]
                    neighbors.append({
                        'sequence': row['sequence'],
                        'distance': float(dist),
                        'idx': int(idx),
                        'id': row.get('id', ''),
                        'source': row.get('source', ''),
                        'length': int(row.get('length', len(row['sequence'])))
                    })
                result['neighbors'] = neighbors
                result['sequence'] = neighbors[0]['sequence']  # 最近的
                result['distance'] = neighbors[0]['distance']
            else:
                # 只返回最近的
                idx = idxs[0]
                dist = dists[0]
                row = self.metadata.iloc[idx]
                
                result['sequence'] = row['sequence']
                result['idx'] = int(idx)
                result['id'] = row.get('id', '')
                result['source'] = row.get('source', '')
                result['length'] = int(row.get('length', len(row['sequence'])))
                
                if return_distances:
                    result['distance'] = float(dist)
            
            results.append(result)
        
        return results
    
    def compute_novelty(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        计算新颖性：query 到训练集最近邻的距离
        
        距离越大 = 越新颖
        """
        distances, _ = self.knn.kneighbors(query_embeddings)
        return distances[:, 0]  # 返回最近邻的距离


def load_decoder(
    embeddings_path: str,
    metadata_path: str,
    splits_path: str = None,
    split: str = "train",
    n_neighbors: int = 5,
    metric: str = "cosine"
) -> NearestNeighborDecoder:
    """
    便捷函数：从文件加载 decoder
    
    Args:
        embeddings_path: .npz 文件路径
        metadata_path: metadata.csv 路径
        splits_path: splits 目录路径（可选，如果只用训练集）
        split: 使用哪个 split ('train', 'val', 'test', 'all')
        n_neighbors: k-NN 的 k
        metric: 距离度量
    
    Returns:
        NearestNeighborDecoder 实例
    """
    print(f"Loading decoder data from {split} split...")
    
    # 加载 metadata
    metadata = pd.read_csv(metadata_path)
    
    # 加载 embeddings
    emb_data = np.load(embeddings_path)
    embeddings = emb_data['embeddings']
    
    # 如果指定了 split，筛选数据
    if split != "all" and splits_path:
        split_file = Path(splits_path) / f"{split}_indices.npy"
        indices = np.load(split_file)
        
        embeddings = embeddings[indices]
        metadata = metadata.iloc[indices].reset_index(drop=True)
        
        print(f"Using {split} split: {len(embeddings)} sequences")
    else:
        print(f"Using all data: {len(embeddings)} sequences")
    
    return NearestNeighborDecoder(
        embeddings=embeddings,
        metadata=metadata,
        n_neighbors=n_neighbors,
        metric=metric
    )


if __name__ == "__main__":
    # 测试
    decoder = load_decoder(
        embeddings_path="data/embeddings/esm3_embeddings_normalized.npz",
        metadata_path="data/embeddings/metadata.csv",
        splits_path="data/splits",
        split="train"
    )
    
    # 随机测试
    test_emb = decoder.embeddings[0:1]  # 取第一个
    results = decoder.decode(test_emb, return_all_neighbors=True)
    
    print("\n=== Test Result ===")
    print(f"Sequence: {results[0]['sequence'][:50]}...")
    print(f"Distance: {results[0]['distance']:.4f}")
    print(f"Top-5 neighbors distances: {[n['distance'] for n in results[0]['neighbors'][:5]]}")


