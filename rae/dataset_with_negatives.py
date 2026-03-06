"""
RAE Dataset with Negative Samples - Phase 4

支持负样本 oversample 的训练数据集
"""

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
from pathlib import Path


class EmbeddingDataset(Dataset):
    """从 .npz 加载 embeddings"""

    def __init__(self, npz_path, indices=None, label=1):
        """
        Args:
            npz_path: path to esm3_embeddings.npz
            indices: 如果提供，只加载指定索引 (用于 train/val/test split)
            label: 标签 (1=正样本, 0=负样本)
        """
        # 使用 mmap_mode 避免一次性加载全部数据到内存
        data = np.load(npz_path, mmap_mode='r')
        self.embeddings = data['embeddings']  # (N, 1536)
        
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.embeddings))
        
        self.label = label
        print(f"Loaded {len(self.indices)} embeddings from {npz_path} (label={label})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """返回单个 embedding 和标签"""
        real_idx = self.indices[idx]
        emb = self.embeddings[real_idx].astype(np.float32)
        return torch.from_numpy(emb), self.label


class MixedDataset(Dataset):
    """混合正负样本的数据集"""
    
    def __init__(self, positive_dataset, negative_datasets, negative_weights=None):
        """
        Args:
            positive_dataset: 正样本数据集
            negative_datasets: 负样本数据集列表
            negative_weights: 每个负样本数据集的权重（用于 oversample）
        """
        self.positive_dataset = positive_dataset
        self.negative_datasets = negative_datasets
        
        if negative_weights is None:
            negative_weights = [1.0] * len(negative_datasets)
        self.negative_weights = negative_weights
        
        # 计算每个数据集的样本数
        self.positive_size = len(positive_dataset)
        self.negative_sizes = [len(ds) for ds in negative_datasets]
        
        # 计算加权后的总大小
        self.negative_weighted_size = sum(
            int(size * weight) for size, weight in zip(self.negative_sizes, self.negative_weights)
        )
        self.total_size = self.positive_size + self.negative_weighted_size
        
        print(f"MixedDataset: {self.positive_size} positive, {self.negative_weighted_size} weighted negative")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """根据索引返回样本"""
        if idx < self.positive_size:
            # 正样本
            return self.positive_dataset[idx]
        else:
            # 负样本（考虑权重）
            idx -= self.positive_size
            cumulative = 0
            for ds, size, weight in zip(self.negative_datasets, self.negative_sizes, self.negative_weights):
                weighted_size = int(size * weight)
                if idx < cumulative + weighted_size:
                    # 在这个数据集中
                    local_idx = (idx - cumulative) % size  # 循环采样
                    return ds[local_idx]
                cumulative += weighted_size
            
            # 如果超出范围，返回最后一个负样本
            return self.negative_datasets[-1][0]


def get_dataloaders_with_negatives(
    positive_npz_path,
    positive_splits_dir,
    negative_npz_paths=None,
    negative_weights=None,
    batch_size=256,
    num_workers=4,
    use_weighted_sampler=False
):
    """
    创建包含负样本的 train/val/test dataloaders
    
    Args:
        positive_npz_path: 正样本 embeddings .npz 路径
        positive_splits_dir: 正样本 splits 目录
        negative_npz_paths: 负样本 embeddings .npz 路径列表
        negative_weights: 负样本权重列表（用于 oversample）
        batch_size: batch size
        num_workers: num workers
        use_weighted_sampler: 是否使用加权采样器（平衡正负样本）
    """
    from pathlib import Path
    
    positive_splits_dir = Path(positive_splits_dir)
    
    # 加载正样本索引
    train_indices = np.load(positive_splits_dir / "train_indices.npy")
    val_indices = np.load(positive_splits_dir / "val_indices.npy")
    test_indices = np.load(positive_splits_dir / "test_indices.npy")
    
    # 创建正样本数据集
    train_positive = EmbeddingDataset(positive_npz_path, train_indices, label=1)
    val_positive = EmbeddingDataset(positive_npz_path, val_indices, label=1)
    test_positive = EmbeddingDataset(positive_npz_path, test_indices, label=1)
    
    # 创建负样本数据集（仅用于训练）
    train_datasets = [train_positive]
    if negative_npz_paths:
        negative_datasets = []
        for npz_path in negative_npz_paths:
            # 负样本不需要 split，全部用于训练
            negative_datasets.append(EmbeddingDataset(npz_path, None, label=0))
        
        # 创建混合数据集
        train_mixed = MixedDataset(train_positive, negative_datasets, negative_weights)
        train_datasets = [train_mixed]
    
    # 创建 dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_positive,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_positive,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """测试 dataset"""
    # 创建虚拟数据
    positive_npz = "data/embeddings/esm3_embeddings.npz"
    splits_dir = "data/splits"
    
    # 测试不带负样本
    train_loader, val_loader, test_loader = get_dataloaders_with_negatives(
        positive_npz, splits_dir, batch_size=32, num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 测试一个 batch
    batch = next(iter(train_loader))
    if isinstance(batch, tuple):
        emb, label = batch
        print(f"Batch shape: {emb.shape}, label shape: {label.shape}")
    else:
        print(f"Batch shape: {batch.shape}")


if __name__ == "__main__":
    test_dataset()


