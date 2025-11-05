"""
RAE Dataset - Phase 2

高效加载 ESM-3 embeddings
使用 mmap 避免内存爆炸
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """从 .npz 加载 embeddings"""

    def __init__(self, npz_path, indices=None):
        """
        Args:
            npz_path: path to esm3_embeddings.npz
            indices: 如果提供，只加载指定索引 (用于 train/val/test split)
        """
        # 使用 mmap_mode 避免一次性加载全部数据到内存
        data = np.load(npz_path, mmap_mode='r')
        self.embeddings = data['embeddings']  # (N, 1536)
        
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.embeddings))
        
        print(f"Loaded {len(self.indices)} embeddings from {npz_path}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """返回单个 embedding"""
        real_idx = self.indices[idx]
        emb = self.embeddings[real_idx].astype(np.float32)
        return torch.from_numpy(emb)


def get_dataloaders(npz_path, splits_dir, batch_size=256, num_workers=4):
    """
    创建 train/val/test dataloaders
    
    Args:
        npz_path: path to esm3_embeddings.npz
        splits_dir: path to data/splits/
        batch_size: batch size
        num_workers: num workers for DataLoader
    """
    from pathlib import Path
    
    splits_dir = Path(splits_dir)
    
    # 加载索引
    train_indices = np.load(splits_dir / "train_indices.npy")
    val_indices = np.load(splits_dir / "val_indices.npy")
    test_indices = np.load(splits_dir / "test_indices.npy")
    
    # 创建 datasets
    train_dataset = EmbeddingDataset(npz_path, train_indices)
    val_dataset = EmbeddingDataset(npz_path, val_indices)
    test_dataset = EmbeddingDataset(npz_path, test_indices)
    
    # 创建 dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """测试 dataset"""
    from pathlib import Path
    
    npz_path = "data/embeddings/esm3_embeddings.npz"
    splits_dir = "data/splits"
    
    train_loader, val_loader, test_loader = get_dataloaders(
        npz_path, splits_dir, batch_size=32, num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 测试一个 batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}")


if __name__ == "__main__":
    test_dataset()



