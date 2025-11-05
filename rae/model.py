"""
RAE (Representation Autoencoder) - Phase 2

学习 ESM-3 embedding 的低维连续潜空间
目标：embedding (1536) → latent (64) → embedding (1536)

设计原则 (Linus 风格)：
1. 简单 MLP，不搞复杂的 ResBlock
2. 正交正则化，确保 latent 维度独立
3. L2 正则化，防止 latent 爆炸
"""

import torch
import torch.nn as nn


class RAEEncoder(nn.Module):
    """Encoder: 1536 → 64"""

    def __init__(self, input_dim=1536, latent_dim=64, hidden_dims=[512, 256]):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        
        # 最后一层：不加激活，让 latent 可以自由分布
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class RAEDecoder(nn.Module):
    """Decoder: 64 → 1536"""

    def __init__(self, latent_dim=64, output_dim=1536, hidden_dims=[256, 512]):
        super().__init__()
        
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        
        # 最后一层：重构 embedding
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class RAE(nn.Module):
    """完整的 RAE 模型"""

    def __init__(self, input_dim=1536, latent_dim=64, hidden_dims=[512, 256]):
        super().__init__()
        self.encoder = RAEEncoder(input_dim, latent_dim, hidden_dims)
        self.decoder = RAEDecoder(latent_dim, input_dim, hidden_dims[::-1])
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_loss(self, x, lambda_z=0.01, lambda_orth=0.1):
        """
        Loss = MSE(x, x_recon) + λ_z * ||z||^2 + λ_orth * ||W W^T - I||^2
        
        Args:
            x: (B, input_dim)
            lambda_z: latent L2 正则化系数
            lambda_orth: 正交正则化系数 (鼓励 latent 维度独立)
        """
        x_recon, z = self.forward(x)
        
        # 重构损失
        mse_loss = nn.functional.mse_loss(x_recon, x)
        
        # Latent L2 正则化
        z_reg = (z ** 2).mean()
        
        # 正交正则化 (鼓励 encoder 最后一层权重正交)
        # W: (latent_dim, hidden_dim)
        W = self.encoder.encoder[-1].weight  # (latent_dim, hidden_dim)
        WWT = W @ W.T  # (latent_dim, latent_dim)
        I = torch.eye(self.latent_dim, device=W.device)
        orth_loss = ((WWT - I) ** 2).mean()
        
        total_loss = mse_loss + lambda_z * z_reg + lambda_orth * orth_loss
        
        return {
            "loss": total_loss,
            "mse": mse_loss.item(),
            "z_reg": z_reg.item(),
            "orth_loss": orth_loss.item(),
        }


def test_model():
    """简单测试"""
    model = RAE(input_dim=1536, latent_dim=64)
    x = torch.randn(16, 1536)
    
    # Forward
    x_recon, z = model(x)
    print(f"Input: {x.shape}")
    print(f"Latent: {z.shape}")
    print(f"Recon: {x_recon.shape}")
    
    # Loss
    loss_dict = model.compute_loss(x)
    print(f"Loss: {loss_dict}")


if __name__ == "__main__":
    test_model()



