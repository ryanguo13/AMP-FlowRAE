"""
Conditional Flow Matching Model - Phase 3

简洁 MLP 实现，Linus 风格
"""

import torch
import torch.nn as nn


class ConditionalFlowModel(nn.Module):
    """
    Conditional Flow Matching
    
    Input:
        z_t: (B, latent_dim) latent at time t
        t:   (B, 1) time in [0, 1]
        c:   (B, condition_dim) conditions
    
    Output:
        v_t: (B, latent_dim) velocity field
    """
    
    def __init__(self, latent_dim=64, condition_dim=3, hidden_dims=[256, 256]):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Build MLP: [latent + time + condition] → [hidden] → [latent]
        input_dim = latent_dim + 1 + condition_dim  # 64 + 1 + 3 = 68
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z_t, t, c):
        """
        Args:
            z_t: (B, latent_dim) latent at time t
            t: (B, 1) or (B,) time
            c: (B, condition_dim) conditions
        
        Returns:
            v_t: (B, latent_dim) predicted velocity
        """
        # Ensure t is (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Concatenate inputs
        x = torch.cat([z_t, t, c], dim=-1)  # (B, 68)
        
        # Predict velocity
        v_t = self.net(x)  # (B, 64)
        
        return v_t
    
    def compute_loss(self, z_1, c):
        """
        Flow Matching Loss
        
        Args:
            z_1: (B, latent_dim) data latents
            c: (B, condition_dim) conditions
        
        Returns:
            loss: scalar
        """
        batch_size = z_1.shape[0]
        device = z_1.device
        
        # 1. Sample noise
        z_0 = torch.randn_like(z_1)  # (B, 64)
        
        # 2. Sample time
        t = torch.rand(batch_size, 1, device=device)  # (B, 1) in [0, 1]
        
        # 3. Interpolate
        z_t = t * z_1 + (1 - t) * z_0
        
        # 4. Compute target velocity
        v_target = z_1 - z_0
        
        # 5. Predict velocity
        v_pred = self.forward(z_t, t, c)
        
        # 6. MSE loss
        loss = nn.functional.mse_loss(v_pred, v_target)
        
        return loss
    
    @torch.no_grad()
    def sample(self, c, n_steps=5, device="cpu"):
        """
        Sample from p(z | c) using ODE solver
        
        Args:
            c: (B, condition_dim) conditions
            n_steps: number of ODE steps
            device: device
        
        Returns:
            z_1: (B, latent_dim) generated latents
        """
        batch_size = c.shape[0]
        
        # Start from noise
        z_t = torch.randn(batch_size, self.latent_dim, device=device)
        
        # ODE solver (Euler method)
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)
            v_t = self.forward(z_t, t, c)
            z_t = z_t + dt * v_t
        
        return z_t


def test_model():
    """Test model"""
    model = ConditionalFlowModel(latent_dim=64, condition_dim=3)
    
    # Test forward
    z_t = torch.randn(16, 64)
    t = torch.rand(16, 1)
    c = torch.rand(16, 3)
    
    v_t = model(z_t, t, c)
    print(f"Forward test:")
    print(f"  Input: z_t={z_t.shape}, t={t.shape}, c={c.shape}")
    print(f"  Output: v_t={v_t.shape}")
    
    # Test loss
    z_1 = torch.randn(16, 64)
    loss = model.compute_loss(z_1, c)
    print(f"\nLoss test:")
    print(f"  Input: z_1={z_1.shape}, c={c.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test sampling
    z_sampled = model.sample(c, n_steps=5)
    print(f"\nSampling test:")
    print(f"  Input: c={c.shape}")
    print(f"  Output: z_sampled={z_sampled.shape}")
    
    print(f"\n✅ Model test passed!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_model()

