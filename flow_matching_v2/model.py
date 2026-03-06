"""
Improved Conditional Flow Matching Model - V2

Key improvements:
1. Larger model architecture (more hidden dims, deeper network)
2. Time embedding (sinusoidal positional encoding)
3. Better condition encoding (FiLM - Feature-wise Linear Modulation)
4. Residual connections
5. Layer normalization
"""

import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) or (B, 1) time in [0, 1]
        Returns:
            (B, dim) time embeddings
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation"""

    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim

        # Simple MLP to generate scale and shift from condition
        self.layers = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
        )

    def forward(self, x, c):
        """
        Args:
            x: (B, feature_dim) feature tensor
            c: (B, condition_dim) condition
        Returns:
            (B, feature_dim) modulated features
        """
        gamma_beta = self.layers(c)  # (B, feature_dim * 2)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, feature_dim)

        return x * (1 + gamma) + beta


class ResidualBlock(nn.Module):
    """Residual block with FiLM conditioning"""

    def __init__(self, dim, condition_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.film = FiLMBlock(dim, condition_dim)

    def forward(self, x, c):
        residual = x
        out = self.net(x)
        out = self.film(out, c)
        return out + residual


class ImprovedFlowModel(nn.Module):
    """
    Improved Conditional Flow Matching Model

    Architecture:
    - Time embedding (sinusoidal)
    - Condition embedding (FiLM)
    - Deep residual network
    """

    def __init__(
        self,
        latent_dim=64,
        condition_dim=3,
        hidden_dims=[512, 512, 512, 256],
        time_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_dim = time_dim

        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)

        # Input projection: latent + time + condition -> hidden
        input_dim = latent_dim + time_dim + condition_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
        )

        # Condition projection for FiLM
        self.condition_proj = nn.Linear(condition_dim, hidden_dims[0])

        # Residual blocks
        layers = []
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else hidden_dims[i]

            if in_dim == out_dim:
                layers.append(ResidualBlock(in_dim, condition_dim, dropout))
            else:
                layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                    )
                )
                # Add FiLM after dimension change
                layers.append(FiLMBlock(out_dim, condition_dim))

        self.layers = nn.ModuleList(layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
        )

    def forward(self, z_t, t, c):
        """
        Args:
            z_t: (B, latent_dim) latent at time t
            t: (B,) or (B, 1) time in [0, 1]
            c: (B, condition_dim) conditions

        Returns:
            v_t: (B, latent_dim) predicted velocity
        """
        # Time embedding
        t_emb = self.time_embedding(t)  # (B, time_dim)

        # Project condition
        c_proj = self.condition_proj(c)  # (B, hidden_dims[0])

        # Concatenate inputs
        x = torch.cat([z_t, t_emb, c], dim=-1)  # (B, latent_dim + time_dim + cond_dim)

        # Input projection
        x = self.input_proj(x)
        x = x + c_proj  # residual-like connection

        # Pass through layers
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, c)
            elif isinstance(layer, FiLMBlock):
                x = layer(x, c)
            else:
                x = layer(x)

        # Output
        v_t = self.output_proj(x)

        return v_t

    def compute_loss(self, z_1, c, drop_condition_prob=0.1):
        """
        Flow Matching Loss with optional classifier-free guidance training

        Args:
            z_1: (B, latent_dim) data latents
            c: (B, condition_dim) conditions
            drop_condition_prob: probability to drop condition during training

        Returns:
            loss: scalar
        """
        batch_size = z_1.shape[0]
        device = z_1.device

        # Randomly drop condition (classifier-free guidance)
        mask = torch.rand(batch_size, device=device) < drop_condition_prob
        c_train = c.clone()
        c_train[mask] = 0  # Set to null condition

        # Sample noise
        z_0 = torch.randn_like(z_1)

        # Sample time
        t = torch.rand(batch_size, 1, device=device)

        # Interpolate
        z_t = t * z_1 + (1 - t) * z_0

        # Compute target velocity
        v_target = z_1 - z_0

        # Predict velocity
        v_pred = self.forward(z_t, t, c_train)

        # MSE loss
        loss = nn.functional.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample(self, c, n_steps=100, guidance_scale=1.0, device="cpu"):
        """
        Sample from p(z | c) using ODE solver with classifier-free guidance

        Args:
            c: (B, condition_dim) conditions
            n_steps: number of ODE steps
            guidance_scale: classifier-free guidance scale (1.0 = no guidance)
            device: device

        Returns:
            z_1: (B, latent_dim) generated latents
        """
        batch_size = c.shape[0]

        # Start from noise
        z_t = torch.randn(batch_size, self.latent_dim, device=device)

        # Create null condition
        c_null = torch.zeros_like(c)

        # ODE solver (Euler method)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)

            # Unconditional prediction
            v_uncond = self.forward(z_t, t, c_null)

            # Conditional prediction
            v_cond = self.forward(z_t, t, c)

            # Apply classifier-free guidance
            if guidance_scale != 1.0:
                v_t = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_t = v_cond

            z_t = z_t + dt * v_t

        return z_t


def test_model():
    """Test model"""
    model = ImprovedFlowModel(
        latent_dim=64,
        condition_dim=3,
        hidden_dims=[512, 512, 512, 256],
    )

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
    print(f"  Loss: {loss.item():.4f}")

    # Test sampling
    z_sampled = model.sample(c, n_steps=10)
    print(f"\nSampling test:")
    print(f"  Input: c={c.shape}")
    print(f"  Output: z_sampled={z_sampled.shape}")

    print(f"\n✅ Model test passed!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_model()
