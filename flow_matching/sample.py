#!/usr/bin/env python3
"""
Flow Matching Sampling - 生成受控的潜空间向量

根据指定的 charge、hydrophobicity、length 条件，通过 ODE 求解生成新的 latent vectors
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from model import ConditionalFlowModel

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Net charge at pH 7 (simplified)
CHARGE = {
    "K": +1, "R": +1,  # positive
    "D": -1, "E": -1,  # negative
}


def compute_charge(seq: str) -> int:
    """计算 net charge at pH 7"""
    return sum(CHARGE.get(aa, 0) for aa in seq)


def compute_hydrophobicity(seq: str) -> float:
    """计算 Kyte-Doolittle 平均疏水性"""
    values = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    return np.mean(values) if values else 0.0


def sample_latents(
    model,
    conditions: torch.Tensor,
    num_samples: int,
    latent_dim: int = 64,
    num_steps: int = 100,
    device: str = "cpu"
):
    """
    从 Flow Matching 模型采样 latent vectors
    
    使用 Euler 方法求解 ODE: dz/dt = f_theta(z_t, t, c)
    从 z_0 ~ N(0, I) 开始，沿着速度场流向 z_1
    
    Args:
        model: 训练好的 ConditionalFlowMatching 模型
        conditions: (num_samples, cond_dim) 条件向量
        num_samples: 生成样本数
        latent_dim: latent space 维度
        num_steps: ODE 求解步数（越大越精确但越慢）
        device: 计算设备
    
    Returns:
        samples: (num_samples, latent_dim) 生成的 latent vectors
    """
    model.eval()
    
    # 从标准正态分布采样初始状态 z_0
    z = torch.randn(num_samples, latent_dim).to(device)
    conditions = conditions.to(device)
    
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for step in tqdm(range(num_steps), desc="ODE Solving"):
            t = torch.ones(num_samples, 1).to(device) * (step * dt)
            
            # 预测速度场 v_t = f_theta(z_t, t, c)
            v = model(z, t, conditions)
            
            # Euler 步: z_{t+dt} = z_t + v_t * dt
            z = z + v * dt
    
    return z.cpu().numpy()


def create_conditions(
    charge_range: tuple,
    hydro_range: tuple,
    length_range: tuple,
    num_samples: int,
    mode: str = "grid"
):
    """
    创建条件向量
    
    Args:
        charge_range: (min, max) for charge
        hydro_range: (min, max) for hydrophobicity
        length_range: (min, max) for length
        num_samples: 生成样本数
        mode: "grid" (网格采样) 或 "random" (随机采样)
    
    Returns:
        conditions: (num_samples, 3) numpy array
    """
    if mode == "grid":
        # 网格采样 - 均匀覆盖整个条件空间
        n_per_dim = int(np.ceil(num_samples ** (1/3)))
        
        charges = np.linspace(charge_range[0], charge_range[1], n_per_dim)
        hydros = np.linspace(hydro_range[0], hydro_range[1], n_per_dim)
        lengths = np.linspace(length_range[0], length_range[1], n_per_dim)
        
        # 创建网格
        charge_grid, hydro_grid, length_grid = np.meshgrid(charges, hydros, lengths, indexing='ij')
        
        conditions = np.stack([
            charge_grid.ravel(),
            hydro_grid.ravel(),
            length_grid.ravel()
        ], axis=1)
        
        # 截取到指定数量
        conditions = conditions[:num_samples]
        
    elif mode == "random":
        # 随机采样
        conditions = np.random.uniform(
            low=[charge_range[0], hydro_range[0], length_range[0]],
            high=[charge_range[1], hydro_range[1], length_range[1]],
            size=(num_samples, 3)
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return conditions


def load_data_stats(metadata_path: Path):
    """加载训练数据的统计信息，用于生成合理的条件"""
    meta = pd.read_csv(metadata_path)
    
    # Ensure required columns exist (compute if missing)
    if "charge" not in meta.columns:
        if "sequence" not in meta.columns:
            raise ValueError(f"metadata must have either 'charge' column or 'sequence' column to compute charge")
        print("⚠️  Computing missing 'charge' column from sequences...")
        meta["charge"] = meta["sequence"].apply(compute_charge)
    
    if "hydrophobicity" not in meta.columns:
        if "sequence" not in meta.columns:
            raise ValueError(f"metadata must have either 'hydrophobicity' column or 'sequence' column to compute hydrophobicity")
        print("⚠️  Computing missing 'hydrophobicity' column from sequences...")
        meta["hydrophobicity"] = meta["sequence"].apply(compute_hydrophobicity)
    
    if "length" not in meta.columns:
        if "sequence" not in meta.columns:
            raise ValueError(f"metadata must have either 'length' column or 'sequence' column to compute length")
        print("⚠️  Computing missing 'length' column from sequences...")
        meta["length"] = meta["sequence"].apply(len)
    
    stats = {
        "charge": {
            "mean": meta["charge"].mean(),
            "std": meta["charge"].std(),
            "min": meta["charge"].min(),
            "max": meta["charge"].max()
        },
        "hydrophobicity": {
            "mean": meta["hydrophobicity"].mean(),
            "std": meta["hydrophobicity"].std(),
            "min": meta["hydrophobicity"].min(),
            "max": meta["hydrophobicity"].max()
        },
        "length": {
            "mean": meta["length"].mean(),
            "std": meta["length"].std(),
            "min": meta["length"].min(),
            "max": meta["length"].max()
        }
    }
    
    return stats


def main(args):
    # 设备
    if torch.backends.mps.is_available() and not args.cpu:
        device = "mps"
    elif torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # 加载数据统计
    print(f"Loading data stats from {args.metadata_path}...")
    stats = load_data_stats(args.metadata_path)
    
    print("\n=== Training Data Statistics ===")
    for key, val in stats.items():
        print(f"{key:15s}: mean={val['mean']:.3f}, std={val['std']:.3f}, range=[{val['min']:.3f}, {val['max']:.3f}]")
    
    # 创建条件
    print(f"\n=== Generating Conditions ({args.mode} mode) ===")
    
    if args.use_data_range:
        # 使用训练数据的实际范围
        charge_range = (stats["charge"]["min"], stats["charge"]["max"])
        hydro_range = (stats["hydrophobicity"]["min"], stats["hydrophobicity"]["max"])
        length_range = (stats["length"]["min"], stats["length"]["max"])
    else:
        # 使用用户指定的范围
        charge_range = args.charge_range
        hydro_range = args.hydro_range
        length_range = args.length_range
    
    print(f"Charge range: {charge_range}")
    print(f"Hydrophobicity range: {hydro_range}")
    print(f"Length range: {length_range}")
    
    conditions = create_conditions(
        charge_range=charge_range,
        hydro_range=hydro_range,
        length_range=length_range,
        num_samples=args.num_samples,
        mode=args.mode
    )
    
    print(f"Generated {len(conditions)} condition vectors")
    
    # 加载模型
    print(f"\nLoading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # 从 checkpoint 读取模型配置（如果存在），否则使用命令行参数
    latent_dim = ckpt.get("latent_dim", args.latent_dim)
    condition_dim = ckpt.get("condition_dim", 3)
    hidden_dims = ckpt.get("hidden_dims", args.hidden_dims)
    
    print(f"Model config: latent_dim={latent_dim}, condition_dim={condition_dim}, hidden_dims={hidden_dims}")
    
    model = ConditionalFlowModel(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dims=hidden_dims
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    
    # 采样
    print(f"\nSampling {args.num_samples} latent vectors...")
    print(f"ODE steps: {args.ode_steps}")
    
    conditions_tensor = torch.from_numpy(conditions).float()
    samples = sample_latents(
        model=model,
        conditions=conditions_tensor,
        num_samples=args.num_samples,
        latent_dim=args.latent_dim,
        num_steps=args.ode_steps,
        device=device
    )
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 latent vectors
    latents_path = output_dir / "sampled_latents.npy"
    np.save(latents_path, samples)
    print(f"\n✅ Saved {len(samples)} latent vectors to {latents_path}")
    
    # 保存对应的条件
    conditions_df = pd.DataFrame(
        conditions,
        columns=["charge", "hydrophobicity", "length"]
    )
    conditions_path = output_dir / "sampled_conditions.csv"
    conditions_df.to_csv(conditions_path, index=False)
    print(f"✅ Saved conditions to {conditions_path}")
    
    # 统计信息
    print(f"\n=== Sampled Latents Statistics ===")
    print(f"Shape: {samples.shape}")
    print(f"Mean: {samples.mean():.4f}")
    print(f"Std: {samples.std():.4f}")
    print(f"Min: {samples.min():.4f}")
    print(f"Max: {samples.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from Flow Matching model")
    
    # 模型参数
    parser.add_argument("--checkpoint", type=str, default="outputs/flow/checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--latent-dim", type=int, default=64,
                        help="Latent space dimensionality")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer dimensions (will be overridden by checkpoint if available)")
    
    # 采样参数
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--ode-steps", type=int, default=100,
                        help="Number of ODE integration steps")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "random"],
                        help="Condition sampling mode: grid or random")
    
    # 条件范围
    parser.add_argument("--use-data-range", action="store_true",
                        help="Use training data range for conditions")
    parser.add_argument("--charge-range", type=float, nargs=2, default=[-5.0, 15.0],
                        help="Charge range (min max)")
    parser.add_argument("--hydro-range", type=float, nargs=2, default=[-1.0, 1.0],
                        help="Hydrophobicity range (min max)")
    parser.add_argument("--length-range", type=float, nargs=2, default=[10.0, 100.0],
                        help="Length range (min max)")
    
    # 数据路径
    parser.add_argument("--metadata-path", type=str, 
                        default="data/embeddings/metadata_with_labels.csv",
                        help="Path to metadata CSV (for stats)")
    
    # 输出
    parser.add_argument("--output-dir", type=str, default="outputs/samples",
                        help="Output directory")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    
    args = parser.parse_args()
    main(args)

