#!/usr/bin/env python3
"""
Improved Sampling Script - V2

Features:
- Classifier-free guidance during sampling
- More ODE steps for better quality
- Better condition generation
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from flow_matching_v2.model import ImprovedFlowModel


HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

CHARGE = {
    "K": +1,
    "R": +1,
    "D": -1,
    "E": -1,
}


def compute_charge(seq: str) -> int:
    return sum(CHARGE.get(aa, 0) for aa in seq)


def compute_hydrophobicity(seq: str) -> float:
    values = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    return np.mean(values) if values else 0.0


def load_data_stats(metadata_path):
    meta = pd.read_csv(metadata_path)

    if "charge" not in meta.columns:
        meta["charge"] = meta["sequence"].apply(compute_charge)
    if "hydrophobicity" not in meta.columns:
        meta["hydrophobicity"] = meta["sequence"].apply(compute_hydrophobicity)
    if "length" not in meta.columns:
        meta["length"] = meta["sequence"].apply(len)

    stats = {
        "charge": {"min": meta["charge"].min(), "max": meta["charge"].max()},
        "hydrophobicity": {
            "min": meta["hydrophobicity"].min(),
            "max": meta["hydrophobicity"].max(),
        },
        "length": {"min": meta["length"].min(), "max": meta["length"].max()},
    }
    return stats


def create_conditions(charge_range, hydro_range, length_range, num_samples, mode="random"):
    """Create condition vectors"""
    if mode == "random":
        conditions = np.random.uniform(
            low=[charge_range[0], hydro_range[0], length_range[0]],
            high=[charge_range[1], hydro_range[1], length_range[1]],
            size=(num_samples, 3),
        )
    elif mode == "grid":
        n_per_dim = int(np.ceil(num_samples ** (1 / 3)))
        charges = np.linspace(charge_range[0], charge_range[1], n_per_dim)
        hydros = np.linspace(hydro_range[0], hydro_range[1], n_per_dim)
        lengths = np.linspace(length_range[0], length_range[1], n_per_dim)

        charge_grid, hydro_grid, length_grid = np.meshgrid(charges, hydros, lengths, indexing="ij")
        conditions = np.stack(
            [charge_grid.ravel(), hydro_grid.ravel(), length_grid.ravel()], axis=1
        )[:num_samples]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return conditions


def normalize_conditions(conditions, stats):
    """Normalize conditions to [0, 1]"""
    charge_min, charge_max = stats["charge"]["min"], stats["charge"]["max"]
    hydro_min, hydro_max = stats["hydrophobicity"]["min"], stats["hydrophobicity"]["max"]
    length_min, length_max = stats["length"]["min"], stats["length"]["max"]

    normalized = np.zeros_like(conditions)
    normalized[:, 0] = (conditions[:, 0] - charge_min) / (charge_max - charge_min + 1e-8)
    normalized[:, 1] = (conditions[:, 1] - hydro_min) / (hydro_max - hydro_min + 1e-8)
    normalized[:, 2] = (conditions[:, 2] - length_min) / (length_max - length_min + 1e-8)

    return normalized


def main(args):
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load data stats
    print(f"Loading data stats from {args.metadata_path}...")
    stats = load_data_stats(args.metadata_path)

    print("\n=== Data Statistics ===")
    print(f"Charge: [{stats['charge']['min']}, {stats['charge']['max']}]")
    print(
        f"Hydrophobicity: [{stats['hydrophobicity']['min']:.2f}, {stats['hydrophobicity']['max']:.2f}]"
    )
    print(f"Length: [{stats['length']['min']}, {stats['length']['max']}]")

    # Create conditions
    if args.use_data_range:
        charge_range = (stats["charge"]["min"], stats["charge"]["max"])
        hydro_range = (stats["hydrophobicity"]["min"], stats["hydrophobicity"]["max"])
        length_range = (stats["length"]["min"], stats["length"]["max"])
    else:
        charge_range = args.charge_range
        hydro_range = args.hydro_range
        length_range = args.length_range

    print(f"\n=== Generating Conditions ===")
    print(f"Charge range: {charge_range}")
    print(f"Hydrophobicity range: {hydro_range}")
    print(f"Length range: {length_range}")

    conditions = create_conditions(
        charge_range, hydro_range, length_range, args.num_samples, args.mode
    )
    print(f"Generated {len(conditions)} conditions")

    # Normalize conditions
    conditions_norm = normalize_conditions(conditions, stats)
    conditions_tensor = torch.from_numpy(conditions_norm).float()

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)

    config = ckpt.get(
        "config",
        {
            "latent_dim": args.latent_dim,
            "condition_dim": args.condition_dim,
            "hidden_dims": args.hidden_dims,
            "time_dim": args.time_dim,
            "dropout": args.dropout,
        },
    )

    model = ImprovedFlowModel(
        latent_dim=config.get("latent_dim", args.latent_dim),
        condition_dim=config.get("condition_dim", args.condition_dim),
        hidden_dims=config.get("hidden_dims", args.hidden_dims),
        time_dim=config.get("time_dim", args.time_dim),
        dropout=config.get("dropout", args.dropout),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from epoch {ckpt.get('epoch', 'unknown')}")

    # Sample
    print(f"\nSampling {args.num_samples} latent vectors...")
    print(f"ODE steps: {args.ode_steps}")
    print(f"Guidance scale: {args.guidance_scale}")

    all_samples = []
    batch_size = 64

    for i in tqdm(range(0, args.num_samples, batch_size), desc="Sampling"):
        batch_c = conditions_tensor[i : i + batch_size].to(device)

        with torch.no_grad():
            samples = model.sample(
                batch_c, n_steps=args.ode_steps, guidance_scale=args.guidance_scale, device=device
            )

        all_samples.append(samples.cpu().numpy())

    samples = np.vstack(all_samples)
    print(f"Generated {len(samples)} latent vectors")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "sampled_latents.npy", samples)
    print(f"\n✅ Saved latents to {output_dir / 'sampled_latents.npy'}")

    # Save conditions
    conditions_df = pd.DataFrame(conditions, columns=["charge", "hydrophobicity", "length"])
    conditions_df.to_csv(output_dir / "sampled_conditions.csv", index=False)
    print(f"✅ Saved conditions to {output_dir / 'sampled_conditions.csv'}")

    # Stats
    print(f"\n=== Latent Statistics ===")
    print(f"Shape: {samples.shape}")
    print(f"Mean: {samples.mean():.4f}")
    print(f"Std: {samples.std():.4f}")
    print(f"Min: {samples.min():.4f}")
    print(f"Max: {samples.max():.4f}")

    print(f"\n✅ Sampling complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--checkpoint", type=str, default="outputs/flow_v2/checkpoints/best.pt")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512, 256])
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Sampling
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--ode-steps", type=int, default=200)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale (1.0 = no guidance)",
    )
    parser.add_argument("--mode", type=str, default="random", choices=["random", "grid"])

    # Condition ranges
    parser.add_argument("--use-data-range", action="store_true")
    parser.add_argument("--charge-range", type=float, nargs=2, default=[-20, 30])
    parser.add_argument("--hydro-range", type=float, nargs=2, default=[-4.5, 4.2])
    parser.add_argument("--length-range", type=float, nargs=2, default=[10, 100])

    # Data
    parser.add_argument("--metadata-path", type=str, default="data/embeddings/metadata.csv")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/samples_v2")

    args = parser.parse_args()
    main(args)
