#!/usr/bin/env python3
"""
Improved Conditional Flow Matching Training Script - V2

Key improvements:
1. Classifier-free guidance training
2. Learning rate warmup
3. Gradient clipping
4. Better validation metrics
5. Mixed precision training (optional)
"""

import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from flow_matching_v2.model import ImprovedFlowModel
from flow_matching.dataset import get_dataloaders


def train_epoch(model, loader, optimizer, device, scaler=None, drop_condition_prob=0.1):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for z_1, c in pbar:
        z_1 = z_1.to(device)
        c = c.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                loss = model.compute_loss(z_1, c, drop_condition_prob)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.compute_loss(z_1, c, drop_condition_prob)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0

    for z_1, c in loader:
        z_1 = z_1.to(device)
        c = c.to(device)
        loss = model.compute_loss(z_1, c)
        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(output_dir / "logs")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.latents_dir,
        args.metadata_path,
        args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = ImprovedFlowModel(
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        hidden_dims=args.hidden_dims,
        time_dim=args.time_dim,
        dropout=args.dropout,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return max(0.1, 0.5 * (1 + math.cos(math.pi * step / (args.epochs * len(train_loader)))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None

    # Training
    best_val_loss = float("inf")
    best_epoch = 0
    patience = args.patience
    no_improve_count = 0

    print(f"\nStarting training for {args.epochs} epochs (patience={patience})...")
    print(f"Classifier-free guidance: drop_prob={args.drop_condition_prob}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler, args.drop_condition_prob
        )

        # Update learning rate
        for _ in range(len(train_loader)):
            scheduler.step()

        # Validate
        val_loss = eval_epoch(model, val_loader, device)

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"LR: {current_lr:.6f}")

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": {
                        "latent_dim": args.latent_dim,
                        "condition_dim": args.condition_dim,
                        "hidden_dims": args.hidden_dims,
                        "time_dim": args.time_dim,
                        "dropout": args.dropout,
                    },
                },
                ckpt_dir / "best.pt",
            )
            print(f"✅ Saved best model (loss: {val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n🛑 Early stopping: no improvement for {patience} epochs")
                print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_dir / f"epoch_{epoch}.pt",
            )

    # Load best model
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Best model loaded from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4f})"
    )

    # Final test
    print("\nEvaluating on test set...")
    test_loss = eval_epoch(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), output_dir / "flow_model_v2_final.pt")

    writer.close()
    print(f"\n✅ Training complete! Output saved to {output_dir}/")


if __name__ == "__main__":
    import math

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--latents-dir", type=str, default="outputs/rae_eval")
    parser.add_argument("--metadata-path", type=str, default="data/embeddings/metadata.csv")
    parser.add_argument("--splits-dir", type=str, default="data/splits")

    # Model
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512, 256])
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument(
        "--drop-condition-prob",
        type=float,
        default=0.1,
        help="Probability to drop condition during training (for classifier-free guidance)",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")

    # Other
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/flow_v2")

    args = parser.parse_args()
    main(args)
