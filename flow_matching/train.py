#!/usr/bin/env python3
"""
Conditional Flow Matching 训练脚本 - Phase 3

简洁实用，Linus 风格
"""

import argparse
from pathlib import Path
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ConditionalFlowModel
from dataset import get_dataloaders


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for z_1, c in pbar:
        z_1 = z_1.to(device)
        c = c.to(device)
        
        optimizer.zero_grad()
        loss = model.compute_loss(z_1, c)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    """评估一个 epoch"""
    model.eval()
    total_loss = 0
    
    for z_1, c in loader:
        z_1 = z_1.to(device)
        c = c.to(device)
        loss = model.compute_loss(z_1, c)
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main(args):
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / "logs")
    
    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 数据加载
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.latents_dir,
        args.metadata_path,
        args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # 模型
    model = ConditionalFlowModel(
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        hidden_dims=args.hidden_dims,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 20
    no_improve_count = 0
    
    print(f"\nStarting training for {args.epochs} epochs (early stop: patience={patience})...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # 验证
        val_loss = eval_epoch(model, val_loader, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 日志
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "latent_dim": args.latent_dim,
                "condition_dim": args.condition_dim,
                "hidden_dims": args.hidden_dims,
            }, ckpt_dir / "best.pt")
            print(f"✅ Saved best model (loss: {val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n🛑 Early stopping: no improvement for {patience} epochs")
                print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break
        
        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "latent_dim": args.latent_dim,
                "condition_dim": args.condition_dim,
                "hidden_dims": args.hidden_dims,
            }, ckpt_dir / f"epoch_{epoch}.pt")
    
    # 加载最佳模型
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best model loaded from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4f})")
    
    # 最终测试
    print("\nEvaluating on test set...")
    test_loss = eval_epoch(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), output_dir / "flow_model_final.pt")
    
    writer.close()
    print(f"\n✅ Training complete! Output saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument("--latents-dir", type=str, default="outputs/rae_eval")
    parser.add_argument("--metadata-path", type=str, default="data/embeddings/metadata.csv")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    
    # 模型
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    
    # 训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    
    # 其他
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/flow")
    
    args = parser.parse_args()
    main(args)

