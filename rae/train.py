#!/usr/bin/env python3
"""
RAE 训练脚本 - Phase 2

训练 Representation Autoencoder:
- 输入: ESM-3 embeddings (1536 维)
- 输出: latent vectors (64 维)
- 目标: 重构误差 < 0.02, latent 空间平滑且可解释

设计原则:
1. 简单优于复杂 - 先跑通再优化
2. 监控关键指标 - MSE, latent norm, orthogonality
3. 保存 checkpoint - 每 5 epochs
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import RAE
from dataset import get_dataloaders
# 负样本与三重防御
from rae.dataset_with_negatives import get_dataloaders_with_negatives
from rae.loss_with_defense import RAELossWithDefense


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_z=0.01,
    lambda_orth=0.1,
    defense_loss: RAELossWithDefense | None = None,
):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_z_reg = 0
    total_orth = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        # 支持 (emb, label) 或单 emb
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            batch, labels = batch
            labels = labels.to(device)
        else:
            labels = None
        batch = batch.to(device)
        
        optimizer.zero_grad()
        if defense_loss is None:
            loss_dict = model.compute_loss(batch, lambda_z, lambda_orth)
        else:
            loss_dict = defense_loss.compute_loss(model, batch, labels)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += loss_dict["mse"]
        total_z_reg += loss_dict["z_reg"]
        total_orth += loss_dict["orth_loss"]
        
        pbar.set_postfix({"loss": loss.item(), "mse": loss_dict["mse"]})
    
    n = len(loader)
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "z_reg": total_z_reg / n,
        "orth_loss": total_orth / n,
    }


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    lambda_z=0.01,
    lambda_orth=0.1,
    defense_loss: RAELossWithDefense | None = None,
):
    """评估一个 epoch"""
    model.eval()
    total_loss = 0
    total_mse = 0
    total_z_reg = 0
    total_orth = 0
    
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            batch, labels = batch
            labels = labels.to(device)
        else:
            labels = None
        batch = batch.to(device)
        if defense_loss is None:
            loss_dict = model.compute_loss(batch, lambda_z, lambda_orth)
        else:
            loss_dict = defense_loss.compute_loss(model, batch, labels)
        
        total_loss += loss_dict["loss"].item()
        total_mse += loss_dict["mse"]
        total_z_reg += loss_dict["z_reg"]
        total_orth += loss_dict["orth_loss"]
    
    n = len(loader)
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "z_reg": total_z_reg / n,
        "orth_loss": total_orth / n,
    }


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
    if args.use_negatives:
        train_loader, val_loader, test_loader = get_dataloaders_with_negatives(
            positive_npz_path=args.npz_path,
            positive_splits_dir=args.splits_dir,
            negative_npz_paths=args.negative_npz_paths,
            negative_weights=args.negative_weights,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            args.npz_path,
            args.splits_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    
    # 模型
    model = RAE(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
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
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss
    defense_loss = None
    if args.use_defense:
        defense_loss = RAELossWithDefense(
            lambda_z=args.lambda_z,
            lambda_orth=args.lambda_orth,
            lambda_nll=args.lambda_nll,
            lambda_classifier=args.lambda_classifier,
            lambda_apoe=args.lambda_apoe,
            apoe_latents_path=args.apoe_latents_path,
            classifier_model=None,
            uniprot_identity_threshold=args.uniprot_identity_threshold,
        )

    # 训练
    best_val_mse = float('inf')
    best_epoch = 0
    patience = 20
    no_improve_count = 0
    
    print(f"\nStarting training for {args.epochs} epochs (early stop: patience={patience})...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda_z,
            args.lambda_orth,
            defense_loss,
        )
        
        # 验证
        val_metrics = eval_epoch(
            model,
            val_loader,
            device,
            args.lambda_z,
            args.lambda_orth,
            defense_loss,
        )
        
        # 学习率调度
        scheduler.step(val_metrics["mse"])
        
        # 日志
        print(f"Train - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}")
        
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for key, val in metrics.items():
                writer.add_scalar(f"{split}/{key}", val, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
        # 保存最佳模型
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_metrics["mse"],
            }, ckpt_dir / "best.pt")
            print(f"✅ Saved best model (MSE: {val_metrics['mse']:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n🛑 Early stopping: no improvement for {patience} epochs")
                print(f"Best val MSE: {best_val_mse:.4f} at epoch {best_epoch}")
                break
        
        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_metrics["mse"],
            }, ckpt_dir / f"epoch_{epoch}.pt")
    
    # 加载最佳模型
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best model loaded from epoch {checkpoint['epoch']} (val MSE: {checkpoint['val_mse']:.4f})")
    
    # 最终测试
    print("\nEvaluating on test set...")
    test_metrics = eval_epoch(model, test_loader, device, args.lambda_z, args.lambda_orth)
    print(f"Test - Loss: {test_metrics['loss']:.4f}, MSE: {test_metrics['mse']:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), output_dir / "rae_final.pt")
    
    writer.close()
    print(f"\n✅ Training complete! Output saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument("--npz-path", type=str, default="data/embeddings/esm3_embeddings.npz")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    
    # 模型
    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    
    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-z", type=float, default=0.01)
    parser.add_argument("--lambda-orth", type=float, default=0.1)
    # 负样本与防御
    parser.add_argument("--use-negatives", action="store_true", help="启用负样本混合训练")
    parser.add_argument(
        "--negative-npz-paths",
        type=str,
        nargs="+",
        default=[],
        help="负样本 embeddings npz 路径列表",
    )
    parser.add_argument(
        "--negative-weights",
        type=float,
        nargs="+",
        default=None,
        help="负样本权重，与 negative-npz-paths 对应；为空则全 1",
    )
    parser.add_argument("--use-defense", action="store_true", help="启用三重防御损失")
    parser.add_argument("--lambda-nll", type=float, default=10.0)
    parser.add_argument("--lambda-classifier", type=float, default=1.0)
    parser.add_argument("--lambda-apoe", type=float, default=5.0)
    parser.add_argument(
        "--apoe-latents-path",
        type=str,
        default=None,
        help="APOE latent 文件路径（.npy/.npz）用于 push-away",
    )
    parser.add_argument(
        "--uniprot-identity-threshold",
        type=float,
        default=0.85,
        help="重建同源性惩罚的 identity 阈值",
    )
    
    # 其他
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs/rae")
    
    args = parser.parse_args()
    main(args)

