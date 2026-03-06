"""
RAE Loss with Triple Defense - Phase 4

三重防御机制：
1. Negative log-likelihood penalty: 重建序列与 UniProt 高同源时惩罚
2. Classifier-guided latent regularization: 使用分类器判断是否像人源蛋白
3. APOE push-away: 最大化与 APOE latent 的距离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class RAELossWithDefense:
    """
    RAE Loss with triple defense mechanisms
    """
    
    def __init__(
        self,
        lambda_z=0.01,
        lambda_orth=0.1,
        lambda_nll=10.0,
        lambda_classifier=1.0,
        lambda_apoe=5.0,
        apoe_latents_path=None,
        classifier_model=None,
        uniprot_identity_threshold=0.85
    ):
        """
        Args:
            lambda_z: latent L2 正则化系数
            lambda_orth: 正交正则化系数
            lambda_nll: NLL penalty 系数（与 UniProt 高同源时惩罚）
            lambda_classifier: classifier-guided 正则化系数
            lambda_apoe: APOE push-away 系数
            apoe_latents_path: APOE latent vectors 路径（.npy 或 .npz）
            classifier_model: 分类器模型（判断是否像人源蛋白）
            uniprot_identity_threshold: UniProt identity 阈值（>此值则惩罚）
        """
        self.lambda_z = lambda_z
        self.lambda_orth = lambda_orth
        self.lambda_nll = lambda_nll
        self.lambda_classifier = lambda_classifier
        self.lambda_apoe = lambda_apoe
        self.uniprot_identity_threshold = uniprot_identity_threshold
        
        # 加载 APOE latents（如果提供）
        self.apoe_latents = None
        if apoe_latents_path:
            apoe_latents_path = Path(apoe_latents_path)
            if apoe_latents_path.suffix == '.npz':
                data = np.load(apoe_latents_path)
                self.apoe_latents = torch.from_numpy(data['latents']).float()
            else:
                self.apoe_latents = torch.from_numpy(np.load(apoe_latents_path)).float()
            print(f"Loaded {len(self.apoe_latents)} APOE latents")
        
        self.classifier = classifier_model
        
    def compute_uniprot_identity(self, x_recon, x_original):
        """
        计算重建序列与原始序列的 identity（简化版，实际应该用 BLAST）
        
        这里用 cosine similarity 作为 proxy
        """
        # 归一化
        x_recon_norm = F.normalize(x_recon, p=2, dim=1)
        x_original_norm = F.normalize(x_original, p=2, dim=1)
        
        # Cosine similarity
        identity = (x_recon_norm * x_original_norm).sum(dim=1)
        return identity
    
    def compute_classifier_loss(self, z):
        """
        分类器引导的 latent 正则化
        
        如果 classifier(z) > threshold，说明像人源蛋白，需要惩罚
        """
        if self.classifier is None:
            return torch.tensor(0.0, device=z.device)
        
        # 分类器输出（假设输出 logits，>0 表示人源蛋白）
        logits = self.classifier(z)
        # 使用 sigmoid 得到概率
        prob_human = torch.sigmoid(logits)
        
        # 惩罚高概率（鼓励低概率，即不像人源蛋白）
        loss = prob_human.mean()
        return loss
    
    def compute_apoe_push_away_loss(self, z):
        """
        APOE push-away loss
        
        最大化与 APOE latent 的距离（cosine distance）
        """
        if self.apoe_latents is None:
            return torch.tensor(0.0, device=z.device)
        
        # 将 APOE latents 移到相同设备
        apoe_latents = self.apoe_latents.to(z.device)
        
        # 归一化
        z_norm = F.normalize(z, p=2, dim=1)  # (B, latent_dim)
        apoe_norm = F.normalize(apoe_latents, p=2, dim=1)  # (N_apoe, latent_dim)
        
        # 计算所有 pairwise cosine similarity
        # z_norm: (B, latent_dim), apoe_norm: (N_apoe, latent_dim)
        cosine_sim = z_norm @ apoe_norm.T  # (B, N_apoe)
        
        # 找到最大 similarity（最接近的 APOE latent）
        max_sim, _ = cosine_sim.max(dim=1)  # (B,)
        
        # Push-away: 最小化 max_sim（即最大化距离）
        # 使用 hinge loss: max(0, threshold - distance)
        # 这里 distance = 1 - similarity，所以 loss = max(0, similarity - threshold)
        threshold = 0.9
        loss = F.relu(max_sim - threshold).mean()
        
        return loss
    
    def compute_loss(self, model, x, labels=None):
        """
        计算总损失
        
        Args:
            model: RAE 模型
            x: 输入 embeddings (B, input_dim)
            labels: 标签 (B,) - 1=正样本, 0=负样本（可选，用于区分是否应用防御）
        
        Returns:
            loss_dict: 包含各项损失的字典
        """
        # Forward pass
        x_recon, z = model(x)
        
        # 基础损失
        mse_loss = F.mse_loss(x_recon, x)
        
        # Latent L2 正则化
        z_reg = (z ** 2).mean()
        
        # 正交正则化
        W = model.encoder.encoder[-1].weight  # (latent_dim, hidden_dim)
        WWT = W @ W.T  # (latent_dim, latent_dim)
        I = torch.eye(model.latent_dim, device=W.device)
        orth_loss = ((WWT - I) ** 2).mean()
        
        # 三重防御（仅对负样本或所有样本应用）
        nll_penalty = torch.tensor(0.0, device=x.device)
        classifier_loss = torch.tensor(0.0, device=x.device)
        apoe_loss = torch.tensor(0.0, device=x.device)
        
        # 如果提供了 labels，可以根据标签决定是否应用防御
        apply_defense = True
        if labels is not None:
            # 可以只对负样本应用防御，或对所有样本应用
            # 这里默认对所有样本应用
            apply_defense = True
        
        if apply_defense:
            # 1. NLL penalty: 如果重建序列与 UniProt 高同源，惩罚
            identity = self.compute_uniprot_identity(x_recon, x)
            high_identity_mask = identity > self.uniprot_identity_threshold
            if high_identity_mask.any():
                nll_penalty = self.lambda_nll * mse_loss * high_identity_mask.float().mean()
            
            # 2. Classifier-guided regularization
            classifier_loss = self.lambda_classifier * self.compute_classifier_loss(z)
            
            # 3. APOE push-away
            apoe_loss = self.lambda_apoe * self.compute_apoe_push_away_loss(z)
        
        # 总损失
        total_loss = (
            mse_loss +
            self.lambda_z * z_reg +
            self.lambda_orth * orth_loss +
            nll_penalty +
            classifier_loss +
            apoe_loss
        )
        
        return {
            "loss": total_loss,
            "mse": mse_loss.item(),
            "z_reg": z_reg.item(),
            "orth_loss": orth_loss.item(),
            "nll_penalty": nll_penalty.item(),
            "classifier_loss": classifier_loss.item(),
            "apoe_loss": apoe_loss.item(),
        }


def test_loss():
    """测试 loss"""
    from rae.model import RAE
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAE(input_dim=1536, latent_dim=64).to(device)
    
    # 创建 loss
    loss_fn = RAELossWithDefense(
        lambda_z=0.01,
        lambda_orth=0.1,
        lambda_nll=10.0,
        lambda_classifier=1.0,
        lambda_apoe=5.0
    )
    
    # 测试数据
    x = torch.randn(32, 1536).to(device)
    
    # 计算损失
    loss_dict = loss_fn.compute_loss(model, x)
    
    print("Loss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    test_loss()


